from django.contrib import admin, messages
from django.core.exceptions import PermissionDenied
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.template.response import TemplateResponse
from django.urls import path, reverse
from django.utils.html import format_html

from . import review
from .importer import apply_run
from .models import ExtractedRecord, ImportChange, IngestionRun, PaperVersion


class AuditAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_readonly_fields(self, request, obj=None):
        return [field.name for field in self.model._meta.fields]


@admin.register(PaperVersion)
class PaperAdmin(AuditAdmin):
    list_display = ["arxiv_id", "version", "title", "existing_in_database", "document"]
    search_fields = ["arxiv_id", "title"]
    exclude = ["pdf"]

    def get_readonly_fields(self, request, obj=None):
        return [field for field in super().get_readonly_fields(request, obj) if field != "pdf"] + ["document"]

    def get_urls(self):
        return [path("<int:pk>/document/", self.admin_site.admin_view(self.document_view),
                     name="ingestion_paperversion_document")] + super().get_urls()

    def document(self, obj):
        if not obj.pdf:
            return "Not downloaded"
        return format_html('<a href="{}">Source PDF</a>', reverse("admin:ingestion_paperversion_document", args=[obj.pk]))

    def document_view(self, request, pk):
        obj = self.get_object(request, str(pk))
        if obj is None or not self.has_view_permission(request, obj):
            raise PermissionDenied
        return FileResponse(obj.pdf.open("rb"), content_type="application/pdf")


@admin.register(IngestionRun)
class RunAdmin(AuditAdmin):
    list_display = ["id", "paper", "status", "provider", "model", "started_at", "estimated_cost", "extractions", "review_link"]
    list_filter = ["status", "provider"]
    search_fields = ["paper__arxiv_id", "paper__title"]
    list_select_related = ["paper"]
    actions = ["validate_selected", "publish_selected"]

    def extractions(self, obj):
        url = reverse("admin:ingestion_extractedrecord_changelist") + f"?run__id__exact={obj.pk}"
        return format_html('<a href="{}">Records and evidence</a>', url)

    @admin.display(description="Review")
    def review_link(self, obj):
        return format_html('<a href="{}">Review</a>', reverse("admin:ingestion_review_run", args=[obj.pk]))

    def get_urls(self):
        view = self.admin_site.admin_view
        return [
            path("review/", view(self.review_list), name="ingestion_review"),
            path("review/refresh-prs/", view(self.refresh_prs), name="ingestion_refresh_prs"),
            path("review/<int:pk>/", view(self.review_run), name="ingestion_review_run"),
            path("review/<int:pk>/crop/<int:record>/<int:index>/", view(self.review_crop), name="ingestion_review_crop"),
        ] + super().get_urls()

    def review_list(self, request):
        if not self.has_view_permission(request):
            raise PermissionDenied
        runs = IngestionRun.objects.select_related("paper").order_by("-pk")
        context = {**self.admin_site.each_context(request), "title": "Review extractions",
                   "pending": runs.filter(status__in=[IngestionRun.Status.REVIEW, IngestionRun.Status.READY]),
                   "imported": runs.filter(status=IngestionRun.Status.IMPORTED)[:50],
                   "retries": runs.filter(status=IngestionRun.Status.SHORTLISTED, retry_of__isnull=False)}
        return TemplateResponse(request, "admin/ingestion/review_list.html", context)

    def refresh_prs(self, request):
        from .publication import configured, start_refresh
        if request.method != "POST" or not self.has_change_permission(request):
            raise PermissionDenied
        if not configured():
            self.message_user(request, "Record keeping is not configured (RECORDS_REPO)", messages.ERROR)
        else:
            start_refresh()
            self.message_user(request, "Refreshing the aggregate records PR in the background", messages.SUCCESS)
        return redirect("admin:ingestion_review")

    def review_run(self, request, pk):
        run = get_object_or_404(IngestionRun.objects.select_related("paper"), pk=pk)
        if not self.has_view_permission(request, run):
            raise PermissionDenied
        if request.method == "POST":
            if not self.has_change_permission(request, run):
                raise PermissionDenied
            self.review_action(request, run)
            return redirect(request.get_full_path())
        show = request.GET.get("show") if request.GET.get("show") in {"all", "uncertain"} else "flagged"
        context = {**self.admin_site.each_context(request), "title": f"Review run {run.pk}: {run.paper}",
                   **review.page_data(run, show)}
        context["editable"] = context["editable"] and self.has_change_permission(request, run)
        return TemplateResponse(request, "admin/ingestion/review_run.html", context)

    def review_action(self, request, run):
        actor, note, action = request.user.get_username(), request.POST.get("note", ""), request.POST.get("action")
        try:
            if action in {"approve", "approve_updates"}:
                problems = review.approve(run, actor, note, allow_updates=action == "approve_updates")
                run.refresh_from_db()
                if problems:
                    self.message_user(request, "Still blocked: " + "; ".join(problems), messages.ERROR)
                else:
                    self.message_user(request, f"Run {run.pk}: {run.status}", messages.SUCCESS)
            elif action == "reject":
                review.reject(run, actor, note)
                self.message_user(request, f"Run {run.pk} rejected", messages.SUCCESS)
            elif action == "reject_retry":
                retry = review.reject_and_retry(run, actor, note)
                self.message_user(request, f"Run {run.pk} rejected; retry queued as run {retry.pk} with your feedback "
                                           "(it is read first at the next ingestion run)", messages.SUCCESS)
            elif action in {"remove", "restore"}:
                record = get_object_or_404(ExtractedRecord, pk=request.POST.get("record"), run=run)
                if action == "remove":
                    review.remove(record, actor, note)
                else:
                    review.restore(record, actor)
                run.refresh_from_db()
                self.message_user(request, f"{'Removed' if action == 'remove' else 'Restored'} {record.key}; run is now "
                                           f"{run.get_status_display().lower()}" + (f": {run.error}" if run.error else ""),
                                  messages.SUCCESS)
            elif action == "accept":
                record = get_object_or_404(ExtractedRecord, pk=request.POST.get("record"), run=run)
                review.accept(record, request.POST.get("field", ""), actor, note)
                self.message_user(request, f"Accepted {record.key}.{request.POST.get('field')}", messages.SUCCESS)
            elif action == "correct":
                record = get_object_or_404(ExtractedRecord, pk=request.POST.get("record"), run=run)
                review.correct(record, request.POST.get("field", ""), request.POST.get("value", ""), actor, note)
                self.message_user(request, f"Corrected {record.key}.{request.POST.get('field')}", messages.SUCCESS)
            else:
                self.message_user(request, "Unknown action", messages.ERROR)
        except ValueError as exc:
            self.message_user(request, str(exc), messages.ERROR)

    def review_crop(self, request, pk, record, index):
        run = get_object_or_404(IngestionRun.objects.select_related("paper"), pk=pk)
        if not self.has_view_permission(request, run):
            raise PermissionDenied
        item = get_object_or_404(ExtractedRecord, pk=record, run=run)
        try:
            citation = item.evidence[index]["citation"] or {}
        except IndexError as exc:
            raise Http404 from exc
        if not citation.get("page") or not run.paper.pdf:
            raise Http404("No PDF citation")
        return HttpResponse(review.crop_png(run, citation), content_type="image/png")

    @admin.action(description="Validate selected runs without publishing", permissions=["change"])
    def validate_selected(self, request, queryset):
        self.process(request, queryset, publish=False)

    @admin.action(description="Publish validated additions (conflicts still require explicit review)", permissions=["change"])
    def publish_selected(self, request, queryset):
        self.process(request, queryset, publish=True)

    def process(self, request, queryset, publish):
        from .publication import publish as publish_run
        for run in queryset:
            actor = request.user.get_username()
            errors = publish_run(run, actor) if publish else apply_run(run, actor=actor)
            self.message_user(request, f"Run {run.pk}: " + ("; ".join(errors) if errors else run.status),
                              messages.ERROR if errors else messages.SUCCESS)


@admin.register(ExtractedRecord)
class RecordAdmin(AuditAdmin):
    list_display = ["id", "run", "kind", "key", "status", "reviewed_by"]
    list_filter = ["kind", "status"]
    search_fields = ["run__paper__arxiv_id", "key"]
    list_select_related = ["run__paper"]


@admin.register(ImportChange)
class ChangeAdmin(AuditAdmin):
    list_display = ["id", "model", "object_id", "record", "run", "origin", "actor", "created_at", "target"]
    list_filter = ["model", "actor"]
    list_select_related = ["record", "run"]

    def target(self, obj):
        app, model = obj.model.split(".")
        return format_html('<a href="{}">Open record</a>', reverse(f"admin:{app}_{model}_change", args=[obj.object_id]))

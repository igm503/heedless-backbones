from django.contrib import admin

from .models import (
    Category,
    Dataset,
    Task,
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    DownstreamHead,
    ClassificationResult,
    InstanceResult,
    FPSMeasurement,
    SemanticSegmentationResult,
)

class SourcedAdmin(admin.ModelAdmin):
    readonly_fields = ["source_record"]

    def save_model(self, request, obj, form, change):
        from ingestion.importer import snapshot
        request.ingestion_before = snapshot(self.model.objects.get(pk=obj.pk)) if change else None
        super().save_model(request, obj, form, change)

    def save_related(self, request, form, formsets, change):
        from ingestion.importer import snapshot
        from ingestion.models import ImportChange
        super().save_related(request, form, formsets, change)
        obj = form.instance
        if obj.source_record_id:
            after = snapshot(obj)
            if request.ingestion_before != after:
                ImportChange.objects.create(record=obj.source_record, run=obj.source_record.run, model=obj._meta.label_lower,
                                            object_id=obj.pk, before=request.ingestion_before, after=after,
                                            actor=request.user.get_username())


admin.site.register(Task)
for model in (Dataset, Backbone, DownstreamHead, ClassificationResult, InstanceResult,
              FPSMeasurement, SemanticSegmentationResult):
    admin.site.register(model, SourcedAdmin)


class CategorizedAdmin(SourcedAdmin):
    def formfield_for_dbfield(self, db_field, request, **kwargs):
        from django import forms
        from .categories import choices
        if db_field.name in {"model_type", "pretrain_method"}:
            return forms.ChoiceField(choices=[("", "---------")] + choices(db_field.name))
        return super().formfield_for_dbfield(db_field, request, **kwargs)


admin.site.register(BackboneFamily, CategorizedAdmin)
admin.site.register(PretrainedBackbone, CategorizedAdmin)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ["scope", "name", "approved_by", "approved_at"]
    readonly_fields = ["approved_by", "approved_at"]

    def has_change_permission(self, request, obj=None):
        return False

    def save_model(self, request, obj, form, change):
        obj.approved_by = request.user.get_username()
        super().save_model(request, obj, form, change)

from django.core.management.base import BaseCommand, CommandError

from ingestion.family_yaml import write_drafts
from ingestion.importer import apply_run, approve_run
from ingestion.models import IngestionRun
from .ingest_papers import run_lock


class Command(BaseCommand):
    help = ("List runs awaiting review, export one as an editable YAML draft, or re-validate it. "
            "Publishing and updates must be explicitly requested.")

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=int, nargs="?")
        parser.add_argument("--list", action="store_true", help="list runs that need review")
        parser.add_argument("--export", action="store_true", help="write family_data/review/<family>-run<id>.yml")
        parser.add_argument("--approve", action="store_true")
        parser.add_argument("--reject-retry", action="store_true",
                            help="reject the run and queue a retry that gets --note as feedback")
        parser.add_argument("--publish", action="store_true")
        parser.add_argument("--allow-updates", action="store_true")
        parser.add_argument("--actor", default="")
        parser.add_argument("--note", default="")

    def handle(self, *args, **options):
        if options["list"]:
            for run in IngestionRun.objects.filter(status=IngestionRun.Status.REVIEW).select_related("paper").order_by("pk"):
                self.stdout.write(f"{run.pk}: {run.paper.arxiv_id} {run.paper.title[:70]}\n    {run.error.splitlines()[0] if run.error else ''}")
            return
        if options["run_id"] is None:
            raise CommandError("Give a run ID, or --list")
        try:
            run = IngestionRun.objects.select_related("paper").get(pk=options["run_id"])
        except IngestionRun.DoesNotExist as exc:
            raise CommandError(str(exc)) from exc
        if options["reject_retry"]:
            from ingestion.review import reject_and_retry
            if not options["actor"] or not options["note"]:
                raise CommandError("--reject-retry requires --actor and --note")
            retry = reject_and_retry(run, options["actor"], options["note"])
            self.stdout.write(f"Run {run.pk} rejected; retry queued as run {retry.pk}")
            return
        if options["export"]:
            for path in write_drafts(run):
                self.stdout.write(self.style.SUCCESS(f"Draft written to {path}"))
            return
        if (options["approve"] or options["allow_updates"]) and (not options["actor"] or not options["note"]):
            raise CommandError("Review requires --actor and --note")
        if options["allow_updates"] and not options["approve"]:
            raise CommandError("--allow-updates requires --approve")
        with run_lock():
            try:
                if options["approve"]:
                    approve_run(run, options["actor"], options["note"])
                if options["publish"]:
                    from ingestion.publication import publish
                    problems = publish(run, options["actor"] or "automatic", allow_updates=options["allow_updates"])
                else:
                    problems = apply_run(run, actor=options["actor"] or "automatic", allow_updates=options["allow_updates"])
            except ValueError as exc:
                raise CommandError(str(exc)) from exc
            if problems:
                raise CommandError("; ".join(problems))
            self.stdout.write(f"Run {run.pk}: {run.status}" + (f" ({run.error})" if run.error else ""))

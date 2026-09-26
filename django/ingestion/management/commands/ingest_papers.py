import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone

from ingestion.agent import AgentRunner, screen
from ingestion.models import IngestionRun, PaperVersion
from ingestion.publication import publish_ready
from ingestion.pipeline import candidates, current_paper_ids, discover, shortlist
from ingestion.sources import Troller


@contextmanager
def run_lock():
    Path(settings.MEDIA_ROOT).mkdir(parents=True, exist_ok=True)
    with open(Path(settings.MEDIA_ROOT) / "ingestion.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CommandError("An ingestion job is already running") from exc
        acquired = False
        try:
            if connection.vendor == "postgresql":
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_try_advisory_lock(73410290)")
                    acquired = cursor.fetchone()[0]
                if not acquired:
                    raise CommandError("An ingestion job is already running on another host")
            yield
        finally:
            if acquired:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_unlock(73410290)")


class Command(BaseCommand):
    help = ("Discover papers, screen abstracts, then read shortlisted papers in full with Claude Code; "
            "publish only when --publish is supplied.")

    def add_arguments(self, parser):
        parser.add_argument("--paper-id", type=int, help="Process one saved paper snapshot, including an explicit retry")
        parser.add_argument("--limit", type=int, default=int(os.getenv("INGESTION_MAX_PAPERS", "3")),
                            help="full-paper reads (and saved publishes) per run")
        parser.add_argument("--screen-limit", type=int, default=int(os.getenv("INGESTION_MAX_SCREENS", "25")),
                            help="abstracts screened per run; 0 reads only the existing shortlist")
        parser.add_argument("--screen-batch", type=int, default=20, help="abstracts per screening session")
        parser.add_argument("--publish", action="store_true")
        parser.add_argument("--discover-only", action="store_true")
        parser.add_argument("--skip-discovery", action="store_true")
        parser.add_argument("--include-existing", action="store_true")
        parser.add_argument("--lookback-days", type=int, default=14)
        parser.add_argument("--model", default=os.getenv("INGESTION_MODEL"), help="Claude Code model for full reads")
        parser.add_argument("--screen-model", default=os.getenv("INGESTION_SCREEN_MODEL"),
                            help="Claude Code model for screening")
        parser.add_argument("--agent-timeout", type=int, default=3600, help="seconds per full read")

    def handle(self, *args, **options):
        if options["limit"] < 0 or options["screen_limit"] < 0 or options["lookback_days"] < 1:
            raise CommandError("Limits must not be negative")
        if options["discover_only"] and options["skip_discovery"]:
            raise CommandError("Cannot skip discovery in a discovery-only run")
        with run_lock():
            # The previous process cannot still be active while we own both locks.
            IngestionRun.objects.filter(status=IngestionRun.Status.RUNNING).update(
                status=IngestionRun.Status.FAILED, error="Previous process was interrupted", finished_at=timezone.now()
            )
            client = None
            if not options["skip_discovery"]:
                if not os.getenv("ARXIV_TROLLER_ACCOUNT"):
                    raise CommandError("ARXIV_TROLLER_ACCOUNT is not configured")
                client = Troller(os.getenv("ARXIV_TROLLER_URL", "https://arxiv-troller.com"),
                                 os.environ["ARXIV_TROLLER_ACCOUNT"])
                try:
                    client.login()
                    created, missing = discover(client, os.getenv("ARXIV_TROLLER_SOURCE_TAG", "backbones"),
                                                os.getenv("ARXIV_TROLLER_TAG", "heedless-backbones"), options["lookback_days"])
                except Exception as exc:
                    raise CommandError(f"Discovery failed: {exc}") from exc
                self.stdout.write(f"Queued {created} new paper snapshots; {len(missing)} existing papers absent from Troller.")
                if missing:
                    self.stdout.write("Missing arXiv IDs: " + ", ".join(missing))
            if options["discover_only"]:
                return
            remaining = options["limit"]
            # Reads only validate; clean runs are left ready and published below (or on the server).
            reader = AgentRunner(publish=False, model=options["model"], timeout=options["agent_timeout"])
            if options["paper_id"]:
                paper = PaperVersion.objects.filter(pk=options["paper_id"]).first()
                if paper is None:
                    raise CommandError("Unknown paper snapshot ID")
                for run in screen([paper], model=options["screen_model"]):
                    self.report(reader.read(run) if run.status == IngestionRun.Status.SHORTLISTED else run)
            else:
                for run in screen(candidates(options["screen_limit"], options["include_existing"]),
                                  batch=options["screen_batch"], model=options["screen_model"]):
                    self.report(run)
                for run in shortlist(remaining):
                    self.report(reader.read(run))
            if options["publish"]:
                published, blocked = publish_ready()
                self.stdout.write(f"Published {len(published)} ready runs" + (f"; {len(blocked)} blocked" if blocked else ""))
            if client and options["publish"]:
                client.sync(os.getenv("ARXIV_TROLLER_SOURCE_TAG", "backbones"),
                            os.getenv("ARXIV_TROLLER_TAG", "heedless-backbones"), sorted(current_paper_ids()))

    def report(self, run):
        self.stdout.write(f"{run.paper.arxiv_id}: {run.status} (run {run.pk})" + (f" — {run.error}" if run.error else ""))

from django.core.management.base import BaseCommand

from ingestion.publication import publish_ready


class Command(BaseCommand):
    help = ("Publish every validated run waiting to be published (the agent's clean runs), record them "
            "in git (one auto.<family> branch and PR each) and refresh the other open auto branches.")

    def add_arguments(self, parser):
        parser.add_argument("--actor", default="agent")

    def handle(self, *args, **options):
        published, blocked = publish_ready(options["actor"])
        for run in published:
            self.stdout.write(f"Published run {run.pk} ({run.paper.arxiv_id})")
        for run in blocked:
            self.stdout.write(f"Run {run.pk} now needs review: {run.error}")

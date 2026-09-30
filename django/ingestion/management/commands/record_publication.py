from django.core.management.base import BaseCommand

from ingestion.models import IngestionRun
from ingestion.publication import record


class Command(BaseCommand):
    help = "Append published runs to the aggregate records branch and pull request."

    def add_arguments(self, parser):
        parser.add_argument("run_ids", nargs="+", type=int)

    def handle(self, *args, **options):
        runs = list(IngestionRun.objects.filter(pk__in=options["run_ids"]))
        for outcome in record(runs):
            self.stdout.write(str(outcome))

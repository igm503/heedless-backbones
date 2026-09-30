from django.core.management.base import BaseCommand, CommandError

from ingestion.publication import refresh


class Command(BaseCommand):
    help = "Refresh the aggregate records PR with normal commits, merging main if needed."

    def add_arguments(self, parser):
        parser.add_argument("--include-legacy", action="store_true",
                            help="Include families from old auto.<family> PRs; leave those PRs open for review.")

    def handle(self, *args, **options):
        for outcome in refresh(include_legacy=options["include_legacy"]):
            if "error" in outcome:
                raise CommandError(outcome["error"])
            self.stdout.write(str(outcome))

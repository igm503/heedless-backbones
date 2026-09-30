from django.core.management.base import BaseCommand, CommandError

from ingestion.publication import refresh


class Command(BaseCommand):
    help = "Refresh the aggregate records PR with normal commits, merging main if needed."

    def handle(self, *args, **options):
        for outcome in refresh():
            if "error" in outcome:
                raise CommandError(outcome["error"])
            self.stdout.write(str(outcome))

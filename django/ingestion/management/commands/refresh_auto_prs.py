from django.core.management.base import BaseCommand

from ingestion.publication import refresh


class Command(BaseCommand):
    help = "Rebuild every open auto.<family> branch on the latest main (run after merging one)."

    def handle(self, *args, **options):
        for outcome in refresh():
            self.stdout.write(str(outcome))

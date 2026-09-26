from django.core.management.base import BaseCommand, CommandError

from ingestion.family_yaml import write_family_yaml
from ...models import BackboneFamily


class Command(BaseCommand):
    help = "Dumps model family information into family_data/<family>.yml"

    def add_arguments(self, parser):
        parser.add_argument("family", type=str, help="Name of the model family to dump, or 'all'")

    def handle(self, *args, **options):
        if options["family"] == "all":
            families = BackboneFamily.objects.order_by("name")
        else:
            families = BackboneFamily.objects.filter(name=options["family"])
            if not families:
                raise CommandError(f"Unknown family {options['family']}")
        for family in families:
            self.stdout.write(self.style.SUCCESS(f"YAML written to {write_family_yaml(family)}"))

from django.core.management.base import BaseCommand, CommandError

from ingestion.models import ExtractedRecord
from stats.categories import choices
from stats.models import Category


class Command(BaseCommand):
    help = "Approve a new model_type or pretrain_method, or list pending proposals."

    def add_arguments(self, parser):
        parser.add_argument("scope", nargs="?", choices=Category.Scope.values)
        parser.add_argument("name", nargs="?")
        parser.add_argument("--actor", default="")
        parser.add_argument("--note", default="", help="why the new category is warranted")
        parser.add_argument("--pending", action="store_true", help="list proposals that are not yet approved")

    def handle(self, *args, **options):
        if options["pending"]:
            for record in ExtractedRecord.objects.filter(kind="category").select_related("run__paper").order_by("pk"):
                scope, value = record.data.get("scope"), record.data.get("value")
                if scope in Category.Scope.values and value not in dict(choices(scope)):
                    self.stdout.write(f"{scope}: {value!r} (run {record.run_id}, {record.run.paper.url})")
            return
        if not (options["scope"] and options["name"] and options["name"].strip()):
            raise CommandError("Give a scope and name, or --pending")
        if not options["actor"] or not options["note"]:
            raise CommandError("Approval requires --actor and --note")
        name = options["name"].strip()
        if name in dict(choices(options["scope"])):
            raise CommandError(f"{name!r} is already an approved {options['scope']}")
        Category.objects.create(scope=options["scope"], name=name, approved_by=options["actor"], note=options["note"])
        self.stdout.write(self.style.SUCCESS(f"Approved {options['scope']} {name!r}"))

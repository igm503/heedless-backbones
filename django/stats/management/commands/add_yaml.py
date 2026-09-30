import getpass
from pathlib import Path

from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError

from ingestion.family_yaml import load, yaml_dir, yaml_to_entries
from ingestion.importer import ImportProblem, apply_entries, write_family_files
from ingestion.models import IngestionRun


class Command(BaseCommand):
    help = "Adds a model family YAML file to the database and refreshes its reference YAML"

    def add_arguments(self, parser):
        parser.add_argument("yaml_file", type=str, help="path, or name of a file in family_data/")
        parser.add_argument("--run", type=int, help="ingestion run this corrected file resolves")
        parser.add_argument("--actor", default=getpass.getuser())
        parser.add_argument("--allow-updates", action="store_true", help="permit changes to existing records")

    def handle(self, *args, **options):
        path = Path(options["yaml_file"])
        if not path.exists():
            path = yaml_dir() / (path.name if path.suffix == ".yml" else path.name + ".yml")
        if not path.exists():
            raise CommandError(f"No such file: {options['yaml_file']}")
        run = None
        if options["run"]:
            try:
                run = IngestionRun.objects.select_related("paper").get(pk=options["run"])
            except IngestionRun.DoesNotExist as exc:
                raise CommandError(f"Unknown run {options['run']}") from exc
        try:
            entries = yaml_to_entries(load(path))
            families = apply_entries(entries, actor=options["actor"], origin=str(path.resolve()), run=run,
                                     allow_updates=options["allow_updates"])
        except (ImportProblem, ValueError, ValidationError) as exc:
            raise CommandError(f"Nothing was imported: {exc}") from exc
        from ingestion.publication import configured, record
        if configured():
            # Recorded like every other publish: append to the aggregate records pull request.
            for outcome in record([run] if run else [], families=families):
                self.stdout.write(self.style.SUCCESS(f"Imported; recorded: {outcome}"))
            return
        written, errors = write_family_files(families)
        for written_path in written:
            self.stdout.write(self.style.SUCCESS(f"Imported; reference YAML: {written_path}"))
        if errors:
            raise CommandError("Imported, but YAML was not written: " + "; ".join(errors))

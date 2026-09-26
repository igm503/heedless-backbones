import json
from pathlib import Path

from django.core.management.base import BaseCommand

from ingestion.agent import validate


class Command(BaseCommand):
    help = "Check an agent's extraction.json exactly as the importer will; nothing is kept in the database."

    def add_arguments(self, parser):
        parser.add_argument("folder")
        parser.add_argument("--read-only", action="store_true", help="skip the rolled-back import check")

    def handle(self, *args, **options):
        report = validate(options["folder"], read_only=options["read_only"])
        (Path(options["folder"]) / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        if report["schema"]:
            self.stdout.write("Invalid extraction.json:\n" + "\n".join(f"- {item}" for item in report["schema"]))
            return
        lines = [f"{report['records']} records."]
        lines += [f"- {item['record']} ({item['kind']}): {item['problem']}" for item in report["problems"][:80]]
        if len(report["problems"]) > 80:
            lines.append(f"... and {len(report['problems']) - 80} more (see report.json)")
        if report["import"]:
            lines.append(f"- import check: {report['import']}")
        lines += [f"- uncertain: {item}" for item in report["uncertain"]]
        lines.append("PUBLISHABLE" if report["publishable"] else
                     "Not publishable yet: fix the problems above that are yours; genuine uncertainties go to review.")
        self.stdout.write("\n".join(lines))

import json
import re
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from ingestion.agent import load
from ingestion.web_sources import snapshot


class Command(BaseCommand):
    help = "Save one of the paper's official web sources into an agent working folder."

    def add_arguments(self, parser):
        parser.add_argument("folder")
        parser.add_argument("url")

    def handle(self, *args, **options):
        folder = Path(options["folder"])
        state = load(folder)
        if options["url"] in state["sources"]:
            raise CommandError(f"Already fetched: {options['url']}")
        try:
            item = snapshot(options["url"], state["pages"], state["links"], state["metadata"])
        except Exception as exc:
            raise CommandError(f"Not fetched: {exc}") from exc
        number = len(list((folder / "sources").glob("*.json"))) + 1
        name = f"{number:02d}-" + re.sub(r"[^a-z0-9]+", "-", options["url"].lower().split("://", 1)[-1])[:60].strip("-")
        (folder / "sources" / f"{name}.json").write_text(json.dumps(item, indent=1, ensure_ascii=False))
        (folder / "sources" / f"{name}.txt").write_text(f"SOURCE: {item['url']}\n\n{item['text']}")
        self.stdout.write(f"Saved sources/{name}.txt ({len(item['text']):,} characters, from {item['fetched_from']}). "
                          f"Cite it with url={item['url']!r} and page=null.")

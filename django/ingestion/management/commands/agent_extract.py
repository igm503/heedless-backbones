import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from ingestion.agent import prepare, run_agent, submit, validate


class Command(BaseCommand):
    help = ("Extract one arXiv paper with a local Claude Code session. Validates the result; "
            "submits it (and publishes) only when asked.")

    def add_arguments(self, parser):
        parser.add_argument("arxiv_id")
        parser.add_argument("--folder", help="working folder (default: INGESTION_STORAGE/agent/<id>-<time>)")
        parser.add_argument("--model")
        parser.add_argument("--timeout", type=int, default=3600)
        parser.add_argument("--submit", action="store_true", help="load the result into the database")
        parser.add_argument("--publish", action="store_true", help="with --submit: publish if it validates")

    def handle(self, *args, **options):
        if options["publish"] and not options["submit"]:
            raise CommandError("--publish requires --submit")
        folder = Path(options["folder"] or Path(settings.MEDIA_ROOT) / "agent" /
                      f"{options['arxiv_id'].replace('/', '_')}-{timezone.now():%Y%m%d-%H%M%S}")
        prepare(folder, options["arxiv_id"])
        self.stdout.write(f"Working folder: {folder}")
        agent = run_agent(folder, options["model"], options["timeout"])
        (folder / "agent.json").write_text(json.dumps(agent, indent=1, default=str))
        self.stdout.write(f"Agent finished in {agent['seconds']}s, {agent['turns']} turns"
                          + (f", ~${agent['cost_usd']:.2f} API-equivalent" if agent["cost_usd"] else "")
                          + (" (error)" if agent["is_error"] else "") + f"\n{agent['summary']}")
        report = validate(folder)
        self.stdout.write(f"Validation: {'publishable' if report['publishable'] else 'needs review'} "
                          f"({len(report['problems'])} problems, {len(report['uncertain'])} uncertain)")
        if options["submit"]:
            run = submit(folder, agent=agent, publish=options["publish"])
            self.stdout.write(f"Run {run.pk}: {run.status}" + (f" ({run.error})" if run.error else ""))

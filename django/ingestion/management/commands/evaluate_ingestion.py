from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from ingestion.evaluate import Evaluation, eval_papers, no_database_writes, score_eval


class Command(BaseCommand):
    help = ("Eval mode: fresh Claude Code sessions read the papers behind the earliest-added families; "
            "results go to files and are scored against the database. Never writes to the database.")

    def add_arguments(self, parser):
        parser.add_argument("--families", type=int, default=20, help="earliest-added families to cover")
        parser.add_argument("--paper", action="append", help="only this arXiv ID (repeatable)")
        parser.add_argument("--model", help="Claude Code model (default: its configured model)")
        parser.add_argument("--agent-timeout", type=int, default=3600, help="seconds per paper")
        parser.add_argument("--out", default=str(settings.EVAL_DIR), help="directory for eval results")
        parser.add_argument("--rescore", metavar="EVAL_DIR", help="re-score a saved eval; no model calls")

    def handle(self, *args, **options):
        with no_database_writes():
            if options["rescore"]:
                return self.report(options["rescore"], score_eval(options["rescore"]))
            papers, skipped = eval_papers(options["families"])
            if skipped:
                self.stdout.write("Skipped (no arXiv paper): " + ", ".join(skipped))
            if options["paper"]:
                unknown = set(options["paper"]) - set(papers)
                if unknown:
                    raise CommandError(f"Not an eval paper: {', '.join(sorted(unknown))}")
                papers = {identifier: papers[identifier] for identifier in options["paper"]}
            directory = Path(options["out"]) / f"{timezone.now():%Y%m%d-%H%M%S}-claude-code"
            directory.mkdir(parents=True)
            Evaluation(directory, model=options["model"], timeout=options["agent_timeout"]).run(papers)
            self.report(directory, score_eval(directory))

    def report(self, directory, summary):
        self.stdout.write((Path(directory) / "summary.md").read_text())
        self.stdout.write(self.style.SUCCESS(f"Results: {directory}"))

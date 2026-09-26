"""Publishing, and the records kept for every publish.

`publish()` is the one way extractions reach the database: the review page's approval, the
agent's clean runs (through `publish_ready`, called on the server at the end of each
scheduled run) and the admin actions all call it. After a successful import it records the
change in git, in the background: for each family the run touched, a branch `auto.<family>`
is rebuilt from the latest main with the family's YAML, a db.json regenerated from the
database, and the README's model table, then force-pushed with an open pull request.
Rebuilding (rather than merging) keeps every open auto branch mergeable: after one is merged,
`refresh()` rebuilds the others on the new main.
"""
import fcntl
import json
import logging
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db.models import Min, Q
from django.utils import timezone

from stats.models import (
    Backbone, BackboneFamily, ClassificationResult, FPSMeasurement, InstanceResult, PretrainedBackbone,
    SemanticSegmentationResult,
)
from .family_yaml import dump as write_yaml, family_to_dict
from .importer import apply_run, families_of
from .models import ImportChange, IngestionRun
from .sources import arxiv_id

logger = logging.getLogger(__name__)
MANAGE = Path(__file__).resolve().parents[1] / "manage.py"
BRANCH_PREFIX = "auto."
TABLE_HEADER = "| Model | Paper | Added |"


# Publishing -----------------------------------------------------------------------------------

def publish(run, actor, allow_updates=False, record=True):
    """Import a run and, if it was imported, record it in git (in the background)."""
    problems = apply_run(run, publish=True, actor=actor, allow_updates=allow_updates)
    run.refresh_from_db()
    if not problems and run.status == IngestionRun.Status.IMPORTED and record:
        start_recording([run])
    return problems


def publish_ready(actor="agent"):
    """Publish every validated run waiting to be published, then record them and refresh the
    open auto branches (run on the server after each scheduled agent run)."""
    published, blocked = [], []
    for run in IngestionRun.objects.filter(status=IngestionRun.Status.READY).order_by("pk"):
        problems = publish(run, actor, record=False)
        (blocked if problems else published).append(run)
    record(published, refresh_others=True)
    return published, blocked


def start_recording(runs):
    """Record runs in a separate process, so a publish (or a page request) never waits on git."""
    if not configured():
        note(runs, {"error": "Record keeping is not configured (RECORDS_REPO)"})
        return
    log = open(Path(settings.MEDIA_ROOT) / "records.log", "a")
    subprocess.Popen([sys.executable, str(MANAGE), "record_publication", *[str(run.pk) for run in runs],
                      f"--settings={settings.SETTINGS_MODULE}"],
                     stdout=log, stderr=log, start_new_session=True, cwd=MANAGE.parent)


def start_refresh():
    log = open(Path(settings.MEDIA_ROOT) / "records.log", "a")
    subprocess.Popen([sys.executable, str(MANAGE), "refresh_auto_prs", f"--settings={settings.SETTINGS_MODULE}"],
                     stdout=log, stderr=log, start_new_session=True, cwd=MANAGE.parent)


def configured():
    return bool(getattr(settings, "RECORDS_REPO", None))


def note(runs, entry):
    for run in runs:
        run.calls = [*run.calls, {"stage": "records", "at": timezone.now().isoformat(), **entry}]
        run.save(update_fields=["calls"])


# What a run changed ----------------------------------------------------------------------------

def run_families(run):
    """Names of the families whose stored records this run created or changed."""
    objects = []
    for change in ImportChange.objects.filter(Q(run=run) | Q(record__run=run)).exclude(model="ingestion.extractedrecord"):
        model = apps.get_model(change.model)
        obj = model.objects.filter(pk=change.object_id).first()
        if obj is not None:
            objects.append(obj)
    return set(families_of(objects))


def added_on(family):
    """When a family was first published (its creation in the change log), or None."""
    first = ImportChange.objects.filter(model="stats.backbonefamily", object_id=family.pk,
                                        before__isnull=True).aggregate(first=Min("created_at"))["first"]
    return first.date() if first else None


# Generated files ---------------------------------------------------------------------------------

def dump_database(names):
    """db.json for the given families plus all shared entities, in `dumpdata --indent 2` format.
    Links to extraction records are blanked: those tables are not part of the dump."""
    families = BackboneFamily.objects.filter(name__in=names)
    backbones = Backbone.objects.filter(family__in=families)
    pretrained = PretrainedBackbone.objects.filter(family__in=families)
    results = {model: model.objects.filter(pretrained_backbone__in=pretrained)
               for model in (ClassificationResult, InstanceResult, SemanticSegmentationResult)}
    fps = FPSMeasurement.objects.filter(Q(backbone__in=backbones) | Q(instanceresult__in=results[InstanceResult])
                                        | Q(semanticsegmentationresult__in=results[SemanticSegmentationResult])).distinct()
    chosen = {BackboneFamily: families, Backbone: backbones, PretrainedBackbone: pretrained, FPSMeasurement: fps, **results}
    objects = []
    for model in apps.get_app_config("stats").get_models():
        queryset = chosen.get(model, model.objects.all()).order_by("pk")
        for obj in queryset:
            if hasattr(obj, "source_record_id"):
                obj.source_record_id = None
            objects.append(obj)
    return serializers.serialize("json", objects, indent=2) + "\n"


def paper_cell(family):
    identifier = arxiv_id(family.paper)
    if identifier:
        return f"[arXiv {identifier}](https://arxiv.org/abs/{identifier})"
    link = family.paper or family.github
    return f"[paper]({link})" if link else ""


def update_readme(text, names):
    """Add model-table rows (and an Updates entry) for families in names that are not listed yet."""
    lines = text.splitlines()
    try:
        header = lines.index(TABLE_HEADER)
    except ValueError:
        raise ValueError(f"README has no model table ({TABLE_HEADER})")
    end = header + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    listed = {line.split("|")[1].strip() for line in lines[header + 2:end]}
    new = []
    for family in BackboneFamily.objects.filter(name__in=set(names) - listed).order_by("pk"):
        day = added_on(family) or timezone.now().date()
        new.append((day, family))
    if not new:
        return text
    rows = [f"| {family.name} | {paper_cell(family)} | {day.isoformat()} |" for day, family in new]
    lines[end:end] = rows
    try:
        updates = lines.index("## Updates")
        by_day = {}
        for day, family in new:
            by_day.setdefault(day, []).append(family.name)
        entries = [f"- {day.month}-{day.day}-{day.year}: added {', '.join(added)}"
                   for day, added in sorted(by_day.items(), reverse=True)]
        lines[updates + 2:updates + 2] = entries
    except ValueError:
        pass
    return "\n".join(lines) + "\n"


# Git and pull requests ---------------------------------------------------------------------------

def slug(name):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower().replace("+", "-plus"))).strip("-")


def branch_for(name):
    return BRANCH_PREFIX + slug(name)


@contextmanager
def records_lock():
    path = Path(settings.MEDIA_ROOT) / "records.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class RecordsRepo:
    """A clone of the repository used only for automatic records (never the site's checkout)."""

    def __init__(self, path, base="main"):
        self.path, self.base = Path(path), base

    def git(self, *args, check=True):
        result = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True)
        if check and result.returncode:
            raise RuntimeError(f"git {' '.join(args[:2])}: {result.stderr.strip() or result.stdout.strip()}")
        return result.stdout.strip()

    def gh(self, *args):
        result = subprocess.run(["gh", *args], capture_output=True, text=True, cwd=self.path)
        if result.returncode:
            raise RuntimeError(f"gh {' '.join(args[:2])}: {result.stderr.strip()}")
        return result.stdout.strip()

    def open_auto_branches(self):
        pulls = json.loads(self.gh("pr", "list", "--state", "open", "--base", self.base,
                                   "--json", "number,headRefName,url", "--limit", "200") or "[]")
        return {pull["headRefName"]: pull for pull in pulls if pull["headRefName"].startswith(BRANCH_PREFIX)}

    def build(self, name, open_pulls, actor_note=""):
        """Rebuild auto.<family> on the latest base; push and open or update its PR if it changed."""
        branch = branch_for(name)
        self.git("reset", "--hard", "-q")
        self.git("clean", "-fdq")
        self.git("checkout", "-q", "-B", branch, f"origin/{self.base}")
        data = self.path / "family_data"
        existed = (data / f"{name}.yml").exists()
        family = BackboneFamily.objects.filter(name=name).first()
        if family is None:
            return {"family": name, "skipped": "family no longer exists"}
        write_yaml(family_to_dict(family), data / f"{name}.yml")
        in_branch = {path.stem for path in data.glob("*.yml")}
        (self.path / "db.json").write_text(dump_database(in_branch))
        readme = self.path / "README.md"
        readme.write_text(update_readme(readme.read_text(), in_branch))
        self.git("add", f"family_data/{name}.yml", "db.json", "README.md")
        if not self.git("diff", "--cached", "--name-only"):
            return {"family": name, "skipped": f"{self.base} already has these records"}
        title = f"{'Update' if existed else 'Add'} {name}"
        self.git("commit", "-q", "-m", title)
        remote_exists = bool(self.git("ls-remote", "--heads", "origin", branch))
        if remote_exists:
            self.git("fetch", "-q", "origin", branch)
            if not self.git("diff", f"origin/{branch}", "HEAD", "--name-only", check=False):
                pull = open_pulls.get(branch)
                return {"family": name, "branch": branch, "url": pull and pull["url"], "unchanged": True}
        self.git("push", "-q", "--force", "origin", f"{branch}:{branch}")
        body = pull_body(family, existed, actor_note)
        pull = open_pulls.get(branch)
        if pull:
            self.gh("pr", "edit", str(pull["number"]), "--body", body)
            url = pull["url"]
        else:
            url = self.gh("pr", "create", "--base", self.base, "--head", branch, "--title", title, "--body", body)
        return {"family": name, "branch": branch, "url": url}


def pull_body(family, existed, actor_note):
    data = family_to_dict(family)
    backbones = data["backbones"]
    pretrained = [item for backbone in backbones for item in backbone["pretrained_backbones"]]
    count = lambda key: sum(len(item[key]) for item in pretrained)
    fps = sum(len(backbone["fps_measurements"]) for backbone in backbones)
    lines = [
        f"{'Updates' if existed else 'Adds'} **{family.name}** ({family.model_type}, "
        f"{'hierarchical' if family.hierarchical else 'isotropic'}), {paper_cell(family) or 'no paper link'}.",
        "",
        f"- {len(backbones)} backbones, {len(pretrained)} pretrained models",
        f"- {count('classification_results')} classification, {count('instance_results')} detection/instance "
        f"segmentation and {count('semantic_seg_results')} semantic segmentation results; {fps} throughput measurements",
        f"- Files regenerated from the database: `family_data/{family.name}.yml`, `db.json`, `README.md`",
    ]
    if actor_note:
        lines.append(f"- {actor_note}")
    return "\n".join(lines)


def record(runs, refresh_others=False, families=()):
    """Rebuild the auto branch of every family the runs touched (plus any families named, e.g.
    from add_yaml, and optionally every other open auto branch), noting the pull requests on
    each run."""
    if not configured():
        note(runs, {"error": "Record keeping is not configured (RECORDS_REPO)"})
        return []
    with records_lock():
        repo = RecordsRepo(settings.RECORDS_REPO, getattr(settings, "RECORDS_BASE", "main"))
        try:
            repo.git("fetch", "-q", "--prune", "origin")
            open_pulls = repo.open_auto_branches()
        except RuntimeError as exc:
            note(runs, {"error": str(exc)})
            return []
        touched = {name: [] for name in families}
        for run in runs:
            for name in run_families(run):
                touched.setdefault(name, []).append(run)
        outcomes = []
        for name, family_runs in sorted(touched.items()):
            summary = ("Published from run " + ", ".join(str(run.pk) for run in family_runs)) if family_runs \
                else "Imported from a family YAML file (add_yaml)"
            try:
                outcome = repo.build(name, open_pulls, summary)
            except RuntimeError as exc:
                outcome = {"family": name, "error": str(exc)}
            outcomes.append(outcome)
            note(family_runs, outcome)
        if refresh_others:
            by_branch = {branch_for(family.name): family.name for family in BackboneFamily.objects.all()}
            for branch in open_pulls:
                name = by_branch.get(branch)
                if name and name not in touched:
                    try:
                        outcomes.append(repo.build(name, open_pulls))
                    except RuntimeError as exc:
                        outcomes.append({"family": name, "error": str(exc)})
        repo.git("checkout", "-q", "--detach", check=False)
        return outcomes


def refresh():
    """Rebuild every open auto branch on the latest main (after one of them is merged)."""
    return record([], refresh_others=True)

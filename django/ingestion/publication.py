"""Publish extractions and accumulate their repository records in one automated PR.

Each batch keeps its history: later publications append commits, and changes to main are
merged with generated files refreshed from the database. A merged batch is followed by a
fresh branch and PR on the next publication. Production publishing precedes Git recording.
"""
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime
from uuid import uuid4
from pathlib import Path

import requests
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
BRANCH_PREFIX = "auto."  # Legacy per-family PRs, used only for explicit consolidation.
BATCH_PREFIX = "auto.records-"
LABEL = "automated"
FOOTER = "_Opened automatically by the Heedless Backbones ingestion pipeline._"
GITHUB_API = "https://api.github.com"
TABLE_HEADER = "| Model | Models | Paper | Added |"
LEGACY_TABLE_HEADER = "| Model | Paper | Added |"


# Publishing -----------------------------------------------------------------------------------

def publish(run, actor, allow_updates=False, record=True):
    """Import a run and, if it was imported, record it in git (in the background)."""
    problems = apply_run(run, publish=True, actor=actor, allow_updates=allow_updates)
    run.refresh_from_db()
    if not problems and run.status == IngestionRun.Status.IMPORTED and record:
        start_recording([run])
    return problems


def publish_ready(actor="agent"):
    """Publish validated runs and append their records to the aggregate PR."""
    published, blocked = [], []
    for run in IngestionRun.objects.filter(status=IngestionRun.Status.READY).order_by("pk"):
        problems = publish(run, actor, record=False)
        (blocked if problems else published).append(run)
    record(published)
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
    # Measurements attached to the families' models, plus any left unattached that name one of
    # their backbones (kept, so the dump does not silently drop stored data).
    fps = FPSMeasurement.objects.filter(
        Q(backbone__in=backbones) | Q(instanceresult__in=results[InstanceResult])
        | Q(semanticsegmentationresult__in=results[SemanticSegmentationResult])
        | Q(backbone__isnull=True, instanceresult__isnull=True, semanticsegmentationresult__isnull=True,
            backbone_name__in=backbones.values("name"))).distinct()
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


def batch_title(added, updated):
    """Describe distinct family changes relative to main (or the preceding commit)."""
    added, updated = sorted(set(added)), sorted(set(updated))
    if added and updated:
        noun = "family" if len(added) == 1 else "families"
        return f"Add {len(added)} backbone {noun}; update {len(updated)}"
    names, verb = (added, "Add") if added else (updated, "Update")
    if not names:
        return "Update backbone records"
    if len(names) > 3:
        return f"{verb} {len(names)} backbone families"
    if len(names) == 1:
        joined = names[0]
    elif len(names) == 2:
        joined = " and ".join(names)
    else:
        joined = ", ".join(names[:-1]) + ", and " + names[-1]
    return f"{verb} {joined}"


def model_table(lines):
    try:
        header = next(i for i, line in enumerate(lines) if line in {TABLE_HEADER, LEGACY_TABLE_HEADER})
    except StopIteration:
        raise ValueError(f"README has no model table ({TABLE_HEADER})")
    end = header + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return header, end


def sort_readme(text):
    """Order the model table and dated updates newest first, preserving handwritten notes."""
    lines = text.splitlines()
    header, end = model_table(lines)

    def row_key(line):
        cells = [cell.strip() for cell in line.split("|")]
        return (-date.fromisoformat(cells[-2]).toordinal(), cells[1].casefold(), cells[1])

    lines[header + 2:end] = sorted(lines[header + 2:end], key=row_key)
    if "## Updates" not in lines:
        return "\n".join(lines) + "\n"
    start = lines.index("## Updates") + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    dated, additions, positions = [], {}, set()
    for i in range(start, end):
        match = re.fullmatch(r"- (\d{1,2})-(\d{1,2})-(\d{4}): (.*)", lines[i])
        if not match:
            continue
        month, day, year = map(int, match.group(1, 2, 3))
        stamp = date(year, month, day)
        positions.add(i)
        # Only combine generated model-addition entries. Mixed handwritten announcements
        # (e.g. "added semantic segmentation; fixed plots") keep their original wording.
        added = re.fullmatch(r"added ([^;]+)", match[4])
        if added:
            additions.setdefault(stamp, set()).update(name.strip() for name in added[1].split(","))
        else:
            dated.append((stamp, lines[i]))
    for stamp, names in additions.items():
        names = ", ".join(sorted(names, key=lambda name: (name.casefold(), name)))
        dated.append((stamp, f"- {stamp.month}-{stamp.day}-{stamp.year}: added {names}"))
    ordered = iter(line for stamp, line in sorted(dated, key=lambda item: item[0], reverse=True))
    result = []
    for i, line in enumerate(lines):
        if i in positions:
            replacement = next(ordered, None)
            if replacement is not None:
                result.append(replacement)
        else:
            result.append(line)
    return "\n".join(result) + "\n"


def update_model_counts(text, records):
    """Count distinct backbone variants from the exact fixture recorded alongside the README."""
    families = {obj["pk"]: obj["fields"] for obj in records if obj["model"] == "stats.backbonefamily"}
    backbones = {obj["pk"]: obj["fields"] for obj in records if obj["model"] == "stats.backbone"}
    totals = Counter(backbone["family"] for backbone in backbones.values())
    by_name = {family["name"]: totals[pk] for pk, family in families.items()}
    by_paper = defaultdict(set)
    for obj in records:
        if obj["model"] != "stats.pretrainedbackbone":
            continue
        fields = obj["fields"]
        paper = fields.get("paper") or families.get(fields["family"], {}).get("paper")
        if paper and fields["backbone"] in backbones:
            by_paper[arxiv_id(paper) or paper].add(fields["backbone"])

    lines = text.splitlines()
    header, end = model_table(lines)
    columns = [cell.strip() for cell in lines[header].split("|")[1:-1]]
    rows = []
    for line in lines[header + 2:end]:
        row = dict(zip(columns, (cell.strip() for cell in line.split("|")[1:-1])))
        count = by_name.get(row["Model"])
        if count is None:
            # Historical paper-specific rows such as FAN STL belong to an existing family.
            # Count the distinct underlying variants with checkpoints from that paper.
            link = re.search(r"\]\(([^)]+)\)", row["Paper"])
            paper = link[1] if link else ""
            variants = by_paper.get(arxiv_id(paper) or paper)
            count = len(variants) if variants else "—"
        rows.append(f"| {row['Model']} | {count} | {row['Paper']} | {row['Added']} |")
    lines[header:end] = [TABLE_HEADER, "|---|---:|---|---|", *rows]
    return "\n".join(lines) + "\n"


def listed_dates(readme):
    lines = readme.splitlines()
    header, end = model_table(lines)
    dates = {}
    for line in lines[header + 2:end]:
        cells = [cell.strip() for cell in line.split("|")]
        try:
            dates[cells[1]] = date.fromisoformat(cells[-2])
        except (IndexError, ValueError):
            continue
    return dates


def unlisted(readme, names, known_dates=None):
    """(day added, family) for families in names that the README's model table does not list yet."""
    lines = readme.splitlines()
    header, end = model_table(lines)
    listed = {line.split("|")[1].strip() for line in lines[header + 2:end]}
    return [(added_on(family) or (known_dates or {}).get(family.name) or timezone.now().date(), family)
            for family in BackboneFamily.objects.filter(name__in=set(names) - listed).order_by("pk")]


def update_readme(text, names, known_dates=None, records=None):
    """Add new families, refresh all variant counts, and maintain chronological ordering."""
    new = unlisted(text, names, known_dates)
    lines = text.splitlines()
    header, end = model_table(lines)
    columns = [cell.strip() for cell in lines[header].split("|")[1:-1]]
    rows = []
    for day, family in new:
        row = {"Model": family.name, "Models": "—", "Paper": paper_cell(family), "Added": day.isoformat()}
        rows.append("| " + " | ".join(row[column] for column in columns) + " |")
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
    if records is None:
        records = json.loads(dump_database(names))
    return sort_readme(update_model_counts("\n".join(lines) + "\n", records))


ABOUT = Path("django/stats/templates/stats/about.html")
UPDATES_START = '<div class="drawerContents show"'


def update_about(text, new):
    """Add "Added <family>" to the about page's Latest Updates, under each family's day."""
    lines = text.splitlines()
    try:
        heading = next(i for i, line in enumerate(lines) if ">Latest Updates<" in line)
        start = next(i for i in range(heading, len(lines)) if lines[i].lstrip().startswith(UPDATES_START))
    except StopIteration:
        raise ValueError("about.html has no Latest Updates section")
    by_day = {}
    for day, family in new:
        if f'{{% url "family" "{family.name}" %}}' not in text:
            by_day.setdefault(day, []).append(family.name)
    for day, names in sorted(by_day.items()):  # oldest first, so the newest ends up on top
        label = f"{day:%B} {day.day}, {day.year}"
        items = [f'            <li style="margin-bottom: 0.3rem;">Added <a href="{{% url "family" "{name}" %}}">{name}</a></li>'
                 for name in names]
        heading = f'          <p style="margin-bottom: 0.3rem; font-weight: 600;">{label}</p>'
        if heading in lines:
            close = next(i for i in range(lines.index(heading), len(lines)) if lines[i].strip() == "</ul>")
            lines[close:close] = items
        else:
            lines[start + 1:start + 1] = [
                f"        <!-- {label} -->",
                '        <div style="margin-bottom: 1rem;">',
                heading,
                '          <ul class="list-disc" style="padding-left: 1.5rem; margin-top: 0; margin-bottom: 0.5rem;">',
                *items,
                "          </ul>",
                "        </div>",
            ]
    return sort_about_dates("\n".join(lines) + "\n")


def sort_about_dates(text):
    """Sort existing dated update blocks too, retaining their handwritten contents."""
    heading = text.index(">Latest Updates<")
    pattern = re.compile(
        r"(?m)^        <!-- ([A-Za-z]+ \d{1,2}, \d{4}) -->\n"
        r"        <div[^\n]*>\n.*?^        </div>(?:\n|$)", re.DOTALL)
    # Scope the reorder to the Latest Updates drawer, not subsequent page sections.
    start = text.index(UPDATES_START, heading)
    end = re.search(r"(?m)^      </div>", text[start:])
    if end is None:
        raise ValueError("about.html has no closing Latest Updates drawer")
    end = start + end.end()
    section = text[start:end]
    blocks = list(pattern.finditer(section))
    ordered = iter(sorted(blocks, key=lambda m: datetime.strptime(m[1], "%B %d, %Y"), reverse=True))
    section = pattern.sub(lambda _: next(ordered)[0], section)
    return text[:start] + section + text[end:]



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


class GitHubApp:
    """The GitHub App the records are pushed and proposed as (e.g. heedless-backbones-agent[bot]):
    a one-hour installation token for git and gh, and the bot's commit identity."""

    def __init__(self, app_id, key_path, repo_slug, session=requests):
        self.app_id, self.key, self.repo, self.session = str(app_id), Path(key_path).read_text(), repo_slug, session
        self._token, self._expires, self._identity = None, 0, None

    def jwt(self):
        import jwt
        now = int(time.time())
        return jwt.encode({"iat": now - 60, "exp": now + 540, "iss": self.app_id}, self.key, algorithm="RS256")

    def call(self, method, path, auth=True):
        headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {self.jwt()}"} if auth else {})}
        response = self.session.request(method, GITHUB_API + path, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def token(self):
        if self._token is None or time.time() > self._expires - 300:
            installation = self.call("GET", f"/repos/{self.repo}/installation")
            self._token = self.call("POST", f"/app/installations/{installation['id']}/access_tokens")["token"]
            self._expires = time.time() + 3600
        return self._token

    def identity(self):
        if self._identity is None:
            login = self.call("GET", "/app")["slug"] + "[bot]"
            user = self.call("GET", f"/users/{login}", auth=False)
            self._identity = (login, f"{user['id']}+{login}@users.noreply.github.com")
        return self._identity


def github_app(path):
    """The configured GitHub App for the records clone at path, or None (then git and gh use
    whatever login the server has)."""
    app_id, key = getattr(settings, "GITHUB_APP_ID", None), getattr(settings, "GITHUB_APP_KEY", None)
    if not (app_id and key):
        return None
    remote = subprocess.run(["git", "-C", str(path), "remote", "get-url", "origin"],
                            capture_output=True, text=True, check=True).stdout.strip()
    slug = re.sub(r"\.git$", "", re.split(r"github\.com[:/]", remote)[-1])
    return GitHubApp(app_id, key, slug)


class RecordsRepo:
    """A clone of the repository used only for automatic records (never the site's checkout)."""

    def __init__(self, path, base="main", app=None):
        self.path, self.base, self.app = Path(path), base, app
        self.labelled = False

    def env(self):
        env = dict(os.environ)
        if self.app:
            env["GH_TOKEN"] = self.app.token()
        return env

    def git(self, *args, check=True):
        options = []
        if self.app:
            # Authenticate as the app (its token in GH_TOKEN) and commit as its bot user.
            name, email = self.app.identity()
            options = ["-c", "credential.helper=",
                       "-c", "credential.helper=!f() { echo username=x-access-token; echo \"password=$GH_TOKEN\"; }; f",
                       "-c", f"user.name={name}", "-c", f"user.email={email}"]
        result = subprocess.run(["git", *options, "-C", str(self.path), *args], capture_output=True, text=True,
                                env=self.env())
        if check and result.returncode:
            raise RuntimeError(f"git {' '.join(args[:2])}: {result.stderr.strip() or result.stdout.strip()}")
        return result.stdout.strip()

    def gh(self, *args):
        result = subprocess.run(["gh", *args], capture_output=True, text=True, cwd=self.path, env=self.env())
        if result.returncode:
            raise RuntimeError(f"gh {' '.join(args[:2])}: {result.stderr.strip()}")
        return result.stdout.strip()

    def ensure_label(self):
        if not self.labelled:
            self.gh("label", "create", LABEL, "--force", "--color", "BFD4F2",
                    "--description", "Opened by the ingestion pipeline")
            self.labelled = True

    def open_auto_branches(self):
        pulls = json.loads(self.gh("pr", "list", "--state", "open", "--base", self.base,
                                   "--json", "number,headRefName,url", "--limit", "200") or "[]")
        return {pull["headRefName"]: pull for pull in pulls if pull["headRefName"].startswith(BRANCH_PREFIX)}

    def family_changes(self, base, target):
        paths = self.git("diff", "--name-only", "-z", base, target, "--", "family_data").split("\0")
        return {Path(path).stem for path in paths if path.endswith(".yml")}

    def family_names(self, ref):
        paths = self.git("ls-tree", "-r", "--name-only", "-z", ref, "--", "family_data").split("\0")
        return {Path(path).stem for path in paths if path.endswith(".yml")}

    def build(self, names, open_pulls):
        """Append to the open batch, resolving only machine-generated merge conflicts."""
        batches = [pull for branch, pull in open_pulls.items() if branch.startswith(BATCH_PREFIX)]
        if len(batches) > 1:
            raise RuntimeError("More than one aggregate PR is open; resolve the duplicate batches first")
        pull = batches[0] if batches else None
        if not pull and not names:
            return {"skipped": "No pending aggregate PR"}
        branch = pull["headRefName"] if pull else BATCH_PREFIX + uuid4().hex[:12]
        base = f"origin/{self.base}"
        self.git("reset", "--hard", "-q")
        self.git("clean", "-fdq")
        self.git("checkout", "-q", "-B", branch, f"origin/{branch}" if pull else base)
        previous = self.git("rev-parse", "HEAD")
        names = set(names)
        if pull:
            ancestor = self.git("merge-base", base, "HEAD")
            names.update(self.family_changes(ancestor, "HEAD"))
        families = {family.name: family for family in BackboneFamily.objects.filter(name__in=names)}
        if names - families.keys():
            raise RuntimeError("Pending families missing from the database: " + ", ".join(sorted(names - families.keys())))
        generated = {"db.json", "README.md", str(ABOUT)}
        generated.update(f"family_data/{name}.yml" for name in names)
        try:
            self.git("merge", "--no-commit", "--no-ff", base)
        except RuntimeError:
            conflicts = set(filter(None, self.git("diff", "--name-only", "-z", "--diff-filter=U").split("\0")))
            if not conflicts or conflicts - generated:
                self.git("merge", "--abort", check=False)
                raise
        known_dates = listed_dates(self.git("show", f"{previous}:README.md"))
        # Rebuild shared output from main plus all pending families, so each batch has one
        # complete snapshot. These paths are machine-managed; unrelated edits survive merges.
        for path in ("README.md", str(ABOUT)):
            (self.path / path).write_text(self.git("show", f"{base}:{path}") + "\n")
        data = self.path / "family_data"
        for name, family in sorted(families.items()):
            write_yaml(family_to_dict(family), data / f"{name}.yml")
        in_branch = {path.stem for path in data.glob("*.yml")}
        snapshot = dump_database(in_branch)
        (self.path / "db.json").write_text(snapshot)
        readme, about = self.path / "README.md", self.path / ABOUT
        new = unlisted(readme.read_text(), in_branch, known_dates)
        about.write_text(update_about(about.read_text(), new))
        readme.write_text(update_readme(readme.read_text(), in_branch, known_dates, records=json.loads(snapshot)))
        self.git("add", "--", *sorted(generated))
        base_names = self.family_names(base)
        pending = self.family_changes(base, "--cached")
        title = batch_title(pending - base_names, pending & base_names)
        merging = bool(self.git("rev-parse", "--verify", "-q", "MERGE_HEAD", check=False))
        changed = bool(self.git("diff", "--cached", "--name-only"))
        if changed or merging:
            changed_families = self.family_changes("HEAD", "--cached")
            previous_names = self.family_names("HEAD")
            commit_title = batch_title(changed_families - previous_names, changed_families & previous_names)
            if merging:
                commit_title = f"Merge {self.base} into automated backbone records"
            self.git("commit", "-q", "-m", commit_title)
        # A repeat publication after a merge must not open an empty replacement PR.
        if not pull and not self.git("diff", "--name-only", base, "HEAD"):
            return {"skipped": f"{self.base} already has these records"}
        self.git("push", "-q", "origin", f"{branch}:{branch}")
        body = batch_body(pending - base_names, pending & base_names)
        self.ensure_label()
        if pull:
            self.gh("pr", "edit", str(pull["number"]), "--title", title, "--body", body, "--add-label", LABEL)
            url = pull["url"]
        else:
            url = self.gh("pr", "create", "--base", self.base, "--head", branch, "--title", title, "--body", body,
                          "--label", LABEL)
        outcome = {"families": sorted(pending), "branch": branch, "url": url}
        if self.git("rev-parse", "HEAD") == previous:
            outcome["unchanged"] = True
        return outcome


def batch_body(added, updated):
    lines = ["Records the following published backbone families in one batch.", "",
             "| Change | Family | Paper |", "|---|---|---|"]
    for label, names in (("Add", added), ("Update", updated)):
        for family in BackboneFamily.objects.filter(name__in=names).order_by("name"):
            lines.append(f"| {label} | {family.name} | {paper_cell(family)} |")
    lines += ["", "Includes each family's YAML, one database snapshot, and the README/About updates.",
              "Dates reflect first publication to the database. Later publications append commits to this PR.",
              "", FOOTER]
    return "\n".join(lines)


def record(runs, families=(), include_legacy=False):
    """Append published families to one PR; optionally consolidate legacy per-family PRs."""
    runs = list(runs)
    if not configured():
        note(runs, {"error": "Record keeping is not configured (RECORDS_REPO)"})
        return []
    with records_lock():
        repo = RecordsRepo(settings.RECORDS_REPO, getattr(settings, "RECORDS_BASE", "main"),
                           app=github_app(settings.RECORDS_REPO))
        try:
            repo.git("fetch", "-q", "--prune", "origin")
            open_pulls = repo.open_auto_branches()
            names = set(families)
            for run in runs:
                names.update(run_families(run))
            if include_legacy:
                by_branch = {branch_for(family.name): family.name for family in BackboneFamily.objects.all()}
                legacy = {branch for branch in open_pulls if not branch.startswith(BATCH_PREFIX)}
                unknown = legacy - by_branch.keys()
                if unknown:
                    raise RuntimeError("Cannot consolidate unknown family branches: " + ", ".join(sorted(unknown)))
                names.update(by_branch[branch] for branch in legacy)
            outcome = repo.build(names, open_pulls)
        except (RuntimeError, ValueError) as exc:
            outcome = {"error": str(exc)}
        finally:
            if repo.git("rev-parse", "--verify", "-q", "MERGE_HEAD", check=False):
                repo.git("merge", "--abort", check=False)
            repo.git("checkout", "-q", "--detach", check=False)
        note(runs, outcome)
        return [outcome]


def refresh(include_legacy=False):
    """Update the current aggregate PR; migration from family PRs is explicit."""
    return record([], include_legacy=include_legacy)

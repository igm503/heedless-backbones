"""The extraction engine: local Claude Code sessions, on the user's claude.ai login.

Screening: batches of abstracts go to `claude -p` with no tools and a JSON schema.
Full reads: for each shortlisted paper the orchestrator prepares a working folder (PDF,
page text, metadata, the prompt and a small `hb` command wrapper), runs `claude -p`
confined to that folder, then validates the agent's extraction.json itself and submits
it through the normal importer. The agent can read the database (lookup), fetch the
paper's official web sources (fetch) and validate its draft (validate); it cannot write
to the database.

API credentials are removed from the sessions' environment, so Claude Code uses the
claude.ai login (subscription) rather than billing an API key.
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import jsonschema
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from . import prompts
from .evidence import validate_evidence
from .importer import REGISTRY, apply_run, check_category
from .models import ExtractedRecord, IngestionRun, PaperVersion, WebSource
from .pdf_text import pdf_links
from .pipeline import code_version, parse_records
from .schema import EXTRACTION, SCREENING
from .sources import fetch_metadata, fetch_pdf

MANAGE = Path(__file__).resolve().parents[1] / "manage.py"
TOOLS = ["Read", "Glob", "Grep", "Write(./**)", "Edit(./**)",
         "Bash(./hb validate *)", "Bash(./hb validate)", "Bash(./hb lookup *)", "Bash(./hb fetch *)"]
DENIED = ["WebFetch", "WebSearch", "Agent", "Task"]
FINAL = "extraction.json"
# Credentials that would make Claude Code bill an API account instead of the claude.ai login.
API_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def claude_env(**extra):
    env = {key: value for key, value in os.environ.items() if key not in API_CREDENTIALS}
    return {**env, **extra}


class Rollback(Exception):
    pass


# Working folders ---------------------------------------------------------------------

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def prepare(folder, identifier, families_hidden=(), read_only=False, feedback="", previous=None):
    """Download the paper and write everything the agent needs into folder. For a retry, feedback
    is the reviewer's feedback and previous the rejected extraction's records."""
    folder = Path(folder)
    (folder / "pages").mkdir(parents=True, exist_ok=True)
    (folder / "sources").mkdir(exist_ok=True)
    metadata = fetch_metadata(identifier)
    version, data, pages = fetch_pdf(identifier)
    (folder / "paper.pdf").write_bytes(data)
    write_json(folder / "metadata.json", {**metadata, "version": version})
    write_json(folder / "pages.json", pages)
    for number, text in enumerate(pages, 1):
        (folder / "pages" / f"page-{number:03d}.txt").write_text(text)
    write_json(folder / "links.json", pdf_links(data))
    write_json(folder / "hidden.json", list(families_hidden))
    write_wrapper(folder, read_only)
    if previous is not None:
        write_json(folder / "previous-extraction.json", previous)
    (folder / "prompt.md").write_text(agent_prompt(identifier, len(pages), feedback))
    return folder


def write_wrapper(folder, read_only):
    """./hb: the only commands the agent may run."""
    env = f"HB_READ_ONLY=1 " if read_only else ""
    python, manage, module = shlex.quote(sys.executable), shlex.quote(str(MANAGE)), shlex.quote(settings.SETTINGS_MODULE)
    here = shlex.quote(str(Path(folder).resolve()))
    script = f"""#!/bin/sh
# Commands available to the extraction agent (see prompt.md).
command="$1"; shift 2>/dev/null
case "$command" in
  validate) exec env {env}{python} {manage} validate_extraction {here} --settings={module} "$@" ;;
  lookup) exec env {env}{python} {manage} lookup --work {here} --settings={module} "$@" ;;
  fetch) exec env {env}{python} {manage} fetch_source {here} --settings={module} "$@" ;;
  *) echo "usage: ./hb validate | ./hb lookup <names|family|model|search> ... | ./hb fetch <url>"; exit 2 ;;
esac
"""
    path = Path(folder) / "hb"
    path.write_text(script)
    path.chmod(0o755)


def load(folder):
    folder = Path(folder)
    sources = {}
    for path in sorted((folder / "sources").glob("*.json")):
        item = read_json(path)
        sources[item["url"]] = item
    return {"pages": read_json(folder / "pages.json", []), "metadata": read_json(folder / "metadata.json", {}),
            "links": read_json(folder / "links.json", []), "sources": sources,
            "extraction": read_json(folder / FINAL)}


# Prompt --------------------------------------------------------------------------------

def feedback_block(feedback):
    if not feedback:
        return ""
    return f"""
REVIEWER FEEDBACK. An earlier extraction of this paper was rejected by the maintainer. Address
every point below. previous-extraction.json holds the rejected records (with their review
status) for reference only: re-check everything against the paper rather than copying it.

{feedback}
"""


def agent_prompt(identifier, page_count, feedback=""):
    return prompts.CRITERIA + feedback_block(feedback) + f"""
You are extracting benchmark records for arXiv {identifier} into the Heedless Backbones
database. Work in the current folder:

- paper.pdf: the paper ({page_count} pages). Read table pages visually (Read with pages)
  to understand their layout.
- pages/page-NNN.txt: the text of each page. Citations are checked against this text, so
  copy quotations from these files exactly (superscripts appear as ^, e.g. 224^2).
- metadata.json: arXiv metadata. links.json: hyperlinks on the first two pages.

Commands (run exactly as shown; nothing else is available):
- ./hb lookup names: existing families, backbones, pretrained backbones, datasets, heads,
  categories and tasks (use these exact names).
- ./hb lookup family "<name>" / ./hb lookup model "<pretrained backbone>": what is stored,
  with each result's settings and source, to see which results already exist.
- ./hb lookup search "<text>": find existing names.
- ./hb fetch <url>: save one of the paper's official sources (its repository linked in
  the paper, that repository's files and releases, or its project page). The text is
  written to sources/*.txt; quote from there and cite with url set and page null.
- ./hb validate: check extraction.json exactly as the importer will. Fix every problem it
  reports that is yours; leave only genuine uncertainties.

Task:
1. Decide whether the paper qualifies (full paper, not just the abstract).
2. Extract the paper's own models and results following the guide. Where the paper
   lacks a value for its own models, check the official sources.
3. With ./hb lookup, find results the paper reports for other models that are already in
   the database but lack a result with the same settings, and add those (see the guide).
4. Write extraction.json: {{"decision": ..., "records": [...]}} in the output format
   below, then run ./hb validate and fix problems until it reports the extraction is
   publishable, or only genuine uncertainties remain.
5. Finish with a short summary: what you extracted, judgment calls, and anything left
   uncertain. Do not ask questions; nobody will answer them.
""" + prompts.guide_block() + prompts.RULES + """
Web citations: {"page": null, "url": "<the fetched url>", "location": "...", "quote": "..."}.

The JSON schema of extraction.json:
""" + json.dumps(EXTRACTION) + "\n\nField reference:\n" + json.dumps(prompts.field_spec(REGISTRY), ensure_ascii=False)


# Running the agent ----------------------------------------------------------------------

def agent_settings(folder):
    path = Path(folder) / ".claude" / "settings.json"
    path.parent.mkdir(exist_ok=True)
    write_json(path, {"permissions": {"allow": TOOLS, "deny": DENIED}})
    return path


def model_used(result, fallback=None):
    """The model Claude Code ran (the one with the most output in modelUsage), else the fallback."""
    usage = result.get("modelUsage") or {}
    if usage:
        return max(usage, key=lambda name: usage[name].get("outputTokens") or 0)
    return fallback


def add_usage(run, call, share=1):
    """Add a Claude Code call's tokens and cost to the run (a screening batch is split evenly)."""
    usage = call.get("usage") or {}
    inputs = sum(usage.get(key) or 0 for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    run.input_tokens += round(inputs / share)
    run.output_tokens += round((usage.get("output_tokens") or 0) / share)
    if call.get("cost_usd") is not None:
        cost = Decimal(str(call["cost_usd"])) / share
        run.estimated_cost = ((run.estimated_cost or 0) + cost).quantize(Decimal("0.000001"))


def run_agent(folder, model=None, timeout=3600, claude="claude"):
    """Run Claude Code in folder; returns the result event (cost, turns, errors) and log path."""
    folder = Path(folder)
    agent_settings(folder)
    command = [claude, "-p", (folder / "prompt.md").read_text(), "--output-format", "stream-json", "--verbose",
               "--permission-mode", "dontAsk", "--allowedTools", *TOOLS, "--disallowedTools", *DENIED,
               "--no-session-persistence"]
    if model:
        command += ["--model", model]
    log_path = folder / "transcript.jsonl"
    started = time.monotonic()
    result, auth, started_model = {}, None, None
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env=claude_env(HB_WORK=str(folder)))
        try:
            deadline = started + timeout
            for line in process.stdout:
                log.write(line)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("subtype") == "init":
                    auth, started_model = event.get("apiKeySource"), event.get("model")
                if event.get("type") == "result":
                    result = event
                if time.monotonic() > deadline:
                    process.kill()
                    result = {"is_error": True, "result": f"Stopped after {timeout}s"}
                    break
        finally:
            process.wait()
    return {"seconds": round(time.monotonic() - started, 1), "transcript": str(log_path),
            "is_error": result.get("is_error", True), "summary": result.get("result", "no result event"),
            "cost_usd": result.get("total_cost_usd"), "turns": result.get("num_turns"),
            "usage": result.get("usage"), "model": model_used(result, started_model or model), "denials": result.get("permission_denials"),
            # "none" means the claude.ai login; anything else is an API credential being billed.
            "auth": auth}


# Screening ---------------------------------------------------------------------------------

def screen_batch(papers, model=None, timeout=900, claude="claude"):
    """Abstract decisions for up to ~20 PaperVersions in one tool-less Claude Code call."""
    listing = [{"arxiv_id": paper.arxiv_id, "title": paper.title, "abstract": paper.abstract} for paper in papers]
    prompt = prompts.screening(listing)
    command = [claude, "-p", prompt, "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--json-schema", json.dumps(SCREENING)]
    if model:
        command += ["--model", model]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=claude_env())
    output = completed.stdout[completed.stdout.find("{"):] if "{" in completed.stdout else ""
    try:
        result = json.loads(output)
    except ValueError as exc:
        raise ValueError(f"Screening returned no JSON: {completed.stdout[-500:]} {completed.stderr[-500:]}") from exc
    if result.get("is_error"):
        raise ValueError(f"Screening failed: {result.get('result')}")
    decisions = (result.get("structured_output") or {}).get("decisions", [])
    jsonschema.validate({"decisions": decisions}, SCREENING)
    call = {"stage": "abstract", "batch": [paper.arxiv_id for paper in papers], "seconds": round(time.monotonic() - started, 1),
            "cost_usd": result.get("total_cost_usd"), "usage": result.get("usage"), "session": result.get("session_id"),
            "model": model_used(result, model)}
    return {decision["arxiv_id"]: decision for decision in decisions}, call


def screen(papers, batch=20, model=None):
    """Create a screened IngestionRun per paper: shortlisted for a full read, or rejected."""
    runs = []
    papers = list(papers)
    for start in range(0, len(papers), batch):
        group = papers[start:start + batch]
        try:
            decisions, call = screen_batch(group, model)
            error = ""
        except (ValueError, subprocess.SubprocessError, jsonschema.ValidationError) as exc:
            decisions, call, error = {}, {"stage": "abstract", "error": str(exc)}, f"{type(exc).__name__}: {exc}"
        for paper in group:
            decision = decisions.get(paper.arxiv_id)
            run = IngestionRun(
                paper=paper, provider="claude-code", model=call.get("model") or model or "", prompt_version=prompts.VERSION,
                code_version=code_version(), calls=[{**call, "decision": decision}], finished_at=timezone.now(),
                decision={**(decision or {}), "evidence": []},
                status=(IngestionRun.Status.FAILED if decision is None else
                        IngestionRun.Status.SHORTLISTED if decision["qualifies"] else IngestionRun.Status.REJECTED),
                error=error or ("" if decision else "No screening decision returned for this paper"))
            add_usage(run, call, share=len(group))
            run.save()
            runs.append(run)
    return runs


# Validation and submission ----------------------------------------------------------------

def record_view(record):
    return SimpleNamespace(reviewed_by="", **{key: record.get(key) for key in
                                              ("kind", "key", "data", "evidence", "uncertain", "inferred", "note")})


def validate(folder, read_only=False):
    """Everything the importer would check, reported per record. The database is only
    written inside a transaction that is always rolled back (or not at all if read_only)."""
    state = load(folder)
    report = {"publishable": False, "schema": [], "problems": [], "import": None, "uncertain": [], "records": 0}
    output = state["extraction"]
    if output is None:
        report["schema"].append(f"{FINAL} not found")
        return report
    report["schema"] = [f"{'/'.join(map(str, error.path)) or '(root)'}: {error.message}"
                        for error in jsonschema.Draft202012Validator(EXTRACTION).iter_errors(output)][:25]
    if report["schema"]:
        return report
    try:
        records = parse_records(output)
    except ValueError as exc:
        report["schema"].append(str(exc))
        return report
    report["records"] = len(records)
    sources = {url: item["text"] for url, item in state["sources"].items()}
    for record in records:
        try:
            validate_evidence(record_view(record), state["pages"], sources)
            if record["kind"] == "category":
                check_category(SimpleNamespace(data=record["data"]))
        except (ValueError, KeyError, TypeError) as exc:
            report["problems"].append({"record": record["key"], "kind": record["kind"], "problem": str(exc)})
        report["uncertain"] += [f"{record['key']}.{item['field']}: {item['reason']}" for item in record["uncertain"]]
    if not read_only and not os.environ.get("HB_READ_ONLY"):
        try:
            with transaction.atomic():
                run = build_run(folder, state, output, records, store_pdf=False)
                problems = apply_run(run)
                report["import"] = problems[0] if problems else None
                raise Rollback
        except Rollback:
            pass
    report["publishable"] = not (report["problems"] or report["import"] or report["uncertain"])
    return report


def paper_version(state):
    metadata = state["metadata"]
    paper, _ = PaperVersion.objects.get_or_create(
        arxiv_id=metadata["arxiv_id"], revision=metadata["updated"],
        defaults={"title": metadata["title"], "abstract": metadata["abstract"],
                  "metadata": {k: v for k, v in metadata.items() if k != "version"}})
    return paper


def build_run(folder, state, output, records, run=None, agent=None, store_pdf=True):
    """Create (or fill in) the IngestionRun, its web sources and extracted records. Validation
    passes store_pdf=False: a file saved to storage would outlive the rolled-back transaction."""
    folder = Path(folder)
    if run is None:
        paper = paper_version(state)
        run = IngestionRun.objects.create(paper=paper, provider="claude-code", model=(agent or {}).get("model") or "",
                                          prompt_version=prompts.VERSION, code_version=code_version())
    paper = run.paper
    if not paper.pages:
        data = (folder / "paper.pdf").read_bytes()
        paper.version = state["metadata"].get("version")
        paper.pages = state["pages"]
        paper.sha256 = hashlib.sha256(data).hexdigest()
        if store_pdf:
            paper.pdf.save(f"{paper.arxiv_id.replace('/', '_')}v{paper.version}-{paper.sha256[:12]}.pdf",
                           ContentFile(data), save=False)
        paper.save(update_fields=["version", "pages", "sha256", "pdf"])
    run.decision = output["decision"]
    if agent:
        run.calls = [*run.calls, {"stage": "agent", **agent}]
    run.save(update_fields=["decision", "calls"])
    for item in state["sources"].values():
        WebSource.objects.create(run=run, url=item["url"], fetched_from=item["fetched_from"],
                                 fetched_at=item["fetched_at"], sha256=item["sha256"], text=item["text"])
    for record in records:
        ExtractedRecord.objects.create(run=run, **record)
    return run


def submit(folder, run=None, agent=None, publish=False, actor="agent"):
    """Load the agent's output into the database and apply it like any other extraction."""
    state = load(folder)
    output = state["extraction"]
    if output is None:
        raise ValueError(f"The agent did not write {FINAL}")
    jsonschema.validate(output, EXTRACTION)
    records = parse_records(output)
    with transaction.atomic():
        run = build_run(folder, state, output, records, run=run, agent=agent)
    if not output["decision"].get("qualifies"):
        run.status = IngestionRun.Status.REJECTED
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "finished_at"])
        return run
    if publish:
        from .publication import publish as publish_run
        publish_run(run, actor)
    else:
        apply_run(run, actor=actor)  # Validation only: a clean run is left ready to publish.
    run.finished_at = timezone.now()
    run.save(update_fields=["finished_at"])
    return run


class AgentRunner:
    """Full reads by Claude Code for shortlisted runs."""

    def __init__(self, *, publish=False, model=None, timeout=3600, root=None):
        self.publish, self.model, self.timeout = publish, model, timeout
        self.root = Path(root or Path(settings.MEDIA_ROOT) / "agent")

    def folder_for(self, run):
        return self.root / f"{run.paper.arxiv_id.replace('/', '_')}-run{run.pk}"

    def read(self, run):
        folder = self.folder_for(run)
        run.status = IngestionRun.Status.RUNNING
        run.provider = "claude-code"
        run.save(update_fields=["status", "provider"])
        try:
            feedback, previous = "", None
            if run.retry_of_id:
                from .review import retry_feedback
                feedback = retry_feedback(run)
                previous = [{"key": r.key, "kind": r.kind, "data": r.data, "status": r.status, "review_note": r.review_note}
                            for r in run.retry_of.records.order_by("pk")]
            prepare(folder, run.paper.arxiv_id, feedback=feedback, previous=previous)
            agent = run_agent(folder, self.model, self.timeout)
            run.model = agent.get("model") or "claude-code"
            add_usage(run, agent)
            run.save(update_fields=["model", "input_tokens", "output_tokens", "estimated_cost"])
            return submit(folder, run=run, agent=agent, publish=self.publish)
        except Exception as exc:
            run.status, run.error = IngestionRun.Status.FAILED, f"{type(exc).__name__}: {exc}"
            run.finished_at = timezone.now()
            run.save(update_fields=["status", "error", "finished_at"])
            return run

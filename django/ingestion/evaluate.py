"""Eval mode: the agent's full reads of papers already in the database, written to files
and scored against the stored records. The database is only read (vocabulary and the
reference families); any write during an eval raises. Each eval is a directory:

    evals/<timestamp>-claude-code/
        config.json            model, timeout, prompt/code version, papers
        summary.md / .json     per-paper and overall scores, tokens and cost
        papers/<arXiv ID>/
            agent/             the agent's working folder: prompt, page text, sources,
                               extraction.json, report.json, transcript.jsonl
            agent.json         turns, time, tokens, API-equivalent cost, auth source
            paper.pdf, pages.json, metadata.json, decision.json
            records.json       extracted records
            reference.yml      the paper's families as stored in the database
            score.json         matches, recall/precision, accuracy and each wrong value
"""
import json
import math
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import yaml
from django.db import connection
from django.utils import timezone

from stats.models import BackboneFamily, PretrainedBackbone
from . import prompts
from .evidence import validate_evidence
from .family_yaml import CLASSIFICATION_FIELDS, INSTANCE_FIELDS, SEMANTIC_FIELDS, family_to_dict
from .importer import vocabulary
from .pipeline import code_version, parse_records
from .sources import arxiv_id


class DatabaseWrite(RuntimeError):
    pass


@contextmanager
def no_database_writes():
    """Reject every statement except reads for the duration of an eval."""
    def guard(execute, sql, params, many, context):
        if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise DatabaseWrite(f"Eval mode never writes to the database: {sql[:80]}")
        return execute(sql, params, many, context)
    with connection.execute_wrapper(guard):
        yield


def eval_papers(count=20):
    """The papers behind the earliest-added families, with all families each paper defines."""
    papers, skipped = defaultdict(list), []
    for family in BackboneFamily.objects.order_by("pk")[:count]:
        identifier = arxiv_id(family.paper)
        if identifier:
            papers[identifier].append(family)
        else:
            skipped.append(family.name)
    return dict(papers), skipped


def hidden_vocabulary(families):
    """The prompt vocabulary without the paper's own family, backbone and pretrained names."""
    words = vocabulary()
    names = {family.name for family in families}
    backbones = set(BackboneFamily.objects.filter(name__in=names).values_list("backbone__name", flat=True))
    pretrained = set(BackboneFamily.objects.filter(name__in=names).values_list("pretrainedbackbone__name", flat=True))
    for kind, hidden in [("family", names), ("backbone", backbones), ("pretrained_backbone", pretrained)]:
        words[kind] = [name for name in words[kind] if name not in hidden]
    return words


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


class Evaluation:
    """Full reads by fresh Claude Code sessions, scored against the database. The sessions are
    read-only: their validator skips the import check, and lookups cannot write."""

    def __init__(self, directory, *, model=None, timeout=3600):
        self.directory, self.model, self.timeout = Path(directory), model, timeout

    def run(self, papers):
        write_json(self.directory / "config.json", {
            "created": timezone.now().isoformat(), "provider": "claude-code", "model": self.model or "default",
            "timeout": self.timeout, "prompt_version": prompts.VERSION, "code_version": code_version(),
            "papers": {identifier: [family.name for family in families] for identifier, families in papers.items()},
        })
        for identifier, families in papers.items():
            self.run_paper(identifier, families)

    def run_paper(self, identifier, families):
        from . import agent
        folder = self.directory / "papers" / identifier.replace("/", "_")
        work = folder / "agent"
        work.mkdir(parents=True, exist_ok=True)
        with open(folder / "reference.yml", "w") as f:
            yaml.dump([family_to_dict(family) for family in families], f, sort_keys=False, allow_unicode=True)
        agent.prepare(work, identifier, families_hidden=[family.name for family in families], read_only=True)
        for name in ("paper.pdf", "pages.json", "metadata.json"):
            (folder / name).write_bytes((work / name).read_bytes())
        result = agent.run_agent(work, self.model, self.timeout)
        usage = result.get("usage") or {}
        write_json(folder / "agent.json", {**result, "input_tokens": usage.get("input_tokens", 0)
                                           + usage.get("cache_read_input_tokens", 0)
                                           + usage.get("cache_creation_input_tokens", 0),
                                           "output_tokens": usage.get("output_tokens", 0)})
        output = agent.read_json(work / agent.FINAL)
        records = []
        if output is not None:
            try:
                records = parse_records(output)
            except (ValueError, KeyError) as exc:
                write_json(folder / "records.error.json", {"error": str(exc)})
            write_json(folder / "decision.json", output.get("decision"))
        write_json(folder / "records.json", records)


# Scoring ---------------------------------------------------------------------

METRICS = {"top_1", "top_5", "gflops", "mAP", "AP50", "AP75", "mAPs", "mAPm", "mAPl", "ms_m_iou",
           "ms_pixel_accuracy", "ms_mean_accuracy", "ss_m_iou", "ss_pixel_accuracy", "ss_mean_accuracy", "fps"}
# Fields that identify an experiment for matching; all other fields are scored.
CORE = {"classification": ["dataset", "resolution"], "instance": ["head", "dataset", "instance_type"],
        "semantic": ["head", "dataset"], "fps": ["gpu", "resolution"]}
FIELDS = {
    "family": ["model_type", "hierarchical", "spiking", "pretrain_method"],
    "backbone": ["m_parameters"],
    "pretrained_backbone": ["pretrain_dataset", "pretrain_method", "pretrain_resolution", "pretrain_epochs"],
    "classification": CLASSIFICATION_FIELDS, "instance": INSTANCE_FIELDS, "semantic": SEMANTIC_FIELDS,
    "fps": ["resolution", "gpu", "precision", "fps", "batch_size"],
}
RESULT_KINDS = ["classification", "instance", "semantic"]
# Left blank by the guide when the paper does not state them, even where stored data has a value.
BLANK_ALLOWED = {("fps", "precision")}


def family_values(data):
    # spiking is written only when true; an omitted flag is false.
    return {f: data.get(f) for f in FIELDS["family"]} | {"spiking": bool(data.get("spiking"))}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold()) if value is not None else None


def decimals(number):
    text = repr(float(number)).rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def same(expected, actual):
    """Equal, or a more precise extracted value that rounds to the stored one (28.6 vs 29)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, rel_tol=1e-3, abs_tol=0.051) or \
            math.isclose(round(actual, decimals(expected)), expected, abs_tol=1e-9)
    if expected in (None, "") or actual in (None, ""):
        return expected in (None, "") and actual in (None, "")
    return norm(expected) == norm(actual)


def truth_items(families, identifier=None):
    """The stored records, each marked "own" when this paper is its source. Records linked
    to another paper or a repository still count as matches, but not toward recall."""
    def from_paper(item):
        link = item.get("paper") or item.get("source")
        return identifier is None or not link or arxiv_id(link) == identifier

    def add(item, own):
        items.append({**item, "own": own})

    items = []
    for family in families:
        data = family_to_dict(family)
        add({"kind": "family", "name": data["name"], **family_values(data)}, True)
        for backbone in data["backbones"]:
            backbone_own = from_paper(backbone)
            add({"kind": "backbone", "name": backbone["name"], "m_parameters": backbone.get("m_parameters")}, backbone_own)
            for fps in backbone["fps_measurements"]:
                add({"kind": "fps", "owner": norm(backbone["name"]), **fps}, backbone_own and from_paper(fps))
            for pretrained in backbone["pretrained_backbones"]:
                pretrained_own = backbone_own and from_paper(pretrained)
                pb = (norm(backbone["name"]), norm(pretrained.get("pretrain_dataset")), norm(pretrained.get("pretrain_method")))
                add({"kind": "pretrained_backbone", "pb": pb, "name": pretrained["name"],
                     **{f: pretrained.get(f) for f in FIELDS["pretrained_backbone"]}}, pretrained_own)
                for key, kind in [("classification_results", "classification"), ("instance_results", "instance"),
                                  ("semantic_seg_results", "semantic")]:
                    for result in pretrained[key]:
                        add({"kind": kind, "pb": pb, **{f: result.get(f) for f in FIELDS[kind]}},
                            pretrained_own and from_paper(result))
    return items


def extracted_items(records):
    by_key = {record["key"]: record for record in records}

    def target(value):
        return by_key.get(value[1:]) if isinstance(value, str) and value.startswith("$") else None

    def name(value):
        record = target(value)
        return record["data"].get("name") if record else value

    pb_of = {record["key"]: (norm(name(record["data"].get("backbone"))), norm(name(record["data"].get("pretrain_dataset"))),
                             norm(record["data"].get("pretrain_method")))
             for record in records if record["kind"] == "pretrained_backbone"}
    items = []
    for record in records:
        data, kind = record["data"], record["kind"]
        if kind == "family":
            items.append({"kind": kind, "name": data.get("name"), **family_values(data)})
        elif kind == "backbone":
            items.append({"kind": kind, "name": data.get("name"), **{f: data.get(f) for f in FIELDS[kind]}})
        elif kind == "pretrained_backbone":
            items.append({"kind": kind, "pb": pb_of[record["key"]], "name": data.get("name"),
                          **{f: name(data.get(f)) for f in FIELDS[kind]}})
        elif kind in RESULT_KINDS:
            ref = data.get("pretrained_backbone")
            pb = pb_of.get(ref[1:]) if isinstance(ref, str) and ref.startswith("$") else stored_identity(ref)
            items.append({"kind": kind, "pb": pb, **{f: name(data.get(f)) for f in FIELDS[kind]}})
        elif kind == "fps":
            owner = target(data.get("owner"))
            if owner is not None and owner["kind"] == "backbone":  # Downstream throughput is not scored.
                items.append({"kind": kind, "owner": norm(owner["data"].get("name")), **{f: data.get(f) for f in FIELDS[kind]}})
    return items


def stored_identity(name):
    """(backbone, pretraining dataset, method) of an existing pretrained backbone named in a
    result (results the paper reports for models already in the database)."""
    stored = PretrainedBackbone.objects.filter(name=name).select_related("backbone", "pretrain_dataset").first() \
        if isinstance(name, str) else None
    if stored is None:
        return None
    return (norm(stored.backbone.name), norm(stored.pretrain_dataset.name), norm(stored.pretrain_method))


def identity(item):
    kind = item["kind"]
    if kind in ("family", "backbone"):
        return (norm(item["name"]),)
    if kind == "pretrained_backbone":
        return item["pb"]
    if kind == "fps":
        return (item["owner"], *(norm(item.get(f)) for f in CORE["fps"]))
    return (item["pb"], *(norm(item.get(f)) for f in CORE[kind]))


def score(truth, extracted):
    """Greedy matching per identity; per-kind counts, field accuracy and each wrong value."""
    pools = defaultdict(list)
    for item in extracted:
        pools[(item["kind"], identity(item))].append(item)
    # "truth" and "found" count records sourced from this paper (recall); "matched" counts
    # every extracted record matching a stored one, from any source (precision).
    report = {kind: {"truth": 0, "found": 0, "extracted": 0, "matched": 0, "fields": 0, "correct": 0,
                     "metric_fields": 0, "metric_correct": 0, "errors": [], "missed": []} for kind in FIELDS}
    for item in extracted:
        report[item["kind"]]["extracted"] += 1
    # Match this paper's own records first, so they are not taken by duplicates from elsewhere.
    for item in sorted(truth, key=lambda item: not item.get("own", True)):
        kind, entry = item["kind"], report[item["kind"]]
        own = item.get("own", True)
        entry["truth"] += own
        label = item.get("name") or " / ".join(str(part) for part in (item.get("pb") or ())[:1]) + " " + \
            " / ".join(str(item.get(f)) for f in CORE.get(kind, []))
        pool = pools.get((kind, identity(item)))
        if not pool:
            if own:
                entry["missed"].append(label)
            continue
        fields = FIELDS[kind]
        best = max(pool, key=lambda other: sum(same(item.get(f), other.get(f)) for f in fields))
        pool.remove(best)
        entry["matched"] += 1
        entry["found"] += own
        for field in fields:
            if item.get(field) in (None, "") and best.get(field) in (None, ""):
                continue
            if (kind, field) in BLANK_ALLOWED and best.get(field) in (None, ""):
                continue
            correct = same(item.get(field), best.get(field))
            entry["fields"] += 1
            entry["correct"] += correct
            if field in METRICS:
                entry["metric_fields"] += 1
                entry["metric_correct"] += correct
            if not correct:
                entry["errors"].append({"item": label, "field": field, "expected": item.get(field),
                                        "extracted": best.get(field)})
    for (kind, _), pool in pools.items():
        report[kind].setdefault("extra", []).extend(item.get("name") or str(identity(item)) for item in pool)
    return report


def ratio(a, b):
    return round(a / b, 3) if b else None


def summarize(report):
    results = [report[kind] for kind in RESULT_KINDS]
    total = lambda key, entries: sum(entry[key] for entry in entries)
    entries = list(report.values())
    return {
        "result_recall": ratio(total("found", results), total("truth", results)),
        "result_precision": ratio(total("matched", results), total("extracted", results)),
        "metric_accuracy": ratio(total("metric_correct", entries), total("metric_fields", entries)),
        "field_accuracy": ratio(total("correct", entries), total("fields", entries)),
        **{f"{kind}_recall": ratio(entry["found"], entry["truth"]) for kind, entry in report.items()},
    }


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def is_baseline(record):
    """A result for a model already in the database (named, not a record in this extraction)."""
    ref = record["data"].get("pretrained_backbone")
    return record["kind"] in RESULT_KINDS and isinstance(ref, str) and not ref.startswith("$")


def score_baselines(records):
    """Results for other models, compared with what is stored for those models."""
    names = {record["data"]["pretrained_backbone"] for record in records}
    families = list(BackboneFamily.objects.filter(pretrainedbackbone__name__in=names).distinct())
    detail = score(truth_items(families), extracted_items(records))
    results = [detail[kind] for kind in RESULT_KINDS]
    matched = sum(entry["matched"] for entry in results)
    return {"extracted": len(records), "already_stored": matched, "new": len(records) - matched,
            "value_errors": [error for entry in results for error in entry["errors"]],
            "records": [f"{record['data']['pretrained_backbone']} / {record['data'].get('dataset')}" for record in records]}


def score_paper(folder, families, identifier=None):
    all_records = read_json(folder / "records.json", [])
    records = [record for record in all_records if not is_baseline(record)]
    baselines = score_baselines([record for record in all_records if is_baseline(record)])
    pages = read_json(folder / "pages.json", [])
    sources = {item["url"]: item["text"] for item in
               (read_json(path) for path in sorted((folder / "agent" / "sources").glob("*.json")))}
    detail = score(truth_items(families, identifier), extracted_items(records))
    evidence_failures = []
    for record in all_records:
        try:
            validate_evidence(SimpleNamespace(kind=record["kind"], data=record["data"], evidence=record["evidence"],
                                              inferred=record.get("inferred", []), note=record.get("note", ""),
                                              reviewed_by=""), pages, sources)
        except (ValueError, KeyError, TypeError) as exc:
            evidence_failures.append(f"{record['kind']} {record['key']}: {exc}")
    calls = [read_json(folder / "agent.json", {})]
    result = {
        "families": [family.name for family in families],
        "paper_qualified": (read_json(folder / "decision.json") or {}).get("qualifies"),
        "errors": [calls[0].get("summary")] if calls[0].get("is_error") else [],
        "auth": calls[0].get("auth"),
        "records": len(all_records),
        "baselines": baselines,
        "uncertain": [f"{record['kind']} {record['key']}.{item['field']}: {item['reason']}"
                      for record in all_records for item in record.get("uncertain", [])],
        "evidence_failures": evidence_failures,
        "notes": [f"{record['kind']} {record['key']}: {record['note']}" for record in all_records if record.get("note")],
        "input_tokens": sum(call.get("input_tokens", 0) for call in calls),
        "output_tokens": sum(call.get("output_tokens", 0) for call in calls),
        "cost_usd": sum(call.get("cost_usd") or 0 for call in calls) if any(call.get("cost_usd") is not None for call in calls) else None,
        "seconds": sum(call.get("seconds", 0) for call in calls),
        "summary": summarize(detail), "detail": detail,
    }
    write_json(folder / "score.json", result)
    return result


def score_eval(directory):
    """(Re)score an eval directory against the database; reads only."""
    directory = Path(directory)
    config = read_json(directory / "config.json")
    families = {family.name: family for family in BackboneFamily.objects.all()}
    per_paper, combined = {}, {kind: defaultdict(int) for kind in FIELDS}
    for identifier, names in config["papers"].items():
        folder = directory / "papers" / identifier.replace("/", "_")
        if not (folder / "records.json").exists():
            continue
        result = score_paper(folder, [families[name] for name in names if name in families], identifier)
        per_paper[identifier] = result
        for kind, entry in result["detail"].items():
            for key, value in entry.items():
                if isinstance(value, int):
                    combined[kind][key] += value
    costs = [paper["cost_usd"] for paper in per_paper.values() if paper["cost_usd"] is not None]
    summary = {
        "model": config["model"], "papers": len(config["papers"]), "scored": len(per_paper),
        "overall": summarize(combined) if per_paper else {},
        "input_tokens": sum(paper["input_tokens"] for paper in per_paper.values()),
        "output_tokens": sum(paper["output_tokens"] for paper in per_paper.values()),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "per_paper": {identifier: {key: value for key, value in paper.items() if key != "detail"}
                      for identifier, paper in per_paper.items()},
    }
    write_json(directory / "summary.json", summary)
    (directory / "summary.md").write_text(markdown(summary))
    return summary


def baselines_cell(paper):
    baselines = paper.get("baselines") or {}
    if not baselines.get("extracted"):
        return "–"
    errors = len(baselines["value_errors"])
    return f"{baselines['already_stored']} / {baselines['new']}" + (f" ({errors} values differ)" if errors else "")


def markdown(summary):
    def pct(value):
        return "–" if value is None else f"{value * 100:.0f}%"

    def usd(value):
        return "–" if value is None else f"${value:.2f}"
    lines = [f"# Extraction eval: {summary['model']}", "",
             f"{summary['scored']} of {summary['papers']} papers scored. "
             f"Tokens: {summary['input_tokens']:,} in / {summary['output_tokens']:,} out. Cost: {usd(summary['cost_usd'])}.",
             "",
             "| Paper | Families | Qualifies | Result recall | Result precision | Metric accuracy | Field accuracy "
             "| Baselines (stored / new) | Uncertain | Citation failures | Tokens in / out | Cost | Time |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for identifier, paper in summary["per_paper"].items():
        s = paper["summary"]
        qualified = {True: "yes", False: "no", None: "–"}[paper["paper_qualified"]]
        lines.append(f"| {identifier} | {', '.join(paper['families'])} | {qualified} | {pct(s['result_recall'])} "
                     f"| {pct(s['result_precision'])} | {pct(s['metric_accuracy'])} | {pct(s['field_accuracy'])} "
                     f"| {baselines_cell(paper)} | {len(paper['uncertain'])} | {len(paper['evidence_failures'])} "
                     f"| {paper['input_tokens']:,} / {paper['output_tokens']:,} | {usd(paper['cost_usd'])} | {paper['seconds']:.0f}s |")
    overall = summary["overall"]
    if overall:
        lines.append(f"| **Overall** | | | {pct(overall['result_recall'])} | {pct(overall['result_precision'])} "
                     f"| {pct(overall['metric_accuracy'])} | {pct(overall['field_accuracy'])} | | | | | {usd(summary['cost_usd'])} | |")
    billed = [identifier for identifier, paper in summary["per_paper"].items() if paper.get("auth") not in (None, "none")]
    lines += ["", "Cost is Claude Code's API-equivalent estimate; sessions on the claude.ai login are not billed per token.",
              "Each paper's `score.json` lists missed and extra items and every wrong value. The reference is the "
              "database itself, so a mismatch can also be an error in the stored record."]
    if billed:
        lines.append(f"**Warning:** these sessions used an API credential and were billed to it: {', '.join(billed)}.")
    return "\n".join(lines) + "\n"

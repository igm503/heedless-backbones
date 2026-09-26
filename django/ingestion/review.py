"""The review page: each extracted value next to its evidence (a crop of the PDF around the
quotation, or the saved web page excerpt), with the agent's notes, inferred sources and
uncertainties, and the reviewer's actions."""
import io
import json
import re
from pathlib import Path

import pdfplumber
from django.conf import settings
from django.db import models, transaction
from django.utils import timezone

from .evidence import normalized
from .importer import REGISTRY, apply_run, approve_run, run_sources, snapshot
from .models import ExtractedRecord, ImportChange, IngestionRun

CROP_PADDING = 18


# Evidence display ---------------------------------------------------------------------------

def squash(text):
    return re.sub(r"[\s^]+", "", normalized(text))


def quote_box(page, quote):
    """The bounding box of a quotation's words on a pdfplumber page, or None."""
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2, use_text_flow=False)
    target = squash(quote)
    if not target:
        return None
    joined, owners = "", []
    for index, word in enumerate(words):
        piece = squash(word["text"])
        joined += piece
        owners += [index] * len(piece)
    start = joined.find(target)
    if start < 0:
        # Fall back to the longest matching prefix, so partial table rows still point somewhere.
        for length in range(len(target) - 1, 12, -4):
            start = joined.find(target[:length])
            if start >= 0:
                target = target[:length]
                break
        else:
            return None
    chosen = [words[i] for i in sorted(set(owners[start:start + len(target)]))]
    return (min(w["x0"] for w in chosen), min(w["top"] for w in chosen),
            max(w["x1"] for w in chosen), max(w["bottom"] for w in chosen))


def crop_png(run, citation):
    """A PNG of the page region around a quotation (cached in private storage)."""
    cache = Path(settings.MEDIA_ROOT) / "crops" / str(run.pk)
    key = re.sub(r"[^a-z0-9]+", "-", f"p{citation.get('page')}-{citation.get('quote', '')}".lower())[:120]
    path = cache / f"{key}.png"
    if path.exists():
        return path.read_bytes()
    with run.paper.pdf.open("rb") as source, pdfplumber.open(io.BytesIO(source.read())) as pdf:
        page = pdf.pages[citation["page"] - 1]
        box = quote_box(page, citation.get("quote", ""))
        if box:
            x0, top, x1, bottom = box
            # Show the whole line width (table rows and their neighbours) around the quotation.
            region = (max(0, min(x0, 30) - 6), max(0, top - CROP_PADDING * 2),
                      min(page.width, max(x1, page.width - 30) + 6), min(page.height, bottom + CROP_PADDING * 2))
            image = page.crop(region).to_image(resolution=150)
            image.draw_rect((x0 - 2, top - 2, x1 + 2, bottom + 2), stroke=(220, 50, 50), fill=(255, 230, 0, 60))
        else:
            image = page.to_image(resolution=60)
        buffer = io.BytesIO()
        (image.annotated if box else image.original).save(buffer, format="PNG")
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def excerpt(text, quote, context=300):
    """(before, match, after) around a quotation in a saved web page, in its original case."""
    words = quote.split()
    match = re.search(r"\s+".join(map(re.escape, words)), text, re.I) if words else None
    if not match:
        return text[:context], "", ""
    return text[max(0, match.start() - context):match.start()], match.group(0), text[match.end():match.end() + context]


# Page data -------------------------------------------------------------------------------------

def field_rows(record, sources):
    uncertain = {item["field"]: item["reason"] for item in record.uncertain}
    inferred = {item["field"]: item["source"] for item in record.inferred}
    overrides = {item["field"]: item for item in record.overrides}
    evidence = {}
    for index, item in enumerate(record.evidence):
        evidence.setdefault(item["field"], []).append({**item, "index": index})
    rows = []
    for name, value in record.data.items():
        items = []
        for item in evidence.get(name, []):
            citation = item.get("citation") or {}
            entry = {"index": item["index"], "location": citation.get("location", ""), "quote": citation.get("quote", ""),
                     "page": citation.get("page"), "url": citation.get("url"), "derivation": item.get("derivation")}
            if entry["url"]:
                entry["excerpt"] = excerpt(sources.get(entry["url"], ""), entry["quote"])
            items.append(entry)
        issue = next((issue for issue in record.issues if name in issue), "")
        flags = {"uncertain": uncertain.get(name), "inferred": inferred.get(name), "override": overrides.get(name),
                 "issue": issue, "derived": any(item["derivation"] for item in items),
                 "web": any(item["url"] for item in items)}
        rows.append({"name": name, "value": value, "evidence": items, **flags,
                     "flagged": bool(flags["uncertain"] or flags["inferred"] or flags["override"] or issue
                                     or flags["derived"] or flags["web"])})
    return rows


KIND_TITLES = {
    "family": "Family", "backbone": "Backbone", "pretrained_backbone": "Pretrained model",
    "classification": "Classification", "instance": "Detection / instance segmentation",
    "semantic": "Semantic segmentation", "fps": "Throughput",
    "category": "New category", "dataset": "New dataset", "head": "New head",
}
# Links that the hierarchy already shows.
STRUCTURAL = {"family", "backbone", "pretrained_backbone", "owner"}
RESULT_ORDER = ["classification", "instance", "semantic"]


def references(record):
    return {value[1:] for value in record.data.values() if isinstance(value, str) and value.startswith("$")}


def descendants(root, records):
    """Records that depend on root through $key links (results under a pretrained model, results
    using a proposed head, ...), transitively."""
    found, frontier = [], {root.key}
    while frontier:
        layer = [record for record in records if record not in found and record is not root
                 and references(record) & frontier]
        found += layer
        frontier = {record.key for record in layer}
    return found


def label(record, by_key):
    data = record.data

    def name(value):
        if isinstance(value, str) and value.startswith("$") and value[1:] in by_key:
            return by_key[value[1:]].data.get("name", value)
        return value
    kind = record.kind
    if kind in ("family", "backbone", "pretrained_backbone", "dataset", "head"):
        return data.get("name") or record.key
    if kind == "category":
        return f"{data.get('scope')}: {data.get('value')}"
    if kind == "classification":
        text = f"{name(data.get('dataset'))} @ {data.get('resolution')}"
        if data.get("fine_tune_dataset"):
            text += f" (fine-tuned on {name(data.get('fine_tune_dataset'))}, {data.get('fine_tune_epochs')} ep)"
        return text
    if kind == "instance":
        return (f"{name(data.get('head'))} · {name(data.get('dataset'))} · {data.get('instance_type')} · "
                f"{data.get('train_epochs')} ep")
    if kind == "semantic":
        return f"{name(data.get('head'))} · {name(data.get('dataset'))} · crop {data.get('crop_size')}"
    if kind == "fps":
        return f"{data.get('gpu')} {data.get('precision') or ''} @ {data.get('resolution')}: {data.get('fps')} img/s"
    return record.key


def model_name(record, by_key):
    """The pretrained model a result belongs to (a record in this run, or an existing name)."""
    ref = record.data.get("pretrained_backbone")
    if isinstance(ref, str) and ref.startswith("$"):
        parent = by_key.get(ref[1:])
        return parent.data.get("name", ref) if parent else ref
    return ref or ""


def uncertain_row(row):
    return bool(row["uncertain"])


def page_data(run, show="flagged"):
    """The run as a tree in the family_data YAML layout, filtered to what needs attention."""
    sources = run_sources(run)
    records = list(run.records.order_by("pk"))
    by_key = {record.key: record for record in records}
    changes = {}
    for change in ImportChange.objects.filter(record__run=run).values("record_id", "model", "object_id").distinct():
        changes.setdefault(change["record_id"], []).append(change)

    def children_of(key, kinds, field):
        return [record for record in records if record.kind in kinds and record.data.get(field) == f"${key}"]

    def node(record, children=(), full=False):
        """full: show every field whatever the filter (results inside a proposal)."""
        rows = [row for row in field_rows(record, sources) if row["name"] not in STRUCTURAL]
        for row in rows:
            # Show links to other records in this run by name ("ATSS (proposed)", not "$head_atss").
            value = row["value"]
            if isinstance(value, str) and value.startswith("$") and value[1:] in by_key:
                row["display"] = f"{label(by_key[value[1:]], by_key)} (proposed in this run)"
        removed = record.status == ExtractedRecord.Status.REJECTED
        if full:
            shown = rows
        elif show == "uncertain":
            shown = [row for row in rows if uncertain_row(row)]
        elif show == "flagged":
            shown = [row for row in rows if row["flagged"]]
        else:
            shown = rows
        children = [child for child in children if child is not None]
        attention = bool(shown) or (show != "uncertain" and bool(record.issues or record.note))
        visible = show == "all" or (not removed and (attention or any(child["visible"] for child in children)))
        return {"record": record, "kind": KIND_TITLES.get(record.kind, record.kind), "label": label(record, by_key),
                "removal_root": removed and "(with " not in record.review_note,
                "rows": shown, "all_rows": len(rows), "children": children, "visible": visible, "removed": removed,
                "changes": changes.get(record.pk, []), "attention": attention,
                "uncertain": sum(uncertain_row(row) for row in rows)}

    def result_node(record, full=False):
        return node(record, [node(fps, full=full) for fps in children_of(record.key, {"fps"}, "owner")], full=full)

    def pretrained_node(record):
        results = [result_node(result) for kind in RESULT_ORDER
                   for result in children_of(record.key, {kind}, "pretrained_backbone")]
        return node(record, results)

    def backbone_node(record):
        return node(record, [node(fps) for fps in children_of(record.key, {"fps"}, "owner")] +
                    [pretrained_node(item) for item in children_of(record.key, {"pretrained_backbone"}, "backbone")])

    families = [node(record, [backbone_node(item) for item in children_of(record.key, {"backbone"}, "family")])
                for record in records if record.kind == "family"]

    def existing(field, kinds):
        """Records attached to entities already in the database, grouped by that entity's name."""
        groups = {}
        for record in records:
            value = record.data.get(field)
            if record.kind in kinds and isinstance(value, str) and not value.startswith("$"):
                groups.setdefault(value, []).append(record)
        return groups

    additions = []
    for name, items in existing("family", {"backbone"}).items():
        additions.append({"title": f"Added to existing family {name}", "nodes": [backbone_node(item) for item in items]})
    for name, items in existing("backbone", {"pretrained_backbone"}).items():
        additions.append({"title": f"Added to existing backbone {name}", "nodes": [pretrained_node(item) for item in items]})
    for name, items in existing("pretrained_backbone", set(RESULT_ORDER)).items():
        additions.append({"title": f"Results for existing model {name}", "nodes": [result_node(item) for item in items]})

    proposals = []
    for record in records:
        if record.kind in ("category", "dataset", "head"):
            proposal = node(record)
            users = [item for item in records if record.key in references(item)] if record.kind != "category" else [
                item for item in records if record.data.get("value") in item.data.values()]
            proposal["used_by"] = [{"record": item, "label": f"{model_name(item, by_key)} — {label(item, by_key)}",
                                    "values": {k: v for k, v in item.data.items() if k in METRIC_FIELDS and v is not None}}
                                   for item in users]
            # The results using a proposed head or dataset, in full, so both are judged together.
            proposal["children"] = [result_node(item, full=True) for item in users if item.kind in RESULT_ORDER]
            for child in proposal["children"]:
                child["visible"] = show == "all" or not child["removed"]
            proposal["visible"] = show == "all" or not proposal["removed"]
            proposals.append(proposal)

    active = [record for record in records if record.status != ExtractedRecord.Status.REJECTED]
    return {"run": run, "show": show, "proposals": proposals, "families": families, "additions": additions,
            "total": len(records), "removed": len(records) - len(active),
            "uncertain": sum(len(record.uncertain) for record in active),
            "agent": next((call for call in run.calls if call.get("stage") == "agent"), None),
            "editable": run.status != IngestionRun.Status.IMPORTED}


METRIC_FIELDS = {"top_1", "top_5", "mAP", "AP50", "AP75", "ms_m_iou", "ss_m_iou", "gflops", "fps"}


# Actions -----------------------------------------------------------------------------------------

def parse_value(record, field, raw):
    model = REGISTRY.get(record.kind)
    raw = raw.strip()
    if raw.lower() in {"", "null", "none"}:
        return None
    try:
        model_field = model._meta.get_field(field) if model else None
    except Exception:
        model_field = None
    if isinstance(model_field, models.ForeignKey) or model_field is None and not raw[:1] in "[{\"0123456789-":
        return raw
    if isinstance(model_field, models.BooleanField):
        return raw.lower() in {"true", "yes", "1"}
    if isinstance(model_field, (models.IntegerField, models.FloatField)):
        number = float(raw)
        return int(number) if isinstance(model_field, models.IntegerField) else number
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def correct(record, field, raw, actor, note):
    """Set one value by hand; the reviewer becomes its source."""
    if not note.strip():
        raise ValueError("A correction needs a note")
    if record.status == ExtractedRecord.Status.REJECTED:
        raise ValueError("Restore this section before correcting it")
    if record.run.status == IngestionRun.Status.IMPORTED:
        raise ValueError("This run is imported; correct the stored record in the stats admin (it is audited there)")
    with transaction.atomic():
        before = snapshot(record)
        old = record.data.get(field)
        record.data[field] = parse_value(record, field, raw)
        record.overrides = [item for item in record.overrides if item["field"] != field] + [
            {"field": field, "old": old, "new": record.data[field], "actor": actor, "note": note,
             "at": timezone.now().isoformat()}]
        record.uncertain = [item for item in record.uncertain if item["field"] != field]
        record.reviewed_by, record.reviewed_at, record.review_note = actor, timezone.now(), note
        record.save()
        ImportChange.objects.create(record=record, run=record.run, model=record._meta.label_lower,
                                    object_id=record.pk, before=before, after=snapshot(record), actor=actor)


def accept(record, field, actor, note):
    """Confirm an uncertain value as extracted; the reviewer becomes its source."""
    correct(record, field, json.dumps(record.data.get(field)), actor, note or "Accepted as extracted")


def approve(run, actor, note, allow_updates=False):
    """Approve the run's proposals and publish; returns the problems that still block it."""
    approve_run(run, actor, note)
    problems = apply_run(run, publish=True, actor=actor, allow_updates=allow_updates)
    return problems


def remove(record, actor, note):
    """Remove a record and everything that depends on it from the run's import."""
    run = record.run
    if run.status == IngestionRun.Status.IMPORTED:
        raise ValueError("This run is imported; correct the stored records in the stats admin")
    records = list(run.records.all())
    with transaction.atomic():
        for item in [record, *descendants(record, records)]:
            if item.status == ExtractedRecord.Status.REJECTED:
                continue
            before = snapshot(item)
            item.status = ExtractedRecord.Status.REJECTED
            item.reviewed_by, item.reviewed_at = actor, timezone.now()
            item.review_note = f"Removed in review: {note or 'no note'}" + ("" if item is record else f" (with {record.key})")
            item.save()
            ImportChange.objects.create(record=item, run=run, model=item._meta.label_lower, object_id=item.pk,
                                        before=before, after=snapshot(item), actor=actor)
    apply_run(run)  # Re-validate what remains (dry run).


def restore(record, actor):
    """Undo a removal: the record and everything removed along with it."""
    run = record.run
    with transaction.atomic():
        for item in run.records.filter(status=ExtractedRecord.Status.REJECTED):
            if item.pk == record.pk or item.review_note.endswith(f"(with {record.key})"):
                before = snapshot(item)
                item.status, item.review_note = ExtractedRecord.Status.PENDING, f"Restored by {actor}"
                item.save()
                ImportChange.objects.create(record=item, run=run, model=item._meta.label_lower, object_id=item.pk,
                                            before=before, after=snapshot(item), actor=actor)
    apply_run(run)


def reject_and_retry(run, actor, note):
    """Reject the run and queue a new full read of the paper that is given the reviewer's feedback."""
    if not note.strip():
        raise ValueError("Say what the retry should do differently")
    reject(run, actor, note)
    return IngestionRun.objects.create(
        paper=run.paper, status=IngestionRun.Status.SHORTLISTED, provider="claude-code", retry_of=run,
        feedback=note, decision=run.decision, prompt_version=run.prompt_version, code_version=run.code_version,
        calls=[{"stage": "retry", "of": run.pk, "actor": actor, "note": note, "at": timezone.now().isoformat()}])


def retry_feedback(run):
    """The reviews of every earlier run in a retry chain, oldest first, for the agent's prompt."""
    chain, previous = [], run.retry_of
    while previous is not None:
        chain.insert(0, previous)
        previous = previous.retry_of
    sections = []
    for number, earlier in enumerate(chain, 1):
        records = list(earlier.records.order_by("pk"))
        by_key = {record.key: record for record in records}
        lines = [f"Review {number} (run {earlier.pk}): {earlier.error or 'rejected'}"]
        removed = [record for record in records if record.status == ExtractedRecord.Status.REJECTED
                   and record.review_note.startswith("Removed in review") and "(with " not in record.review_note]
        lines += [f"- Removed {KIND_TITLES.get(r.kind, r.kind).lower()} {label(r, by_key)}: "
                  f"{r.review_note.removeprefix('Removed in review: ')}" for r in removed]
        lines += [f"- Corrected {label(r, by_key)} {item['field']}: {item['old']!r} -> {item['new']!r} ({item['note']})"
                  for r in records for item in r.overrides]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def reject(run, actor, note):
    if not note.strip():
        raise ValueError("A rejection needs a note")
    with transaction.atomic():
        # Sections removed earlier keep their own notes (they explain the removal).
        run.records.exclude(status__in=[ExtractedRecord.Status.IMPORTED, ExtractedRecord.Status.REJECTED]).update(
            status=ExtractedRecord.Status.REJECTED, reviewed_by=actor, reviewed_at=timezone.now(), review_note=note)
        run.status, run.error = IngestionRun.Status.REJECTED, f"Rejected by {actor}: {note}"
        run.save(update_fields=["status", "error"])

"""family_data/<family>.yml: the human-readable reference copy of each model family.

The nested format is the one used by add_yaml and dump_yaml. Review drafts use the
same format plus a top-level ``review`` block that must be removed before import.
"""
from pathlib import Path

import yaml
from django.conf import settings

from stats.models import (
    Backbone, BackboneFamily, ClassificationResult, InstanceResult, PretrainedBackbone,
    SemanticSegmentationResult,
)
from .evidence import validate_evidence
from .importer import run_sources
from .models import ExtractedRecord

FAMILY_FIELDS = ["name", "model_type", "hierarchical", "pretrain_method", "pub_date", "paper", "github"]
BACKBONE_FIELDS = ["name", "m_parameters", "paper", "github"]
PRETRAINED_FIELDS = ["name", "pretrain_dataset", "pretrain_method", "pretrain_resolution", "pretrain_epochs", "paper", "github"]
FPS_FIELDS = ["resolution", "gpu", "precision", "fps", "batch_size", "source"]
CLASSIFICATION_FIELDS = [
    "dataset", "resolution", "top_1", "top_5", "gflops",
    "fine_tune_dataset", "fine_tune_epochs", "fine_tune_resolution",
    "intermediate_fine_tune_dataset", "intermediate_fine_tune_epochs", "intermediate_fine_tune_resolution",
]
INSTANCE_FIELDS = [
    "head", "dataset", "instance_type", "train_dataset", "train_epochs", "mAP", "AP50", "AP75",
    "mAPs", "mAPm", "mAPl", "gflops", "intermediate_train_dataset", "intermediate_train_epochs",
]
SEMANTIC_FIELDS = [
    "head", "dataset", "train_dataset", "train_epochs", "crop_size", "ms_m_iou", "ms_pixel_accuracy",
    "ms_mean_accuracy", "ss_m_iou", "ss_pixel_accuracy", "ss_mean_accuracy", "flip_test", "gflops",
    "intermediate_train_dataset", "intermediate_train_epochs",
]
# Omitted from dumps when empty, as in the hand-written files.
OPTIONAL = {"paper", "github", "source", "batch_size", "fine_tune_dataset", "fine_tune_epochs",
            "fine_tune_resolution", "intermediate_fine_tune_dataset", "intermediate_fine_tune_epochs",
            "intermediate_fine_tune_resolution", "intermediate_train_dataset", "intermediate_train_epochs"}
RESULT_LISTS = {"classification_results": ("classification", CLASSIFICATION_FIELDS),
                "instance_results": ("instance", INSTANCE_FIELDS),
                "semantic_seg_results": ("semantic", SEMANTIC_FIELDS)}


def yaml_dir():
    return Path(settings.FAMILY_DATA_DIR)


def plain(value):
    if hasattr(value, "name") and not isinstance(value, str):
        return value.name  # Dataset, head or task
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def pick(obj, fields, family_paper=None):
    data = {}
    for name in fields:
        value = plain(getattr(obj, name))
        if name in OPTIONAL and value in (None, "") or name == "paper" and value == family_paper:
            continue
        data[name] = value
    return data


def fps_list(owner):
    return [pick(fps, FPS_FIELDS) for fps in owner.fps_measurements.order_by("pk")]


def family_to_dict(family):
    data = pick(family, FAMILY_FIELDS)
    data["pub_date"] = str(family.pub_date)
    data.setdefault("paper", family.paper)
    data.setdefault("github", family.github)
    data["backbones"] = []
    for backbone in Backbone.objects.filter(family=family).order_by("pk"):
        backbone_data = pick(backbone, BACKBONE_FIELDS, family.paper)
        backbone_data["fps_measurements"] = fps_list(backbone)
        backbone_data["pretrained_backbones"] = []
        for pretrained in PretrainedBackbone.objects.filter(backbone=backbone).order_by("pk"):
            pretrained_data = pick(pretrained, PRETRAINED_FIELDS, family.paper)
            for key, model, fields in [
                ("classification_results", ClassificationResult, CLASSIFICATION_FIELDS),
                ("instance_results", InstanceResult, INSTANCE_FIELDS),
                ("semantic_seg_results", SemanticSegmentationResult, SEMANTIC_FIELDS),
            ]:
                pretrained_data[key] = []
                for result in model.objects.filter(pretrained_backbone=pretrained).order_by("pk"):
                    result_data = pick(result, fields + ["paper", "github"], family.paper)
                    if key != "classification_results":
                        result_data["fps_measurements"] = fps_list(result)
                    pretrained_data[key].append(result_data)
            backbone_data["pretrained_backbones"].append(pretrained_data)
        data["backbones"].append(backbone_data)
    return data


def dump(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def write_family_yaml(family):
    return dump(family_to_dict(family), yaml_dir() / f"{family.name}.yml")


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def yaml_to_entries(data):
    """Flatten a nested family file into (key, kind, data) entries for the importer."""
    if not isinstance(data, dict):
        raise ValueError("A family file must be a mapping")
    if "review" in data:
        raise ValueError("Resolve the review items, then delete the review block before importing")
    entries = []

    def add(kind, fields):
        key = f"{kind}-{len(entries)}"
        entries.append((key, kind, fields))
        return "$" + key

    def children(parent, name):
        items = parent.get(name) or []
        if not isinstance(items, list):
            raise ValueError(f"{name} must be a list")
        return items

    def without(item, *names):
        if not isinstance(item, dict):
            raise ValueError("Expected a mapping")
        return {key: value for key, value in item.items() if key not in names}

    for dataset in children(data, "datasets"):
        add("dataset", dict(dataset))
    for head in children(data, "heads"):
        add("head", dict(head))
    family = add("family", without(data, "backbones", "datasets", "heads"))
    for backbone in children(data, "backbones"):
        backbone_ref = add("backbone", {**without(backbone, "fps_measurements", "pretrained_backbones"), "family": family})
        for fps in children(backbone, "fps_measurements"):
            add("fps", {**fps, "backbone_name": backbone["name"], "owner": backbone_ref})
        for pretrained in children(backbone, "pretrained_backbones"):
            pretrained_ref = add("pretrained_backbone", {**without(pretrained, *RESULT_LISTS),
                                                         "family": family, "backbone": backbone_ref})
            for key, (kind, _) in RESULT_LISTS.items():
                for result in children(pretrained, key):
                    result_ref = add(kind, {**without(result, "fps_measurements"), "pretrained_backbone": pretrained_ref})
                    for fps in children(result, "fps_measurements"):
                        add("fps", {**fps, "backbone_name": backbone["name"], "owner": result_ref})
    return entries


def review_items(run):
    """Everything that kept a run out of the database, per record, with the cited evidence."""
    from stats.categories import choices
    items, seen = [], set()

    def note(record, **item):
        label = f"{record.kind} {record.data.get('name') or record.key}" if record else "paper"
        entry = {"record": label, **item}
        marker = repr(entry)
        if marker not in seen:
            seen.add(marker)
            items.append(entry)

    for record in run.records.order_by("pk"):
        evidence = {item["field"]: item.get("citation") for item in record.evidence}
        for uncertain in record.uncertain:
            note(record, field=uncertain["field"], uncertain=uncertain["reason"],
                 evidence=evidence.get(uncertain["field"]))
        try:
            validate_evidence(record, run.paper.pages, run_sources(run))
        except ValueError as exc:
            note(record, problem=str(exc))
        scope = record.data.get("scope")
        if record.kind == "category" and scope in {"model_type", "pretrain_method"} \
                and record.data.get("value") not in dict(choices(scope)):
            note(record, problem=f"New {scope} {record.data.get('value')!r} needs approve_category")
        for issue in record.issues:
            note(record, problem=issue)
    if run.error and not items:
        note(None, problem=run.error)
    return items


def run_to_drafts(run):
    """Nested drafts, one per family, for a run that needs manual review."""
    records = list(run.records.exclude(status=ExtractedRecord.Status.REJECTED).order_by("pk"))
    by_key = {record.key: record for record in records}
    families, backbones, pretrained = {}, {}, {}

    def ref_name(value):
        if isinstance(value, str) and value.startswith("$") and value[1:] in by_key:
            return by_key[value[1:]].data.get("name", value)
        return value

    def node(record, fields, drop=()):
        data = {name: ref_name(value) for name, value in record.data.items() if name not in drop}
        ordered = {name: data.pop(name) for name in fields if name in data}
        ordered.update(data)  # Unknown fields stay visible; add_yaml will reject them.
        return {name: value for name, value in ordered.items() if not (name in OPTIONAL and value in (None, ""))}

    def family_node(name):
        if name not in families:
            existing = BackboneFamily.objects.filter(name=name).first()
            families[name] = family_to_dict(existing) if existing else {"name": name, "backbones": []}
            for backbone in families[name]["backbones"]:
                backbones[backbone["name"]] = backbone
                for item in backbone["pretrained_backbones"]:
                    pretrained[item["name"]] = item
        return families[name]

    def backbone_node(name):
        if name not in backbones:
            existing = Backbone.objects.get(name=name)
            family_node(existing.family.name)
        return backbones[name]

    def pretrained_node(name):
        if name not in pretrained:
            existing = PretrainedBackbone.objects.get(name=name)
            family_node(existing.family.name)
        return pretrained[name]

    for record in records:
        if record.kind == "family":
            data = node(record, FAMILY_FIELDS)
            data.setdefault("pub_date", run.paper.metadata.get("created", "")[:10] or None)
            data.setdefault("paper", run.paper.url)
            data.setdefault("github", "")
            families[data["name"]] = {**data, "backbones": []}
    extras = {"datasets": [], "heads": []}
    for record in records:
        kind = record.kind
        if kind == "backbone":
            data = {**node(record, BACKBONE_FIELDS, ["family"]), "fps_measurements": [], "pretrained_backbones": []}
            backbones[data["name"]] = data
            family_node(ref_name(record.data["family"]))["backbones"].append(data)
        elif kind in ("dataset", "head"):
            extras[kind + "s"].append(node(record, []))
    for record in records:
        if record.kind == "pretrained_backbone":
            data = node(record, PRETRAINED_FIELDS, ["family", "backbone"])
            data.update({key: [] for key in RESULT_LISTS})
            pretrained[data["name"]] = data
            backbone_node(ref_name(record.data["backbone"]))["pretrained_backbones"].append(data)
    results = {}
    for record in records:
        for key, (kind, fields) in RESULT_LISTS.items():
            if record.kind == kind:
                data = node(record, fields, ["pretrained_backbone"])
                if kind != "classification":
                    data["fps_measurements"] = []
                results[record.key] = data
                pretrained_node(ref_name(record.data["pretrained_backbone"]))[key].append(data)
    for record in records:
        if record.kind == "fps":
            data = node(record, FPS_FIELDS, ["owner", "backbone_name"])
            owner = str(record.data.get("owner", ""))[1:]
            if owner in results:
                results[owner]["fps_measurements"].append(data)
            else:
                backbone_node(ref_name("$" + owner))["fps_measurements"].append(data)

    review = {
        "run": run.pk,
        "paper": run.paper.url,
        "instructions": "Fix the values below (see the cited PDF pages), delete this review block, "
                        f"then run: python manage.py add_yaml <this file> --run {run.pk}",
        "items": review_items(run),
    }
    drafts = {}
    for name, family in families.items():
        draft = {"review": review}
        for kind, items in extras.items():
            if items:
                draft[kind] = items
        draft.update(family)
        drafts[name] = draft
    return drafts


def write_drafts(run):
    paths = []
    for name, draft in run_to_drafts(run).items():
        paths.append(dump(draft, yaml_dir() / "review" / f"{name}-run{run.pk}.yml"))
    return paths

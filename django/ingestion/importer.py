"""Validate proposals and apply them atomically, with a permanent change ledger."""
import json
import logging
import math

from django.core.exceptions import ValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db import connection, models, transaction
from django.db.models import Q
from django.utils import timezone

from stats.models import (
    Backbone, BackboneFamily, ClassificationResult, Dataset, DownstreamHead,
    FPSMeasurement, InstanceResult, PretrainedBackbone, SemanticSegmentationResult, Task,
)
from .evidence import check_citation, validate_evidence
from .models import ExtractedRecord, ImportChange, IngestionRun

logger = logging.getLogger(__name__)

REGISTRY = {
    "dataset": Dataset, "head": DownstreamHead, "family": BackboneFamily,
    "backbone": Backbone, "pretrained_backbone": PretrainedBackbone,
    "classification": ClassificationResult, "instance": InstanceResult,
    "semantic": SemanticSegmentationResult, "fps": FPSMeasurement,
}
METRICS = {"top_1", "top_5", "mAP", "AP50", "AP75", "mAPs", "mAPm", "mAPl",
           "ms_m_iou", "ms_pixel_accuracy", "ms_mean_accuracy", "ss_m_iou", "ss_pixel_accuracy", "ss_mean_accuracy"}
MEASUREMENTS = METRICS | {"gflops", "paper", "github", "source_record"}
TASKS = {"Classification", "Object Detection", "Instance Segmentation", "Semantic Segmentation"}


class ImportProblem(ValueError):
    def __init__(self, key, message):
        super().__init__(f"{key}: {message}")
        self.key = key


def snapshot(obj):
    data = {field.attname: getattr(obj, field.attname) for field in obj._meta.fields}
    for field in obj._meta.many_to_many:
        data[field.name] = list(getattr(obj, field.name).order_by("pk").values_list("pk", flat=True))
    return json.loads(json.dumps(data, cls=DjangoJSONEncoder))


def vocabulary():
    from stats.categories import choices
    return {
        "model_types": list(dict(choices("model_type"))),
        "pretrain_methods": list(dict(choices("pretrain_method"))),
        **{kind: list(model.objects.values_list("name", flat=True))
           for kind, model in REGISTRY.items() if kind in {"dataset", "head", "family", "backbone", "pretrained_backbone"}},
        "tasks": sorted(TASKS),
    }


def resolve(field, value, resolved):
    if value is None:
        return None
    model = field.related_model
    if not isinstance(value, str):
        raise ValueError(f"{field.name} needs a named reference")
    if value.startswith("$"):
        obj = resolved.get(value[1:])
        if obj is None:
            raise KeyError(value[1:])
        if not isinstance(obj, model):
            raise ValueError(f"{field.name} references the wrong record type")
        return obj
    matches = list(model.objects.filter(name=value)[:2])
    if len(matches) != 1:
        raise ValueError(f"{field.name}: expected one existing {model.__name__} named {value!r}")
    return matches[0]


def model_values(kind, data, resolved, paper=None, strict=True, links=()):
    """Field values for a record. With a paper, links and dates default to that exact source;
    strict (automatic) imports also reject any other link."""
    model = REGISTRY[kind]
    allowed = {field.name: field for field in model._meta.fields if not field.primary_key and field.name != "source_record"}
    values = {}
    for name, value in data.items():
        if name == "tasks" and kind in {"dataset", "head"} or name == "owner" and kind == "fps":
            continue
        if name not in allowed:
            raise ValueError(f"Unexpected field {name}")
        field = allowed[name]
        if isinstance(field, models.ForeignKey):
            value = resolve(field, value, resolved)
        elif isinstance(field, (models.IntegerField, models.FloatField)) and value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or isinstance(field, models.IntegerField) and value != int(value):
                raise ValueError(f"Invalid value for {name}")
        elif isinstance(field, models.BooleanField) and value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        if isinstance(field, models.DateField) and value is not None:
            value = field.to_python(value)
        values[name] = value
    if paper is None:
        return values
    if "paper" in allowed:
        # A record may instead link to an official source fetched for this run (models that
        # only the repository reports).
        if strict and values.get("paper") and values["paper"] not in {paper.url, *links}:
            raise ValueError("A record's paper must be the version inspected in this run or a source fetched for it")
        if not values.get("paper") or strict and values["paper"] not in links:
            values["paper"] = paper.url
    if kind == "family" and not values.get("pub_date"):
        values["pub_date"] = model._meta.get_field("pub_date").to_python(paper.metadata["created"][:10])
    if kind == "fps" and (strict or not values.get("source")):
        values["source"] = paper.url
    return values


def existing_object(model, values, kind, owner=None, ignore=()):
    """The object this record describes; ``ignore`` excludes results created by the same import."""
    if kind in {"dataset", "head", "family", "backbone", "pretrained_backbone"}:
        identity = {"name": values.get("name")}
    elif kind == "fps":
        identity = {key: value for key, value in values.items() if key != "source"}
        matches = list(owner.fps_measurements.filter(**identity).exclude(pk__in=ignore)[:2])
        if len(matches) > 1:
            raise ValueError("Multiple existing throughput measurements match")
        return matches[0] if matches else None
    else:
        # Include all experiment settings, including explicit unknown values.
        identity = {
            field.name: values.get(field.name, field.get_default() if field.has_default() else None)
            for field in model._meta.fields if not field.primary_key and field.name not in MEASUREMENTS
        }
    matches = list(model.objects.filter(**identity).exclude(pk__in=ignore)[:2])
    if len(matches) > 1:
        raise ValueError("Multiple existing records match; resolve the ambiguity first")
    if not matches and kind in {"classification", "instance", "semantic"}:
        compatible = model.objects.exclude(pk__in=ignore)
        for name, value in identity.items():
            if value is not None:
                compatible = compatible.filter(Q(**{name: value}) | Q(**{name + "__isnull": True}))
        if compatible.exists():
            raise ValueError("Incomplete experiment metadata may duplicate an existing result; reconcile its settings first")
    return matches[0] if matches else None


def run_sources(run):
    """{url: text} of the web pages fetched for a run's extraction (cached on the run)."""
    if not hasattr(run, "_sources"):
        run._sources = dict(run.sources.values_list("url", "text")) if run.pk else {}
    return run._sources


def check_category(record):
    from stats.categories import choices
    scope, value = record.data.get("scope"), record.data.get("value")
    if set(record.data) != {"scope", "value"} or scope not in {"model_type", "pretrain_method"}:
        raise ValueError("Invalid category proposal")
    if value not in dict(choices(scope)):
        raise ValueError(f"New {scope} {value!r} needs approval: manage.py approve_category {scope} {value!r}")


def apply_record(record, resolved, actor, allow_updates):
    if record.uncertain:
        raise ValueError("Uncertain: " + "; ".join(f"{item['field']} ({item['reason']})" for item in record.uncertain))
    validate_evidence(record, record.run.paper.pages, run_sources(record.run))
    if record.kind == "category":
        check_category(record)
        return None
    if record.kind in {"dataset", "head"} and not record.reviewed_by:
        name = record.data.get("name")
        if not REGISTRY[record.kind].objects.filter(name=name).exists():
            raise ValueError(f"New {record.kind} {name!r} needs approval: "
                             f"manage.py review_ingestion {record.run_id} --approve --actor <you> --note <why>")
    return write_object(record.kind, record.data, resolved, allow_updates=allow_updates,
                        ledger={"record": record, "run": record.run, "actor": actor},
                        source=record, paper=record.run.paper, links=tuple(run_sources(record.run)))


def write_object(kind, data, resolved, *, allow_updates, ledger, source=None, paper=None, strict=True, created=None,
                 links=()):
    """Create or match one benchmark object and record the change in the ledger.

    With ``created`` (a dict of model -> pks), results created earlier in the same import are
    not matched, so a file may list experiments whose recorded settings coincide.
    """
    model = REGISTRY[kind]
    values = model_values(kind, data, resolved, paper, strict, links)
    owner = None
    if kind == "fps":
        ref = data.get("owner", "")
        if not isinstance(ref, str) or not ref.startswith("$"):
            raise ValueError("Throughput measurement needs an owner reference")
        owner = resolved.get(ref[1:])
        if owner is None:
            raise KeyError(ref[1:])
        if not isinstance(owner, (Backbone, InstanceResult, SemanticSegmentationResult)):
            raise ValueError("Invalid throughput owner")
    ignore = created.setdefault(model, set()) if created is not None else ()
    obj = existing_object(model, values, kind, owner, ignore)
    before = snapshot(obj) if obj else None
    obj = obj or model()
    differences = []
    for name, value in values.items():
        old = getattr(obj, name) if before else None
        if before:
            # Preserve an existing primary paper/code link and known metadata.
            if name in {"paper", "github", "source"} and old or value is None:
                continue
            if old != value:
                differences.append(f"{name} ({old!r} stored, {value!r} in this paper)")
        setattr(obj, name, value)
    if differences and not allow_updates:
        raise ValueError(f"Existing record differs: {'; '.join(differences)}. Proposed correction; "
                         "apply with review_ingestion --approve --allow-updates")
    if hasattr(obj, "source_record_id") and (not before or differences):
        obj.source_record = source
    # An unknown optional link or text is null in an extraction (the field spec calls blank
    # fields nullable) but stored as "" in columns that do not allow null.
    for field in model._meta.fields:
        if isinstance(field, (models.CharField, models.TextField)) and not field.null and getattr(obj, field.attname) is None:
            setattr(obj, field.attname, "")
    obj.full_clean()
    if isinstance(obj, PretrainedBackbone) and obj.backbone.family_id != obj.family_id:
        raise ValueError("Pretrained backbone and architecture families disagree")
    task_names = data.get("tasks")
    if isinstance(obj, (Dataset, DownstreamHead)):
        if not isinstance(task_names, list) or not task_names or not set(task_names) <= TASKS:
            raise ValueError("Dataset/head needs supported task names")
        tasks = list(Task.objects.filter(name__in=task_names))
        if len(tasks) != len(set(task_names)):
            raise ValueError("A referenced task is not configured")
        if before and set(before["tasks"]) != {task.pk for task in tasks} and not allow_updates:
            raise ValueError("Existing dataset/head tasks differ; update approval required")
    if isinstance(obj, InstanceResult) and obj.instance_type.name not in {"Object Detection", "Instance Segmentation"}:
        raise ValueError("Only detection and instance segmentation are supported here")
    if hasattr(obj, "head"):
        task_name = obj.instance_type.name if isinstance(obj, InstanceResult) else "Semantic Segmentation"
        if not obj.head.tasks.filter(name=task_name).exists():
            raise ValueError("Head does not support the result's task")
    if isinstance(obj, InstanceResult) and not obj.dataset.tasks.filter(name=obj.instance_type.name).exists():
        raise ValueError("Dataset does not support the instance task")
    if hasattr(obj, "dataset") and not obj.dataset.eval:
        raise ValueError("Result requires an evaluation dataset")
    for metric in METRICS:
        value = getattr(obj, metric, None)
        is_error_rate = isinstance(obj, ClassificationResult) and obj.dataset.name in {"ImageNet-C", "ImageNet-C-bar"}
        if value is not None and not is_error_rate and not 0 <= value <= 100:
            raise ValueError(f"{metric} must be between 0 and 100")
    obj.save()
    if created is not None and not before:
        created[model].add(obj.pk)
    if task_names is not None:
        obj.tasks.set(tasks)
    if owner:
        owner_before = snapshot(owner)
        owner.fps_measurements.add(obj)
        owner_after = snapshot(owner)
        if owner_before != owner_after:
            ImportChange.objects.create(model=owner._meta.label_lower, object_id=owner.pk,
                                        before=owner_before, after=owner_after, **ledger)
    ImportChange.objects.create(model=obj._meta.label_lower, object_id=obj.pk,
                                before=before, after=snapshot(obj), **ledger)
    return obj


def lock_imports():
    # Also serialize admin and manual imports, not just the scheduled process.
    if connection.vendor == "postgresql":
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(73410291)")


def apply_all(items, apply):
    """Apply (key, item) pairs in dependency order; references use $key."""
    pending, resolved = list(items), {}
    while pending:
        remaining = []
        for key, item in pending:
            try:
                resolved[key] = apply(item, resolved)
            except KeyError:
                remaining.append((key, item))
            except (ValueError, ValidationError) as exc:
                raise ImportProblem(key, str(exc)) from exc
        if len(remaining) == len(pending):
            raise ImportProblem(remaining[0][0], "Missing or cyclic record reference")
        pending = remaining
    return resolved


def families_of(objects):
    families = set()
    for obj in objects:
        if isinstance(obj, BackboneFamily):
            families.add(obj.name)
        elif isinstance(obj, (Backbone, PretrainedBackbone)):
            families.add(obj.family.name)
        elif hasattr(obj, "pretrained_backbone"):
            families.add(obj.pretrained_backbone.family.name)
    return sorted(families)


def write_family_files(names):
    """Refresh family_data/<family>.yml; the database remains correct if this fails."""
    from .family_yaml import write_family_yaml
    written, errors = [], []
    for name in names:
        try:
            written.append(write_family_yaml(BackboneFamily.objects.get(name=name)))
        except OSError as exc:
            logger.exception("Could not write YAML for %s", name)
            errors.append(f"{name}: {exc}")
    return written, errors


def apply_entries(entries, *, actor, origin, run=None, allow_updates=False):
    """Import human-supplied (key, kind, data) entries, e.g. from a corrected YAML file."""
    ledger = {"record": None, "run": run, "actor": actor, "origin": origin}
    created = {}
    with transaction.atomic():
        lock_imports()
        if run is not None:
            run = IngestionRun.objects.select_for_update().get(pk=run.pk)
            if run.status == IngestionRun.Status.IMPORTED:
                raise ValueError(f"Run {run.pk} has already been imported")
        resolved = apply_all(
            ((key, (kind, data)) for key, kind, data in entries),
            lambda item, resolved: write_object(item[0], item[1], resolved, allow_updates=allow_updates, ledger=ledger,
                                                paper=run.paper if run else None, strict=False, created=created),
        )
        if run is not None:
            run.status, run.error = IngestionRun.Status.IMPORTED, ""
            run.save(update_fields=["status", "error"])
    return families_of(resolved.values())


def apply_run(run, *, publish=False, actor="automatic", allow_updates=False):
    """Dry runs keep proposals but roll back all benchmark writes and ledger entries."""
    if run.status == IngestionRun.Status.IMPORTED:
        return []
    # Records a reviewer removed (with everything under them) are left out of the import.
    records = list(run.records.exclude(status=ExtractedRecord.Status.REJECTED)
                   .select_related("run__paper").order_by("pk"))
    for record in records:
        record.issues = []
    problem = None
    try:
        with transaction.atomic():
            lock_imports()
            locked = IngestionRun.objects.select_for_update().get(pk=run.pk)
            if locked.status == IngestionRun.Status.IMPORTED:
                return []
            decision = run.decision
            if not all(decision.get(key) is True for key in ("qualifies", "architecture_or_pretraining_contribution", "imagenet_1k_results")):
                raise ImportProblem("decision", "Paper does not meet the selection criteria")
            if not decision.get("evidence"):
                raise ImportProblem("decision", "Missing full-paper selection evidence")
            for citation in decision["evidence"]:
                check_citation(citation, run.paper.pages, run_sources(run))
            if not any(record.kind == "classification" and record.data.get("dataset") == "ImageNet-1k" for record in records):
                raise ImportProblem("decision", "Need an extracted ImageNet-1k classification result")
            resolved = apply_all(((record.key, record) for record in records),
                                 lambda record, resolved: apply_record(record, resolved, actor, allow_updates))
            if publish:
                run.records.exclude(kind="category").exclude(status=ExtractedRecord.Status.REJECTED).update(
                    status=ExtractedRecord.Status.IMPORTED)
                run.status = IngestionRun.Status.IMPORTED
                run.save(update_fields=["status"])
            else:
                transaction.set_rollback(True)
    except (ImportProblem, ValueError, ValidationError) as exc:
        problem = exc
    if problem:
        key = getattr(problem, "key", "decision")
        for record in records:
            if record.key == key or key == "decision":
                record.issues = [str(problem)]
        run.status = IngestionRun.Status.REVIEW
        run.error = str(problem)
    else:
        if not publish:
            run.status = IngestionRun.Status.READY
        run.error = ""
    for record in records:
        record.save(update_fields=["issues"])
    run.save(update_fields=["status", "error"])
    return [str(problem)] if problem else []


def approve_run(run, actor, note):
    if not note.strip():
        raise ValueError("A review note is required")
    # Approval accepts documented derivation assumptions and value updates. It does not
    # approve categories (see approve_category) or resolve fields flagged as uncertain.
    with transaction.atomic():
        for record in run.records.exclude(status__in=[ExtractedRecord.Status.IMPORTED, ExtractedRecord.Status.REJECTED]):
            before = snapshot(record)
            record.status = ExtractedRecord.Status.APPROVED
            record.reviewed_by, record.reviewed_at, record.review_note = actor, timezone.now(), note
            record.save()
            ImportChange.objects.create(record=record, model=record._meta.label_lower, object_id=record.pk,
                                        before=before, after=snapshot(record), actor=actor)

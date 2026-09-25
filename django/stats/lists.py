from collections import defaultdict
from urllib.parse import urlencode

from django.db.models import Count, Q, Max, Prefetch
from django.urls import reverse

from .models import (
    BackboneFamily,
    Dataset,
    DownstreamHead,
    Task,
    TaskType,
    ClassificationResult,
    InstanceResult,
    SemanticSegmentationResult,
)


def get_dataset_lists():
    return get_task_lists(Dataset, "dataset", ("website",))


def get_head_lists():
    return get_task_lists(DownstreamHead, "head", ("github", "paper"))


def get_task_lists(model, view_name, link_fields):
    lists = defaultdict(lambda: defaultdict(list))
    for obj in get_result_restricting_data(model):
        for task in obj._tasks:
            row = get_count_date_row(obj, task)
            if not row["# results"]:
                continue
            links = {
                "name": reverse(view_name, args=[obj.name])
                + f"?{urlencode({'task': task.name})}",
            }
            for field in link_fields:
                row[field] = "link"
                links[field] = getattr(obj, field)
            lists[task.name]["rows"].append(row)
            lists[task.name]["links"].append(links)
    return {
        task: dict(table, headers=list(table["rows"][0]))
        for task, table in lists.items()
    }


def get_family_list():
    families = BackboneFamily.objects.all()
    rows = []
    row_links = []
    for family in families:
        row = {
            "name": family.name,
            "model type": family.model_type,
            "pretraining method": family.pretrain_method,
            "hierarchical": family.hierarchical,
            "publication date": family.pub_date,
            "github": "link",
            "paper": "link",
        }
        links = {
            "name": reverse("family", args=[family.name]),
            "github": family.github,
            "paper": family.paper,
        }
        rows.append(row)
        row_links.append(links)

    headers = list(rows[0].keys()) if rows else []
    return {"rows": rows, "links": row_links, "headers": headers}


def get_result_restricting_data(model):
    if model == Dataset:
        queryset = model.objects.filter(eval=True)
    else:
        queryset = model.objects.all()
    queryset = queryset.prefetch_related(Prefetch("tasks", Task.objects.all(), to_attr="_tasks"))
    queryset = queryset.annotate(
        det_count=get_count(InstanceResult, TaskType.DETECTION),
        inst_count=get_count(InstanceResult, TaskType.INSTANCE_SEG),
        sem_count=get_count(SemanticSegmentationResult),
        det_date=get_date(InstanceResult, TaskType.DETECTION),
        inst_date=get_date(InstanceResult, TaskType.INSTANCE_SEG),
        sem_date=get_date(SemanticSegmentationResult),
    )
    if model == Dataset:
        queryset = queryset.annotate(
            class_count=get_count(ClassificationResult),
            class_date=get_date(ClassificationResult),
        )

    return queryset


def get_count(result_model, result_type=None):
    if result_model == InstanceResult:
        return Count(
            "instanceresult",
            filter=Q(instanceresult__instance_type__name=result_type.value),
        )
    else:
        return Count(result_model._meta.model_name)


def get_date(result_model, result_type=None):
    if result_model == InstanceResult:
        return Max(
            "instanceresult__pretrained_backbone__family__pub_date",
            filter=Q(instanceresult__instance_type__name=result_type.value),
        )
    else:
        return Max(result_model._meta.model_name + "__pretrained_backbone__family__pub_date")


def get_count_date_row(obj, task):
    if task.name == TaskType.CLASSIFICATION.value:
        num_results = obj.class_count
        last_result = obj.class_date
    elif task.name == TaskType.DETECTION.value:
        num_results = obj.det_count
        last_result = obj.det_date
    elif task.name == TaskType.INSTANCE_SEG.value:
        num_results = obj.inst_count
        last_result = obj.inst_date
    elif task.name == TaskType.SEMANTIC_SEG.value:
        num_results = obj.sem_count
        last_result = obj.sem_date
    else:
        num_results = 0
        last_result = 0
    return {"name": obj.name, "# results": num_results, "last result": last_result}

from collections import defaultdict

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
)


def get_dataset_lists():
    datasets = get_result_restricting_data(Dataset)
    lists = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for task in dataset._tasks:
            row = get_count_date_row(dataset, task)
            if not row["# results"]:
                continue
            row["website"] = "link"
            links = {
                "name": reverse("dataset", args=[dataset.name]),
                "website": dataset.website,
            }
            lists[task.name]["rows"].append(row)
            lists[task.name]["links"].append(links)
    for task, lis in lists.items():
        lists[task]["headers"] = list(lis["rows"][0].keys()) if lis["rows"] else []
    return {k: dict(v) for k, v in lists.items()}


def get_head_lists():
    heads = get_result_restricting_data(DownstreamHead)
    lists = defaultdict(lambda: defaultdict(list))
    for head in heads:
        for task in head._tasks:
            row = get_count_date_row(head, task)
            if not row["# results"]:
                continue
            row["github"] = "link"
            row["paper"] = "link"
            links = {
                "name": reverse("head", args=[head.name]),
                "github": head.github,
                "paper": head.paper,
            }
            lists[task.name]["rows"].append(row)
            lists[task.name]["links"].append(links)
    for task, lis in lists.items():
        lists[task]["headers"] = list(lis["rows"][0].keys()) if lis["rows"] else []
    return {k: dict(v) for k, v in lists.items()}


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
        det_date=get_date(InstanceResult, TaskType.DETECTION),
        inst_date=get_date(InstanceResult, TaskType.INSTANCE_SEG),
    )
    if model == Dataset:
        queryset = queryset.annotate(
            class_count=get_count(ClassificationResult),
            class_date=get_date(ClassificationResult),
        )

    return queryset


def get_count(result_model, result_type=None):
    if result_model == ClassificationResult:
        return Count("classificationresult")
    else:
        assert result_type is not None
        return Count(
            "instanceresult",
            filter=Q(instanceresult__instance_type__name=result_type.value),
        )


def get_date(result_model, result_type=None):
    if result_model == ClassificationResult:
        return Max("classificationresult__pretrained_backbone__family__pub_date")
    else:
        assert result_type is not None
        return Max(
            "instanceresult__pretrained_backbone__family__pub_date",
            filter=Q(instanceresult__instance_type__name=result_type.value),
        )


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
    else:
        num_results = 0
        last_result = 0
    return {"name": obj.name, "# results": num_results, "last result": last_result}

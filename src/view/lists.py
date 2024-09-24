from collections import defaultdict

from django.db.models import Count, Subquery, OuterRef, Max, Prefetch
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


def subquery_filter(model, outer, instance_type):
    if outer == Dataset:
        queryset = model.objects.filter(dataset=OuterRef("pk"))
    elif outer == DownstreamHead:
        queryset = model.objects.filter(head=OuterRef("pk"))
    else:
        raise ValueError("outer must be either Dataset or DownstreamHead")
    if instance_type:
        queryset = queryset.filter(instance_type__name=instance_type.value)
    return queryset


def get_count_subquery(model, outer, outer_string, instance_type=None):
    queryset = subquery_filter(model, outer, instance_type)
    return Subquery(queryset.values(outer_string).annotate(count=Count("id")).values("count"))


def get_date_subquery(model, outer, outer_string, instance_type=None):
    queryset = subquery_filter(model, outer, instance_type)
    return Subquery(
        queryset.values(outer_string)
        .values("dataset")
        .annotate(latest_date=Max("pretrained_backbone__family__pub_date"))
        .values("latest_date")
    )


def get_count_date_row(obj, task):
    if task.name == TaskType.CLASSIFICATION.value:
        num_results = obj.class_count
        last_result = obj.class_date
    elif task.name == TaskType.DETECTION.value:
        num_results = obj.det_count
        last_result = obj.det_date
    elif task.name == TaskType.INSTANCE_SEG.value:
        num_results = obj.instance_count
        last_result = obj.instance_date
    else:
        num_results = 0
        last_result = 0
    return {"name": obj.name, "# results": num_results, "last result": last_result}


def get_dataset_lists():
    datasets = (
        Dataset.objects.filter(eval=True)
        .prefetch_related(Prefetch("tasks", Task.objects.all(), "_tasks"))
        .annotate(
            class_count=get_count_subquery(ClassificationResult, Dataset, "dataset"),
            det_count=get_count_subquery(InstanceResult, Dataset, "dataset", TaskType.DETECTION),
            instance_count=get_count_subquery(
                InstanceResult, Dataset, "dataset", TaskType.INSTANCE_SEG
            ),
            class_date=get_date_subquery(ClassificationResult, Dataset, "dataset"),
            det_date=get_date_subquery(InstanceResult, Dataset, "dataset", TaskType.DETECTION),
            instance_date=get_date_subquery(
                InstanceResult, Dataset, "dataset", TaskType.INSTANCE_SEG
            ),
        )
        .all()
    )

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
    heads = (
        DownstreamHead.objects.filter()
        .prefetch_related(Prefetch("tasks", Task.objects.all(), "_tasks"))
        .annotate(
            det_count=get_count_subquery(
                InstanceResult, DownstreamHead, "head", TaskType.DETECTION
            ),
            instance_count=get_count_subquery(
                InstanceResult, DownstreamHead, "head", TaskType.INSTANCE_SEG
            ),
            det_date=get_date_subquery(InstanceResult, DownstreamHead, "head", TaskType.DETECTION),
            instance_date=get_date_subquery(
                InstanceResult, DownstreamHead, "head", TaskType.INSTANCE_SEG
            ),
        )
        .all()
    )

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

from collections import defaultdict

from django.db.models import Count, Q, Subquery, OuterRef, Max, Prefetch
from django.db.models.functions import Coalesce
from django.urls import reverse

from .models import (
    BackboneFamily,
    Backbone,
    Dataset,
    DownstreamHead,
    Task,
    TaskType,
    PretrainMethod,
    ModelType,
    GPU,
    Precision,
    ClassificationResult,
    InstanceResult,
)


def get_dataset_lists():
    datasets = (
        Dataset.objects.filter(eval=True)
        .prefetch_related(Prefetch("tasks", Task.objects.all(), "_tasks"))
        .annotate(
            classification_result_count=Subquery(
                ClassificationResult.objects.filter(dataset=OuterRef("pk"))
                .values("dataset")
                .annotate(count=Count("id"))
                .values("count")
            ),
            detection_result_count=Subquery(
                InstanceResult.objects.filter(
                    dataset=OuterRef("pk"), instance_type__name=TaskType.DETECTION.value
                )
                .values("dataset")
                .annotate(count=Count("id"))
                .values("count")
            ),
            instance_seg_result_count=Subquery(
                InstanceResult.objects.filter(
                    dataset=OuterRef("pk"),
                    instance_type__name=TaskType.INSTANCE_SEG.value,
                )
                .values("dataset")
                .annotate(count=Count("id"))
                .values("count")
            ),
            latest_classification_family_date=Subquery(
                ClassificationResult.objects.filter(dataset=OuterRef("pk"))
                .values("dataset")
                .annotate(latest_date=Max("pretrainedbackbone__family__pub_date"))
                .values("latest_date")
            ),
            latest_detection_family_date=Subquery(
                InstanceResult.objects.filter(
                    dataset=OuterRef("pk"), instance_type__name=TaskType.DETECTION.value
                )
                .values("dataset")
                .annotate(latest_date=Max("pretrainedbackbone__family__pub_date"))
                .values("latest_date")
            ),
            latest_instance_seg_family_date=Subquery(
                InstanceResult.objects.filter(
                    dataset=OuterRef("pk"),
                    instance_type__name=TaskType.INSTANCE_SEG.value,
                )
                .values("dataset")
                .annotate(latest_date=Max("pretrainedbackbone__family__pub_date"))
                .values("latest_date")
            ),
        )
        .all()
    )

    lists = defaultdict(lambda: defaultdict(list))
    for dataset in datasets:
        for task in dataset._tasks:
            if task.name == TaskType.CLASSIFICATION.value:
                num_results = dataset.classification_result_count
                last_result = dataset.latest_classification_family_date
            elif task.name == TaskType.DETECTION.value:
                num_results = dataset.detection_result_count
                last_result = dataset.latest_detection_family_date
            elif task.name == TaskType.INSTANCE_SEG.value:
                num_results = dataset.instance_seg_result_count
                last_result = dataset.latest_instance_seg_family_date
            else:
                num_results = 0
                last_result = 0
            if not num_results:
                continue
            row = {
                "name": dataset.name,
                "# results": num_results,
                "last result": last_result,
                "website": "link",
            }
            links = {
                "name": reverse("dataset", args=[dataset.name]),
                "website": dataset.website,
            }
            lists[task.name]["rows"].append(row)
            lists[task.name]["links"].append(links)
    for task, lis in lists.items():
        if lis["rows"]:
            lists[task]["headers"] = list(lis["rows"][0].keys())
        else:
            lists[task]["headers"] = []
    lists = {k: dict(v) for k, v in lists.items()}
    return lists


def get_family_list():
    pass


def get_head_lists():
    pass

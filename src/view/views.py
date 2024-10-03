import time

from django.shortcuts import get_object_or_404, render

# from .logging import logger
from .form import PlotForm
from .lists import get_family_list, get_head_lists, get_dataset_lists
from .models import BackboneFamily, Backbone, Dataset, DownstreamHead, TaskType
from .plot import PlotRequest, get_plot_and_table
from .tables import (
    get_family_classification_table,
    get_family_instance_table,
    get_head_instance_table,
    get_dataset_classification_table,
    get_dataset_instance_table,
)

plot, table = None, None
family_plot, family_table = None, None
head_plot, head_table = None, None
dataset_plot, dataset_table = None, None

dataset_lists = None
head_lists = None
family_list = None


def all(request):
    global plot, table, headers
    form = PlotForm(request.GET or get_default_request())
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        plot, table = get_plot_and_table(plot_request)
    return render(
        request,
        "view/all.html",
        {
            "form": form,
            "plot": plot,
            "table": table,
        },
    )


def family(request, family_name):
    global family_plot, family_table
    try:
        family = BackboneFamily.objects.get(name=family_name)
    except BackboneFamily.DoesNotExist:
        backbone = get_object_or_404(Backbone, name=family_name)
        family = backbone.family
    form = PlotForm(request.GET or get_default_request(family=family))
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        family_plot, family_table = get_plot_and_table(
            plot_request, page="family", family_name=family_name
        )
    class_table = get_family_classification_table(family.name)
    det_tables = get_family_instance_table(family.name, TaskType.DETECTION)
    instance_tables = get_family_instance_table(family.name, TaskType.INSTANCE_SEG)

    return render(
        request,
        "view/family.html",
        {
            "family": family,
            "form": form,
            "plot": family_plot,
            "table": family_table,
            "classification_table": class_table,
            "detection_tables": det_tables,
            "instance_tables": instance_tables,
        },
    )


def head(request, head_name):
    global head_plot, head_table
    head = get_object_or_404(DownstreamHead, name=head_name)
    form = PlotForm(request.GET or get_default_request(head=head), head=head)
    if form.is_valid() and form.is_ready():
        form.cleaned_data["x_head"] = head
        form.cleaned_data["y_head"] = head
        plot_request = PlotRequest(form.cleaned_data)
        head_plot, head_table = get_plot_and_table(plot_request, page="head")
    head_tasks = [task.name for task in head.tasks.all()]
    det_tables = None
    instance_tables = None
    if TaskType.DETECTION.value in head_tasks:
        det_tables = get_head_instance_table(head.name, TaskType.DETECTION)
    if TaskType.INSTANCE_SEG.value in head_tasks:
        instance_tables = get_head_instance_table(head.name, TaskType.INSTANCE_SEG)

    return render(
        request,
        "view/head.html",
        {
            "head": head,
            "form": form,
            "plot": head_plot,
            "table": head_table,
            "detection_tables": det_tables,
            "instance_tables": instance_tables,
        },
    )


def dataset(request, dataset_name):
    global dataset_plot, dataset_table
    dataset = get_object_or_404(Dataset, name=dataset_name)
    form = PlotForm(request.GET or get_default_request(dataset=dataset), dataset=dataset)
    if form.is_valid() and form.is_ready():
        form.cleaned_data["x_dataset"] = dataset
        form.cleaned_data["y_dataset"] = dataset
        plot_request = PlotRequest(form.cleaned_data)
        dataset_plot, dataset_table = get_plot_and_table(plot_request, page="dataset")
    dataset_tasks = [task.name for task in dataset.tasks.all()]
    classification_table = None
    det_table = None
    instance_table = None
    if TaskType.CLASSIFICATION.value in dataset_tasks:
        classification_table = get_dataset_classification_table(dataset.name)
    if TaskType.DETECTION.value in dataset_tasks:
        det_table = get_dataset_instance_table(dataset.name, TaskType.DETECTION)
    if TaskType.INSTANCE_SEG.value in dataset_tasks:
        instance_table = get_dataset_instance_table(dataset.name, TaskType.INSTANCE_SEG)

    return render(
        request,
        "view/dataset.html",
        {
            "dataset": dataset,
            "form": form,
            "plot": dataset_plot,
            "table": dataset_table,
            "classification_table": classification_table,
            "detection_table": det_table,
            "instance_table": instance_table,
        },
    )


def datasets(request):
    global dataset_lists
    if dataset_lists is None:
        dataset_lists = get_dataset_lists()
    return render(request, "view/datasets.html", {"datasets": dataset_lists})


def families(request):
    global family_list
    if family_list is None:
        family_list = get_family_list()
    return render(request, "view/families.html", {"families": family_list})


def heads(request):
    global head_lists
    if head_lists is None:
        head_lists = get_head_lists()
    return render(request, "view/heads.html", {"heads": head_lists})


def get_default_request(family=None, head=None, dataset=None):
    if family:
        return {
            "y_axis": "results",
            "y_dataset": 1,
            "y_task": 4,
            "y_metric": "top_1",
            "x_axis": "gflops",
            "legend_attribute": "pretrain_dataset.name",
        }
    elif head:
        tasks = head.tasks.all()
        first_task = tasks.first()
        dataset = Dataset.objects.filter(tasks=first_task, eval=True).first()
        return {
            "y_axis": "results",
            "y_task": first_task.pk,
            "y_dataset": dataset.pk,
            "y_head": head.pk,
            "y_metric": "mAP",
            "x_axis": "gflops",
            "legend_attribute": "family.name",
        }
    elif dataset:
        task = dataset.tasks.first()
        if task.name == TaskType.CLASSIFICATION.value:
            metric = "top_1"
        else:
            metric = "mAP"
        return {
            "y_axis": "results",
            "y_task": task.pk,
            "y_dataset": dataset.pk,
            "y_metric": metric,
            "x_axis": "gflops",
            "legend_attribute": "family.name",
        }
    else:
        return {
            "y_axis": "results",
            "y_dataset": 1,
            "y_task": 4,
            "y_metric": "top_1",
            "x_axis": "gflops",
            "legend_attribute": "family.name",
        }

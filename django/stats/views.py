from django.shortcuts import get_object_or_404, render

from .form import PlotForm, get_default_request
from .lists import get_family_list, get_head_lists, get_dataset_lists
from .models import (
    BackboneFamily,
    Backbone,
    Dataset,
    DownstreamHead,
    TaskType,
)
from .plot import PlotRequest, get_plot_and_table
from .tables import (
    get_family_classification_table,
    get_family_downstream_table,
    get_head_downstream_table,
    get_dataset_classification_table,
    get_dataset_downstream_table,
)

plot, table = None, None
family_plot, family_table = None, None
head_plot, head_table = None, None
dataset_plot, dataset_table = None, None


def is_plot_update(request):
    """A request from the page's own script, which replaces the plot section in place."""
    return request.headers.get("X-Plot-Update") == "1"


def plot_update(request, form, get_result):
    """The plot section alone, or only the options while they are incomplete. The page has
    already loaded plotly.js, so the plot leaves it out; it is never stored in the globals
    that full page loads reuse."""
    ready = form.is_valid() and form.is_ready()
    plot, table = get_result(include_plotlyjs=False) if ready else (None, None)
    response = render(
        request,
        "components/plot_update.html",
        {"form": form, "plot": plot, "table": table, "ready": ready},
    )
    response["Cache-Control"] = "no-store"  # never shown in place of the full page
    return response



def all(request):
    global plot, table, headers
    form = PlotForm(request.GET or get_default_request())
    if is_plot_update(request):
        return plot_update(
            request, form, lambda **kw: get_plot_and_table(PlotRequest(form.cleaned_data), **kw)
        )
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        plot, table = get_plot_and_table(plot_request)
    return render(
        request,
        "stats/all.html",
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
    if any(field in request.GET for field in PlotForm.base_fields):
        form = PlotForm(request.GET)
    else:
        form = PlotForm(get_default_request(family=family, task_query=request.GET.get("task")))
    if is_plot_update(request):
        return plot_update(
            request,
            form,
            lambda **kw: get_plot_and_table(
                PlotRequest(form.cleaned_data), page="family", family_name=family_name, **kw
            ),
        )
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        family_plot, family_table = get_plot_and_table(
            plot_request, page="family", family_name=family_name
        )
    class_table = get_family_classification_table(family.name)
    det_tables = get_family_downstream_table(family.name, TaskType.DETECTION)
    instance_tables = get_family_downstream_table(family.name, TaskType.INSTANCE_SEG)
    semantic_tables = get_family_downstream_table(family.name, TaskType.SEMANTIC_SEG)

    return render(
        request,
        "stats/family.html",
        {
            "family": family,
            "form": form,
            "plot": family_plot,
            "table": family_table,
            "classification_table": class_table,
            "detection_tables": det_tables,
            "instance_tables": instance_tables,
            "semantic_tables": semantic_tables,
        },
    )


def head(request, head_name):
    global head_plot, head_table
    head = get_object_or_404(DownstreamHead, name=head_name)
    if any(field in request.GET for field in PlotForm.base_fields):
        form = PlotForm(request.GET, head=head)
    else:
        form = PlotForm(get_default_request(head=head, task_query=request.GET.get("task")), head=head)

    def head_result(**kwargs):
        form.cleaned_data["x_head"] = head
        form.cleaned_data["y_head"] = head
        return get_plot_and_table(PlotRequest(form.cleaned_data), page="head", **kwargs)

    if is_plot_update(request):
        return plot_update(request, form, head_result)
    if form.is_valid() and form.is_ready():
        head_plot, head_table = head_result()
    head_tasks = [task.name for task in head.tasks.all()]
    det_tables = None
    instance_tables = None
    semantic_tables = None
    if TaskType.DETECTION.value in head_tasks:
        det_tables = get_head_downstream_table(head.name, TaskType.DETECTION)
    if TaskType.INSTANCE_SEG.value in head_tasks:
        instance_tables = get_head_downstream_table(head.name, TaskType.INSTANCE_SEG)
    if TaskType.SEMANTIC_SEG.value in head_tasks:
        semantic_tables = get_head_downstream_table(head.name, TaskType.SEMANTIC_SEG)

    return render(
        request,
        "stats/head.html",
        {
            "head": head,
            "form": form,
            "plot": head_plot,
            "table": head_table,
            "detection_tables": det_tables,
            "instance_tables": instance_tables,
            "semantic_tables": semantic_tables,
        },
    )


def dataset(request, dataset_name):
    global dataset_plot, dataset_table
    dataset = get_object_or_404(Dataset, name=dataset_name)
    if any(field in request.GET for field in PlotForm.base_fields):
        form = PlotForm(request.GET, dataset=dataset)
    else:
        form = PlotForm(
            get_default_request(dataset=dataset, task_query=request.GET.get("task")),
            dataset=dataset,
        )

    def dataset_result(**kwargs):
        form.cleaned_data["x_dataset"] = dataset
        form.cleaned_data["y_dataset"] = dataset
        return get_plot_and_table(PlotRequest(form.cleaned_data), page="dataset", **kwargs)

    if is_plot_update(request):
        return plot_update(request, form, dataset_result)
    if form.is_valid() and form.is_ready():
        dataset_plot, dataset_table = dataset_result()
    dataset_tasks = [task.name for task in dataset.tasks.all()]
    classification_table = None
    det_table = None
    instance_table = None
    semantic_table = None
    if TaskType.CLASSIFICATION.value in dataset_tasks:
        classification_table = get_dataset_classification_table(dataset.name)
    if TaskType.DETECTION.value in dataset_tasks:
        det_table = get_dataset_downstream_table(dataset.name, TaskType.DETECTION)
    if TaskType.INSTANCE_SEG.value in dataset_tasks:
        instance_table = get_dataset_downstream_table(dataset.name, TaskType.INSTANCE_SEG)
    if TaskType.SEMANTIC_SEG.value in dataset_tasks:
        semantic_table = get_dataset_downstream_table(dataset.name, TaskType.SEMANTIC_SEG)

    return render(
        request,
        "stats/dataset.html",
        {
            "dataset": dataset,
            "form": form,
            "plot": dataset_plot,
            "table": dataset_table,
            "classification_table": classification_table,
            "detection_table": det_table,
            "instance_table": instance_table,
            "semantic_table": semantic_table,
        },
    )


def datasets(request):
    dataset_lists = get_dataset_lists()
    return render(request, "stats/datasets.html", {"datasets": dataset_lists})


def families(request):
    family_list = get_family_list()
    return render(request, "stats/families.html", {"families": family_list})


def heads(request):
    head_lists = get_head_lists()
    return render(request, "stats/heads.html", {"heads": head_lists})


def about(request):
    return render(request, "stats/about.html")

import random

from django import forms
from django.db.models.fields.related_lookups import In
from django.shortcuts import render
from django.views.generic import FormView
from django.db.models import Prefetch
from plotly.offline import plot
import plotly.graph_objs as go

from .models import (
    Dataset,
    DownstreamHead,
    Task,
    ClassificationResult,
    InstanceResult,
    PretrainedBackbone,
    TaskType,
)

AXIS_CHOICES = {
    "results": "Results",
    "gflops": "GFLOPs",
    "m_parameters": "Parameters (M)",
}

CLASSIFICATION_METRICS = {
    "": "----------",
    "top_1": "Top-1 Accuracy",
    "top_5": "Top-5 Accuracy",
}

INSTANCE_METRICS = {
    "": "----------",
    "mAP": "mAP",
    "AP50": "AP@50",
    "AP75": "AP@75",
    "mAPs": "mAP Small",
    "mAPm": "mAP Medium",
    "mAPl": "mAP Large",
}

FIELD_ORDER = [
    "y_axis",
    "y_task",
    "y_dataset",
    "y_metric",
    "y_head",
    "x_axis",
    "x_task",
    "x_dataset",
    "x_metric",
    "x_head",
]

HIDDEN = forms.HiddenInput


class DatasetTaskForm(forms.Form):
    y_axis = forms.ChoiceField(choices=AXIS_CHOICES, initial="results")
    x_axis = forms.ChoiceField(choices=AXIS_CHOICES, initial="gflops")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        args = args[0]
        if not args:
            self.init_defaults()
        else:
            for axis in ["x", "y"]:
                if args[f"{axis}_axis"] == "results":
                    self.init_extras(args, axis)

        self.reorder_fields()

    def init_defaults(self):
        self.fields["y_task"] = forms.ModelChoiceField(queryset=Task.objects.all(), initial=1)
        self.fields["y_dataset"] = forms.ModelChoiceField(queryset=Dataset.objects.all(), initial=1)
        self.fields["y_metric"] = forms.ChoiceField(choices=CLASSIFICATION_METRICS, initial="top_1")

    def init_extras(self, args, axis):
        self.fields[f"{axis}_task"] = forms.ModelChoiceField(
            queryset=Task.objects.all(), required=False
        )
        if args.get(f"{axis}_task"):
            task_name = Task.objects.get(pk=args[f"{axis}_task"]).name
            self.fields[f"{axis}_dataset"] = forms.ModelChoiceField(
                queryset=Dataset.objects.filter(tasks__name=task_name), required=False
            )
            if task_name == TaskType.CLASSIFICATION.value:
                self.fields[f"{axis}_metric"] = forms.ChoiceField(
                    choices=CLASSIFICATION_METRICS, required=False
                )
            else:
                self.fields[f"{axis}_metric"] = forms.ChoiceField(
                    choices=INSTANCE_METRICS, required=False
                )
                self.fields[f"{axis}_head"] = forms.ModelChoiceField(
                    queryset=DownstreamHead.objects.filter(tasks__name=task_name),
                    required=False,
                )

    def reorder_fields(self):
        field_names = list(self.fields.keys())
        self.order_fields(field_order=sorted(field_names, key=lambda x: FIELD_ORDER.index(x)))

    def is_ready(self):
        values = list(self.cleaned_data.values())
        return None not in values and "" not in values


def plot_view(request):
    form = DatasetTaskForm(request.GET or None)
    plot = None
    if form.is_valid() and form.is_ready():
        plot = get_plot(form.cleaned_data)
    return render(request, "view/all.html", {"form": form, "plot": plot})


def get_plot(plot_args):
    y_type = plot_args["y_axis"]
    x_type = plot_args["x_axis"]

    if x_type == "results" or y_type == "results":
        x_dataset = plot_args.get("x_dataset")
        x_task = plot_args.get("x_task")
        x_metric = plot_args.get("x_metric")
        x_head = plot_args.get("x_head")
        y_dataset = plot_args.get("y_dataset")
        y_task = plot_args.get("y_task")
        y_metric = plot_args.get("y_metric")
        y_head = plot_args.get("y_head")

        if x_type == "results" and y_type == "results":
            if x_dataset != y_dataset or x_task != y_task or x_head != y_head:
                data = get_multi_result_data(x_dataset, x_task, x_head, y_dataset, y_task, y_head)
                plot = get_multi_plot(data, x_metric, y_metric, x_dataset.name, y_dataset.name)
            else:
                data = get_single_result_data(x_dataset, x_task, x_head)
                plot = get_single_plot(data, x_metric, y_metric, x_dataset.name)

        elif x_type == "results":
            data = get_single_result_data(x_dataset, x_task, x_head)
            plot = get_single_plot(data, x_metric, y_type, x_dataset.name)

        else:
            data = get_single_result_data(y_dataset, y_task, y_head)
            plot = get_single_plot(data, x_type, y_metric, y_dataset.name)

    else:
        data = get_all_results_data()
        plot = get_all_plot(data, x_type, y_type)

    return plot


def get_multi_result_data(x_dataset, x_task, x_head, y_dataset, y_task, y_head):
    query = PretrainedBackbone.objects.select_related("family").select_related("backbone")
    if x_task.name == TaskType.CLASSIFICATION.value:
        query = filter_results(query, ClassificationResult, x_dataset).prefetch_related(
            get_result_prefetch("x_results", ClassificationResult, x_dataset)
        )
    else:
        query = filter_results(query, InstanceResult, x_dataset, x_task, x_head).prefetch_related(
            get_result_prefetch("x_results", InstanceResult, x_dataset, x_task, x_head)
        )
    if y_task.name == TaskType.CLASSIFICATION.value:
        query = filter_results(query, ClassificationResult, y_dataset).prefetch_related(
            get_result_prefetch("y_results", ClassificationResult, y_dataset)
        )
    else:
        query = filter_results(query, InstanceResult, y_dataset, y_task, y_head).prefetch_related(
            get_result_prefetch("y_results", InstanceResult, y_dataset, y_task, y_head)
        )
    return query


def get_all_results_data():
    return (
        PretrainedBackbone.objects.select_related("family")
        .select_related("backbone")
        .prefetch_related(get_result_prefetch("_classification_results", ClassificationResult))
        .prefetch_related(get_result_prefetch("_instance_results", InstanceResult))
    )


def get_single_result_data(dataset, task, head):
    query = PretrainedBackbone.objects.select_related("family").select_related("backbone")
    if task.name == TaskType.CLASSIFICATION.value:
        data = filter_results(query, ClassificationResult, dataset).prefetch_related(
            get_result_prefetch("filtered_results", ClassificationResult, dataset)
        )
    else:
        if head:
            data = filter_results(query, InstanceResult, dataset, task, head).prefetch_related(
                get_result_prefetch("filtered_results", InstanceResult, dataset, task, head)
            )
        else:
            data = filter_results(query, InstanceResult, dataset, task).prefetch_related(
                get_result_prefetch("filtered_results", InstanceResult, dataset, task)
            )
    return data


def filter_results(query, model, dataset=None, task=None, head=None, resolution=None):
    if model == ClassificationResult:
        filter_args = {
            "classification_results__dataset": dataset,
            "classification_results__resolution": resolution,
        }
    else:
        filter_args = {
            "instance_results__dataset": dataset,
            "instance_results__instance_type": task,
            "instance_results__head": head,
            "instance_results__fine_tune_resolution": resolution,
        }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    return query.filter(**filter_args) if filter_args else query


def get_result_prefetch(name, model, dataset=None, task=None, head=None, resolution=None):
    filter_args = {
        "dataset": dataset,
        "instance_type": task,
        "head": head,
        "fine_tune_resolution": resolution,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    queryset = model.objects.filter(**filter_args) if filter_args else model.objects.all()
    lookup = "classification_results" if model == ClassificationResult else "instance_results"
    return Prefetch(lookup, queryset=queryset, to_attr=name)


def get_single_plot(pretrained_backbones, x_type, y_type, dataset_name):
    y_title = get_axis_title(y_type)
    x_title = get_axis_title(x_type)
    data = []
    family_colors = {}
    used_families = set()

    for pb in pretrained_backbones:
        x_values = []
        y_values = []
        hovers = []
        for result in pb.filtered_results:
            x = pb.backbone.m_parameters if x_type == "m_parameters" else getattr(result, x_type)
            y = pb.backbone.m_parameters if y_type == "m_parameters" else getattr(result, y_type)
            hover = (
                f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
            )
            x_values.append(x)
            y_values.append(y)
            hovers.append(hover)

        if pb.family.name not in family_colors:
            family_colors[pb.family.name] = (
                f"rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.8)"
            )

        show_legend = pb.family.name not in used_families
        used_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=dict(size=10, color=family_colors[pb.family.name]),
                showlegend=show_legend,
            )
        )

    layout = go.Layout(
        title=f"{y_title} on {dataset_name}",
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        hovermode="closest",
    )
    fig = go.Figure(data=data, layout=layout)
    return plot(fig, output_type="div", include_plotlyjs=True)


def get_multi_plot(pretrained_backbones, x_type, y_type, x_dataset_name, y_dataset_name):
    y_title = get_axis_title(y_type)
    x_title = get_axis_title(x_type)
    data = []
    family_colors = {}
    used_families = set()

    for pb in pretrained_backbones:
        x_values = []
        y_values = []
        hovers = []
        for x_result in pb.x_results:
            for y_result in pb.y_results:
                x = getattr(x_result, x_type)
                y = getattr(y_result, y_type)
                hover = f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
                x_values.append(x)
                y_values.append(y)
                hovers.append(hover)

        if pb.family.name not in family_colors:
            family_colors[pb.family.name] = (
                f"rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.8)"
            )

        show_legend = pb.family.name not in used_families
        used_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=dict(size=10, color=family_colors[pb.family.name]),
                showlegend=show_legend,
            )
        )

    layout = go.Layout(
        title=f"{y_title} on {y_dataset_name} vs. {x_title} on {x_dataset_name}",
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        hovermode="closest",
    )
    fig = go.Figure(data=data, layout=layout)
    return plot(fig, output_type="div", include_plotlyjs=True)


def get_axis_title(db_name):
    return (
        AXIS_CHOICES.get(db_name)
        or INSTANCE_METRICS.get(db_name)
        or CLASSIFICATION_METRICS.get(db_name)
    )


def get_all_plot(pretrained_backbones, x_type, y_type):
    y_title = get_axis_title(y_type)
    x_title = get_axis_title(x_type)
    data = []
    family_colors = {}
    used_families = set()
    for pb in pretrained_backbones:
        x_values = []
        y_values = []
        hovers = []
        for results in [pb._instance_results, pb._classification_results]:
            for result in results:
                x = (
                    pb.backbone.m_parameters
                    if x_type == "m_parameters"
                    else getattr(result, x_type)
                )
                y = (
                    pb.backbone.m_parameters
                    if y_type == "m_parameters"
                    else getattr(result, y_type)
                )
                hover = f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
                x_values.append(x)
                y_values.append(y)
                hovers.append(hover)

        if pb.family.name not in family_colors:
            family_colors[pb.family.name] = (
                f"rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.8)"
            )

        show_legend = pb.family.name not in used_families
        used_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=dict(size=10, color=family_colors[pb.family.name]),
                showlegend=show_legend,
            )
        )

    layout = go.Layout(
        title=f"{y_title} against {x_title}",
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title),
        hovermode="closest",
    )

    fig = go.Figure(data=data, layout=layout)
    return plot(fig, output_type="div", include_plotlyjs=True)


def show_family(request, family):
    pass


def show_dataset(request, dataset):
    pass


def show_downstream_head(request, downstream_head):
    pass

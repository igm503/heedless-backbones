import random

from django import forms
from django.shortcuts import render
from django.db.models import Prefetch, Exists, OuterRef, F, Value, Subquery, FloatField
from django.db.models.functions import Coalesce
from plotly.offline import plot
import plotly.graph_objs as go

from .models import (
    Dataset,
    DownstreamHead,
    Task,
    ClassificationResult,
    InstanceResult,
    PretrainedBackbone,
    FPSMeasurement,
    TaskType,
    GPU,
    Precision,
)

AXIS_CHOICES = {
    "results": "Results",
    "gflops": "GFLOPs",
    "m_parameters": "Parameters (M)",
    "fps": "Images / Second",
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
    "y_gpu",
    "y_precision",
    "x_axis",
    "x_task",
    "x_dataset",
    "x_metric",
    "x_head",
    "x_gpu",
    "x_precision",
]

last_plot = None


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
                    self.init_result_extras(args, axis)
                elif args[f"{axis}_axis"] == "fps":
                    self.init_fps_extras(axis)

        self.reorder_fields()
        self.init_graph = False

    def init_defaults(self):
        self.fields["y_task"] = forms.ModelChoiceField(queryset=Task.objects.all(), initial=4)
        self.fields["y_dataset"] = forms.ModelChoiceField(queryset=Dataset.objects.all(), initial=1)
        self.fields["y_metric"] = forms.ChoiceField(choices=CLASSIFICATION_METRICS, initial="top_1")
        self.init_graph = True

    def init_result_extras(self, args, axis):
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

    def init_fps_extras(self, axis):
        self.fields[f"{axis}_gpu"] = forms.ChoiceField(
            choices={name.value: name.value for name in GPU}, required=False
        )
        self.fields[f"{axis}_precision"] = forms.ChoiceField(
            choices={name.value: name.value for name in Precision}, required=False
        )

    def reorder_fields(self):
        field_names = list(self.fields.keys())
        self.order_fields(field_order=sorted(field_names, key=lambda x: FIELD_ORDER.index(x)))

    def is_ready(self):
        values = list(self.cleaned_data.values())
        return self.init_graph or (None not in values and "" not in values)


def plot_view(request):
    global last_plot
    form = DatasetTaskForm(request.GET or None)
    plot = last_plot
    if form.is_valid() and form.is_ready():
        plot = get_plot(form.cleaned_data)
        last_plot = plot
    elif not request.GET:
        plot = get_default_plot()
        last_plot = plot
    return render(request, "view/all.html", {"form": form, "plot": plot})


def get_default_plot():
    y_dataset = Dataset.objects.get(pk=1)
    y_task = Task.objects.get(pk=4)
    y_metric = "top_1"
    data = get_single_result_data(y_dataset, y_task, None)
    plot = get_single_plot(data, "gflops", y_metric, y_dataset.name)
    return plot


# class GraphQuery:
#     def __init__(self, args):
#         self.data = data
#         self.x_metric = x_metric
#         self.y_metric = y_metric
#         self.x_dataset = x_dataset
#         self.y_dataset = y_dataset


def get_plot(plot_args):
    y_type = plot_args["y_axis"]
    x_type = plot_args["x_axis"]

    x_dataset = plot_args.get("x_dataset")
    x_task = plot_args.get("x_task")
    x_metric = plot_args.get("x_metric")
    x_head = plot_args.get("x_head")
    x_gpu = plot_args.get("x_gpu")
    x_precision = plot_args.get("x_precision")
    y_dataset = plot_args.get("y_dataset")
    y_task = plot_args.get("y_task")
    y_metric = plot_args.get("y_metric")
    y_head = plot_args.get("y_head")
    y_gpu = plot_args.get("y_gpu")
    y_precision = plot_args.get("y_precision")

    if x_type == "results" or y_type == "results":
        if x_type == "results" and y_type == "results":
            if x_dataset != y_dataset or x_task != y_task or x_head != y_head:
                data = get_multi_result_data(x_dataset, x_task, x_head, y_dataset, y_task, y_head)
                plot = get_multi_plot(data, x_metric, y_metric, x_dataset.name, y_dataset.name)
            else:
                data = get_single_result_data(x_dataset, x_task, x_head)
                plot = get_single_plot(data, x_metric, y_metric, x_dataset.name)

        elif x_type == "results":
            fps = y_gpu is not None and y_precision is not None
            data = get_single_result_data(
                x_dataset, x_task, x_head, fps=fps, gpu=y_gpu, precision=y_precision
            )
            plot = get_single_plot(data, x_metric, y_type, x_dataset.name)

        else:
            fps = x_gpu is not None and x_precision is not None
            data = get_single_result_data(
                y_dataset, y_task, y_head, fps=fps, gpu=x_gpu, precision=x_precision
            )
            plot = get_single_plot(data, x_type, y_metric, y_dataset.name)

    else:
        fps = (x_gpu is not None and x_precision is not None) or (
            y_gpu is not None and y_precision is not None
        )
        if fps:
            gpu = x_gpu or y_gpu
            precision = x_precision or y_precision
            data = get_all_results_data(fps=True, gpu=gpu, precision=precision)
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


def get_all_results_data(fps=False, gpu=None, precision=None, resolution=None):
    query = PretrainedBackbone.objects.select_related("family").select_related("backbone")
    query = filter_results(query, ClassificationResult, fps=fps, gpu=gpu, precision=precision)
    query = query.prefetch_related(
        get_result_prefetch(
            "_classification_results",
            ClassificationResult,
            fps=fps,
            gpu=gpu,
            precision=precision,
        )
    )
    query = query.prefetch_related(
        get_result_prefetch(
            "_instance_results", InstanceResult, fps=fps, gpu=gpu, precision=precision
        )
    )
    return query


def get_single_result_data(
    dataset, task, head, fps=False, gpu=None, precision=None, resolution=None
):
    if fps:
        assert gpu is not None
        assert precision is not None
    query = PretrainedBackbone.objects.select_related("family").select_related("backbone")
    if task.name == TaskType.CLASSIFICATION.value:
        data = filter_results(
            query,
            ClassificationResult,
            dataset,
            resolution=resolution,
            fps=fps,
            gpu=gpu,
            precision=precision,
        ).prefetch_related(
            get_result_prefetch(
                "filtered_results",
                ClassificationResult,
                dataset,
                resolution=resolution,
                fps=fps,
                gpu=gpu,
                precision=precision,
            )
        )

    else:
        data = filter_results(
            query,
            InstanceResult,
            dataset,
            task,
            head,
            fps=fps,
            gpu=gpu,
            precision=precision,
        ).prefetch_related(
            get_result_prefetch(
                "filtered_results",
                InstanceResult,
                dataset,
                task,
                head,
                fps=fps,
                gpu=gpu,
                precision=precision,
            )
        )
    return data


def filter_results(
    query,
    model,
    dataset=None,
    task=None,
    head=None,
    resolution=None,
    fps=False,
    gpu=None,
    precision=None,
):
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
            "instance_results__fine_tune_resolution": resolution,  # not in db yet
            "instance_results__fps_measurements__gpu": gpu,
            "instance_results__fps_measurements__precision": precision,
        }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}

    if fps and model == ClassificationResult:
        return query.filter(
            Exists(
                ClassificationResult.objects.filter(
                    pretrainedbackbone=OuterRef("pk"),
                    pretrainedbackbone__backbone__fps_measurements__resolution=F("resolution"),
                    pretrainedbackbone__backbone__fps_measurements__gpu=gpu,
                    pretrainedbackbone__backbone__fps_measurements__precision=precision,
                )
            ),
            **filter_args,
        ).distinct()

    else:
        return query.filter(**filter_args).distinct() if filter_args else query


def get_result_prefetch(
    name,
    model,
    dataset=None,
    task=None,
    head=None,
    resolution=None,
    fps=False,
    gpu=None,
    precision=None,
):
    filter_args = {
        "dataset": dataset,
        "instance_type": task,
        "head": head,
        "fine_tune_resolution": resolution,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    lookup = "classification_results" if model == ClassificationResult else "instance_results"
    if fps:
        if model == ClassificationResult:
            queryset = ClassificationResult.objects.filter(
                pretrainedbackbone__backbone__fps_measurements__resolution=F("resolution"),
                pretrainedbackbone__backbone__fps_measurements__gpu=gpu,
                pretrainedbackbone__backbone__fps_measurements__precision=precision,
                **filter_args,
            ).annotate(
                fps=Coalesce(
                    Subquery(
                        FPSMeasurement.objects.filter(
                            backbone__pretrainedbackbone__classification_results=OuterRef("pk"),
                            resolution=OuterRef("resolution"),
                            gpu=gpu,
                            precision=precision,
                        ).values("fps")[:1]
                    ),
                    Value(None),
                    output_field=FloatField(),
                )
            )
            return Prefetch(lookup, queryset=queryset, to_attr=name)
        else:
            filter_args["fps_measurements__gpu"] = gpu
            filter_args["fps_measurements__precision"] = precision
            queryset = (
                InstanceResult.objects.filter(**filter_args) if filter_args else model.objects.all()
            ).annotate(
                fps=Coalesce(
                    Subquery(
                        FPSMeasurement.objects.filter(
                            instanceresult=OuterRef("pk"),
                            # resolution=OuterRef("resolution"),
                            gpu=gpu,
                            precision=precision,
                        ).values("fps")[:1]
                    ),
                    Value(None),
                    output_field=FloatField(),
                )
            )
            return Prefetch(lookup, queryset=queryset, to_attr=name)
    else:
        queryset = model.objects.filter(**filter_args) if filter_args else model.objects.all()
        return Prefetch(lookup, queryset=queryset, to_attr=name)


def get_single_plot(pretrained_backbones, x_type, y_type, dataset_name):
    y_title = get_axis_title(y_type)
    x_title = get_axis_title(x_type)
    data = []
    families = {pb.family.name for pb in pretrained_backbones}
    marker_configs = get_marker_configs(families)
    seen_families = set()

    for pb in pretrained_backbones:
        x_values = []
        y_values = []
        hovers = []
        params = pb.backbone.m_parameters
        for result in pb.filtered_results:
            x = params if x_type == "m_parameters" else getattr(result, x_type)
            y = params if y_type == "m_parameters" else getattr(result, y_type)
            hover = (
                f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
            )
            x_values.append(x)
            y_values.append(y)
            hovers.append(hover)

        show_legend = pb.family.name not in seen_families
        seen_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=marker_configs[pb.family.name],
                showlegend=show_legend,
            )
        )

    title = f"{y_title} on {dataset_name}"
    return get_plot_div(title, x_title, y_title, data)


def get_multi_plot(pretrained_backbones, x_type, y_type, x_dataset_name, y_dataset_name):
    y_title = get_axis_title(y_type)
    x_title = get_axis_title(x_type)
    data = []
    families = {pb.family.name for pb in pretrained_backbones}
    marker_configs = get_marker_configs(families)
    seen_families = set()

    for pb in pretrained_backbones:
        x_values = []
        y_values = []
        hovers = []
        for x_result in pb.x_results:
            for y_result in pb.y_results:
                x = getattr(x_result, x_type)
                y = getattr(y_result, y_type)
                hover = f"<b>{pb.name}</b><br>Family: {pb.family.name}<br>{y_title}: {y:.2f}<br>{x_title}: {x:.2f}"
                x_values.append(x)
                y_values.append(y)
                hovers.append(hover)

        show_legend = pb.family.name not in seen_families
        seen_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=marker_configs[pb.family.name],
                showlegend=show_legend,
            )
        )
    title = f"{y_title} on {y_dataset_name} vs. {x_title} on {x_dataset_name}"
    return get_plot_div(title, x_title, y_title, data)


def get_all_plot(pretrained_backbones, x_type, y_type):
    y_title = get_axis_title(y_type)
    x_title = get_axis_title(x_type)
    data = []
    families = {pb.family.name for pb in pretrained_backbones}
    marker_configs = get_marker_configs(families)
    seen_families = set()

    for pb in pretrained_backbones:
        x_values = []
        y_values = []
        hovers = []
        params = pb.backbone.m_parameters
        for results in [pb._instance_results, pb._classification_results]:
            for result in results:
                x = params if x_type == "m_parameters" else getattr(result, x_type)
                y = params if y_type == "m_parameters" else getattr(result, y_type)
                hover = f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
                x_values.append(x)
                y_values.append(y)
                hovers.append(hover)

        show_legend = pb.family.name not in seen_families
        seen_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=marker_configs[pb.family.name],
                showlegend=show_legend,
            )
        )

    title = f"{y_title} against {x_title}"
    return get_plot_div(title, x_title, y_title, data)


def get_axis_title(db_name):
    return (
        AXIS_CHOICES.get(db_name)
        or INSTANCE_METRICS.get(db_name)
        or CLASSIFICATION_METRICS.get(db_name)
    )


def get_marker_configs(names):
    if not names:
        return {}
    hue_increment = 360 / len(names)
    hue = random.randint(0, 360)
    marker_configs = {}
    for name in names:
        color = f"hsla({hue},70%,50%,0.8)"
        marker_configs[name] = dict(size=10, line=dict(width=0, color="black"), color=color)
        hue += hue_increment
        hue %= 360
    return marker_configs


def get_plot_div(title, x_title, y_title, data):
    layout = go.Layout(
        title=dict(text=title, font=dict(size=24, color="black")),
        xaxis=dict(
            title=x_title,
            gridcolor="rgba(0,0,0,1)",
            linecolor="black",
            showline=True,
            zerolinecolor="black",
            zerolinewidth=1,
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="rgba(0,0,0,1)",
            linecolor="black",
            showline=True,
            zerolinecolor="black",
            zerolinewidth=1,
        ),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line=dict(
            color="black",
            width=1,
        ),
    )
    plot_div = plot(fig, output_type="div", include_plotlyjs=True, config={"responsive": True})
    plot_div = plot_div.replace("<div", '<div class="floating-plot"', 1)

    return plot_div


def show_family(request, family):
    pass


def show_dataset(request, dataset):
    pass


def show_downstream_head(request, downstream_head):
    pass

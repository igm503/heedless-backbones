import random
from dataclasses import dataclass

from django.db.models import Prefetch, Exists, OuterRef, F, Value, Subquery, FloatField
from django.db.models.functions import Coalesce
from plotly.offline import plot
import plotly.graph_objs as go

from .constants import CLASSIFICATION_METRICS, INSTANCE_METRICS, AXIS_CHOICES
from .models import (
    Dataset,
    DownstreamHead,
    Task,
    ClassificationResult,
    InstanceResult,
    PretrainedBackbone,
    FPSMeasurement,
    TaskType,
)


class PlotRequest:
    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"

    @dataclass
    class DataArgs:
        dataset: Dataset | None = None
        task: Task | None = None
        head: DownstreamHead | None = None
        resolution: int | None = None
        gpu: str | None = None
        precision: str | None = None
        fps: bool = False

        def __post_init__(self):
            self.fps = self.gpu is not None and self.precision is not None

    @dataclass
    class PlotArgs:
        x_attr: str
        y_attr: str
        dataset: Dataset | None = None
        x_dataset: Dataset | None = None
        y_dataset: Dataset | None = None

    def __init__(self, args):
        self.y_type = args["y_axis"]
        self.x_type = args["x_axis"]

        self.x_dataset = args.get("x_dataset")
        self.x_task = args.get("x_task")
        self.x_metric = args.get("x_metric")
        self.x_head = args.get("x_head")
        self.x_gpu = args.get("x_gpu")
        self.x_precision = args.get("x_precision")

        self.y_dataset = args.get("y_dataset")
        self.y_task = args.get("y_task")
        self.y_metric = args.get("y_metric")
        self.y_head = args.get("y_head")
        self.y_gpu = args.get("y_gpu")
        self.y_precision = args.get("y_precision")

        if self.y_type == "results" and self.x_type == "results":
            if (
                self.x_dataset == self.y_dataset
                and self.x_task == self.y_task
                and self.x_metric == self.y_metric
            ):
                self.query_type = PlotRequest.SINGLE
                self.init_single_two_metric()
            else:
                self.query_type = PlotRequest.MULTI
                self.init_multi()
        elif self.y_type == "results" or self.x_type == "results":
            self.query_type = PlotRequest.SINGLE
            self.init_single()
        else:
            self.query_type = PlotRequest.NONE
            self.init_none()

    def init_multi(self):
        self.plot_args = PlotRequest.PlotArgs(
            x_attr=self.x_metric,
            y_attr=self.y_metric,
            x_dataset=self.x_dataset,
            y_dataset=self.y_dataset,
        )
        self.x_args = PlotRequest.DataArgs(
            dataset=self.x_dataset,
            task=self.x_task,
            head=self.x_head,
        )
        self.y_args = PlotRequest.DataArgs(
            dataset=self.y_dataset,
            task=self.y_task,
            head=self.y_head,
        )

    def init_single(self):
        if self.x_type == "results":
            self.data_args = PlotRequest.DataArgs(
                dataset=self.x_dataset,
                task=self.x_task,
                head=self.x_head,
                gpu=self.y_gpu,
                precision=self.y_precision,
            )
            self.plot_args = PlotRequest.PlotArgs(
                x_attr=self.x_metric,
                y_attr=self.y_type,
                dataset=self.x_dataset,
            )
        else:
            self.data_args = PlotRequest.DataArgs(
                dataset=self.y_dataset,
                task=self.y_task,
                head=self.y_head,
                gpu=self.x_gpu,
                precision=self.x_precision,
            )
            self.plot_args = PlotRequest.PlotArgs(
                x_attr=self.x_type,
                y_attr=self.y_metric,
                dataset=self.y_dataset,
            )

    def init_single_two_metric(self):
        self.data_args = PlotRequest.DataArgs(
            dataset=self.x_dataset,
            task=self.x_task,
            head=self.x_head,
        )
        self.plot_args = PlotRequest.PlotArgs(
            x_attr=self.x_metric,
            y_attr=self.y_metric,
            dataset=self.x_dataset,
        )

    def init_none(self):
        self.data_args = PlotRequest.DataArgs(
            gpu=self.x_gpu or self.y_gpu,
            precision=self.x_precision or self.y_precision,
        )
        self.plot_args = PlotRequest.PlotArgs(x_attr=self.x_type, y_attr=self.y_type)


def get_plot(request):
    queryset = PretrainedBackbone.objects.select_related("family").select_related("backbone")
    if request.query_type == PlotRequest.MULTI:
        queryset = filter_and_add_results(queryset, request.x_args, "x_results")
        queryset = filter_and_add_results(queryset, request.y_args, "y_results")
        return get_multi_plot(queryset, request.plot_args)
    elif request.query_type == PlotRequest.SINGLE:
        queryset = filter_and_add_results(queryset, request.data_args, "filtered_results")
        return get_single_plot(queryset, request.plot_args)
    else:
        queryset = get_all_results_data(request.data_args)
        return get_all_plot(queryset, request.plot_args)


def get_all_results_data(args):
    queryset = PretrainedBackbone.objects.select_related("family").select_related("backbone")
    queryset = filter_by_classification(queryset, args)
    queryset = queryset.prefetch_related(get_instance_prefetch("_instance_results", args))
    return queryset.prefetch_related(get_classification_prefetch("_classification_results", args))


def filter_and_add_results(queryset, args, to_attr):
    if args.task.name == TaskType.CLASSIFICATION.value:
        queryset = filter_by_classification(queryset, args)
        return queryset.prefetch_related(get_classification_prefetch(to_attr, args))
    else:
        queryset = filter_by_instance(queryset, args)
        return queryset.prefetch_related(get_instance_prefetch(to_attr, args))


def filter_by_classification(query, args):
    filter_args = {
        "classification_results__dataset": args.dataset,
        "classification_results__resolution": args.resolution,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    if args.fps:
        return query.filter(
            Exists(
                ClassificationResult.objects.filter(
                    pretrainedbackbone=OuterRef("pk"),
                    pretrainedbackbone__backbone__fps_measurements__resolution=F("resolution"),
                    pretrainedbackbone__backbone__fps_measurements__gpu=args.gpu,
                    pretrainedbackbone__backbone__fps_measurements__precision=args.precision,
                )
            ),
            **filter_args,
        ).distinct()
    else:
        return query.filter(**filter_args).distinct() if filter_args else query


def filter_by_instance(query, args):
    filter_args = {
        "instance_results__dataset": args.dataset,
        "instance_results__instance_type": args.task,
        "instance_results__head": args.head,
        "instance_results__fine_tune_resolution": args.resolution,  # not in db yet
        "instance_results__fps_measurements__gpu": args.gpu,
        "instance_results__fps_measurements__precision": args.precision,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    return query.filter(**filter_args).distinct() if filter_args else query


def get_classification_prefetch(name, args):
    filter_args = {
        "dataset": args.dataset,
        "resolution": args.resolution,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    queryset = ClassificationResult.objects.filter(**filter_args)
    if args.fps:
        queryset = queryset.filter(
            pretrainedbackbone__backbone__fps_measurements__resolution=F("resolution"),
            pretrainedbackbone__backbone__fps_measurements__gpu=args.gpu,
            pretrainedbackbone__backbone__fps_measurements__precision=args.precision,
        ).annotate(
            fps=Coalesce(
                Subquery(
                    FPSMeasurement.objects.filter(
                        backbone__pretrainedbackbone__classification_results=OuterRef("pk"),
                        resolution=OuterRef("resolution"),
                        gpu=args.gpu,
                        precision=args.precision,
                    ).values("fps")[:1]
                ),
                Value(None),
                output_field=FloatField(),
            )
        )
    return Prefetch("classification_results", queryset=queryset, to_attr=name)


def get_instance_prefetch(name, args):
    filter_args = {
        "dataset": args.dataset,
        "instance_type": args.task,
        "head": args.head,
        "resolution": args.resolution,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    queryset = InstanceResult.objects.filter(**filter_args)
    if args.fps:
        queryset = queryset.filter(
            fps_measurements__gpu=args.gpu,
            fps_measurements__precision=args.precision,
        ).annotate(
            fps=Coalesce(
                Subquery(
                    FPSMeasurement.objects.filter(
                        instanceresult=OuterRef("pk"),
                        gpu=args.gpu,
                        precision=args.precision,
                    ).values("fps")[:1]
                ),
                Value(None),
                output_field=FloatField(),
            )
        )
    return Prefetch("instance_results", queryset=queryset, to_attr=name)


def get_plot_object(pbs, plot_args, get_pb_data, title_func):
    y_title = get_axis_title(plot_args.y_attr)
    x_title = get_axis_title(plot_args.x_attr)
    title = title_func(x_title, y_title)
    data = []
    families = {pb.family.name for pb in pbs}
    marker_configs = get_marker_configs(families)
    seen_families = set()

    for pb in pbs:
        x, y, hovers = get_pb_data(pb, plot_args.x_attr, plot_args.y_attr, x_title, y_title)

        show_legend = pb.family.name not in seen_families
        seen_families.add(pb.family.name)

        data.append(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=pb.family.name,
                text=hovers,
                hoverinfo="text",
                marker=marker_configs[pb.family.name],
                showlegend=show_legend,
            )
        )

    return get_plot_div(title, x_title, y_title, data)


def get_single_plot_data(pb, x_type, y_type, x_title, y_title):
    x_values, y_values, hovers = [], [], []
    params = pb.backbone.m_parameters
    for result in pb.filtered_results:
        x = params if x_type == "m_parameters" else getattr(result, x_type)
        y = params if y_type == "m_parameters" else getattr(result, y_type)
        hover = f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
        x_values.append(x)
        y_values.append(y)
        hovers.append(hover)
    return x_values, y_values, hovers


def get_multi_plot_data(pb, x_type, y_type, x_title, y_title):
    x_values, y_values, hovers = [], [], []
    for x_result in pb.x_results:
        for y_result in pb.y_results:
            x = getattr(x_result, x_type)
            y = getattr(y_result, y_type)
            hover = f"<b>{pb.name}</b><br>Family: {pb.family.name}<br>{y_title}: {y:.2f}<br>{x_title}: {x:.2f}"
            x_values.append(x)
            y_values.append(y)
            hovers.append(hover)
    return x_values, y_values, hovers


def get_all_plot_data(pb, x_type, y_type, x_title, y_title):
    x_values, y_values, hovers = [], [], []
    params = pb.backbone.m_parameters
    for results in [pb._instance_results, pb._classification_results]:
        for result in results:
            x = params if x_type == "m_parameters" else getattr(result, x_type)
            y = params if y_type == "m_parameters" else getattr(result, y_type)
            hover = (
                f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
            )
            x_values.append(x)
            y_values.append(y)
            hovers.append(hover)
    return x_values, y_values, hovers


def get_single_plot(queryset, plot_args):
    def title_func(x_title, y_title):
        return f"{y_title} on {plot_args.dataset.name}"

    return get_plot_object(queryset, plot_args, get_single_plot_data, title_func)


def get_multi_plot(queryset, plot_args):
    def title_func(x_title, y_title):
        return (
            f"{y_title} on {plot_args.y_dataset.name} vs. {x_title} on {plot_args.x_dataset.name}"
        )

    return get_plot_object(queryset, plot_args, get_multi_plot_data, title_func)


def get_all_plot(queryset, plot_args):
    def title_func(x_title, y_title):
        return f"{y_title} against {x_title}"

    return get_plot_object(queryset, plot_args, get_all_plot_data, title_func)


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


def get_default_plot():
    plot_request = PlotRequest(
        {
            "y_axis": "results",
            "x_axis": "gflops",
            "y_dataset": Dataset.objects.get(pk=1),
            "y_task": Task.objects.get(pk=4),
            "y_metric": "top_1",
        }
    )
    return get_plot(plot_request)

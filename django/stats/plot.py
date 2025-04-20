import random
from collections import defaultdict

from django.db.models import Prefetch, OuterRef, F, Subquery
from plotly.offline import plot
import plotly.graph_objs as go

from .request import PlotRequest
from .tables import get_plot_table
from .data_utils import (
    get_value,
    get_nested_attr,
    get_pretrain_string,
    get_finetune_string,
    get_train_string,
)
from .models import (
    ClassificationResult,
    SemanticSegmentationResult,
    InstanceResult,
    PretrainedBackbone,
    FPSMeasurement,
    TaskType,
)


def get_plot_and_table(plot_request, page="", family_name=None):
    queryset = get_plot_data(plot_request, family_name)
    plot = get_plot(queryset, plot_request)
    if queryset:
        table = get_plot_table(queryset, plot_request, page=page)
    else:
        table = None
    return plot, table


def get_plot_data(request, family_name=None):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")

    if family_name:
        queryset = queryset.filter(family__name=family_name)
    if request.pretrain_dataset:
        queryset = queryset.filter(pretrain_dataset=request.pretrain_dataset)
    if request.pretrain_method:
        queryset = queryset.filter(pretrain_method=request.pretrain_method)

    if request.query_type == PlotRequest.SINGLE:
        queryset = filter_and_add_results(
            queryset, request.data_args, "filtered_results"
        )
    elif request.query_type == PlotRequest.MULTI:
        queryset = filter_and_add_results(queryset, request.x_args, "x_results")
        queryset = filter_and_add_results(queryset, request.y_args, "y_results")
    else:
        queryset = get_all_results_data(queryset, request.data_args)

    return queryset


def get_all_results_data(queryset, args):
    queryset = filter_by_classification(queryset, args)
    queryset = queryset.prefetch_related(
        get_instance_prefetch("_instance_results", args)
    )
    queryset = queryset.prefetch_related(
        get_semantic_prefetch("_semantic_results", args)
    )
    return queryset.prefetch_related(
        get_classification_prefetch("_classification_results", args)
    )


def filter_and_add_results(queryset, args, to_attr):
    if args.task.name == TaskType.CLASSIFICATION.value:
        queryset = filter_by_classification(queryset, args)
        return queryset.prefetch_related(get_classification_prefetch(to_attr, args))
    elif args.task.name == TaskType.SEMANTIC_SEG.value:
        queryset = filter_by_semantic(queryset, args)
        return queryset.prefetch_related(get_semantic_prefetch(to_attr, args))
    else:
        queryset = filter_by_instance(queryset, args)
        return queryset.prefetch_related(get_instance_prefetch(to_attr, args))


def filter_by_classification(query, args):
    filter_args = {
        "classificationresult__dataset": args.dataset,
        "classificationresult__resolution": args.resolution,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    return query.filter(**filter_args).distinct() if filter_args else query


def filter_by_semantic(query, args):
    filter_args = {
        "semanticsegmentationresult__dataset": args.dataset,
        "semanticsegmentationresult__crop_size": args.resolution,
        "semanticsegmentationresult__head": args.head,
        "semanticsegmentationresult__fps_measurements__gpu": args.gpu,
        "semanticsegmentationresult__fps_measurements__precision": args.precision,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    return query.filter(**filter_args).distinct() if filter_args else query


def filter_by_instance(query, args):
    filter_args = {
        "instanceresult__dataset": args.dataset,
        "instanceresult__instance_type": args.task,
        "instanceresult__head": args.head,
        "instanceresult__fps_measurements__gpu": args.gpu,
        "instanceresult__fps_measurements__precision": args.precision,
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
        fps_subquery = FPSMeasurement.objects.filter(
            backbone=OuterRef("pretrained_backbone__backbone"),
            resolution=OuterRef("resolution"),
            gpu=args.gpu,
            precision=args.precision,
        )
        queryset = queryset.filter(
            pretrained_backbone__backbone__fps_measurements__resolution=F("resolution"),
            pretrained_backbone__backbone__fps_measurements__gpu=args.gpu,
            pretrained_backbone__backbone__fps_measurements__precision=args.precision,
        ).annotate(fps=Subquery(fps_subquery.values("fps")[:1]))
    return Prefetch("classificationresult_set", queryset, name)


def get_semantic_prefetch(name, args):
    filter_args = {
        "dataset": args.dataset,
        "crop_size": args.resolution,
        "head": args.head,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    queryset = SemanticSegmentationResult.objects.filter(**filter_args)
    if args.fps:
        fps_subquery = FPSMeasurement.objects.filter(
            semanticsegmentationresult=OuterRef("pk"),
            gpu=args.gpu,
            precision=args.precision,
        )
        queryset = queryset.filter(
            fps_measurements__gpu=args.gpu,
            fps_measurements__precision=args.precision,
        ).annotate(fps=Subquery(fps_subquery.values("fps")[:1]))
    return Prefetch("semanticsegmentationresult_set", queryset, name)


def get_instance_prefetch(name, args):
    filter_args = {
        "dataset": args.dataset,
        "instance_type": args.task,
        "head": args.head,
    }
    filter_args = {k: v for k, v in filter_args.items() if v is not None}
    queryset = InstanceResult.objects.filter(**filter_args)
    if args.fps:
        fps_subquery = FPSMeasurement.objects.filter(
            instanceresult=OuterRef("pk"),
            gpu=args.gpu,
            precision=args.precision,
        )
        queryset = queryset.filter(
            fps_measurements__gpu=args.gpu,
            fps_measurements__precision=args.precision,
        ).annotate(fps=Subquery(fps_subquery.values("fps")[:1]))
    return Prefetch("instanceresult_set", queryset, name)


def get_plot(queryset, request):
    args = request.plot_args
    x_title = args.x_title.split("&")[0]
    y_title = args.y_title.split("&")[0]

    data = defaultdict(lambda: defaultdict(list))
    keys = set()

    if request.query_type == PlotRequest.SINGLE:
        title = f"{y_title} on {args.dataset.name}"
        for pb in queryset:
            for result in pb.filtered_results:
                add_point(pb, result, result, args, x_title, y_title, data, keys)

    elif request.query_type == PlotRequest.MULTI:
        x_title += f" ({args.x_dataset})"
        y_title += f" ({args.y_dataset})"
        title = (
            f"{y_title} on {args.y_dataset.name} vs. {x_title} on {args.x_dataset.name}"
        )
        for pb in queryset:
            for x_result in pb.x_results:
                for y_result in pb.y_results:
                    add_point(
                        pb, x_result, y_result, args, x_title, y_title, data, keys
                    )
    else:
        title = f"{y_title} against {x_title}"
        for pb in queryset:
            for result in pb._instance_results + pb._classification_results:
                add_point(pb, result, result, args, x_title, y_title, data, keys)

    marker_configs = get_marker_configs(keys)

    scatters = [
        go.Scatter(
            x=points["x"],
            y=points["y"],
            mode="markers",
            name=str(key),
            text=points["hovers"],
            hoverinfo="text",
            marker=marker_configs[key],
        )
        for key, points in data.items()
    ]

    return get_plot_div(title, x_title, y_title, scatters)


def add_point(pb, x_result, y_result, args, x_title, y_title, data, keys):
    key = get_group_key(pb, x_result, y_result, args.group_by)
    if args.group_by_second:
        key += " / " + get_group_key(pb, x_result, y_result, args.group_by_second)
    x = get_value(pb, x_result, args.x_attr)
    y = get_value(pb, y_result, args.y_attr)
    if x is None or y is None:
        return
    hover = get_hover(pb, x, y, x_title, y_title, x_result, y_result)
    data[key]["x"].append(x)
    data[key]["y"].append(y)
    data[key]["hovers"].append(hover)
    keys.add(key)


def get_group_key(pb, x_result, y_result, attr):
    attrs = attr.split(".")
    if attrs[0] == "classification":
        assert (
            type(x_result) is ClassificationResult
            or type(y_result) is ClassificationResult
        )
        if type(x_result) is ClassificationResult:
            return get_nested_attr(x_result, attrs[1:])
        elif type(y_result) is ClassificationResult:
            return get_nested_attr(y_result, attrs[1:])
    elif attrs[0] == "instance":
        assert type(x_result) is InstanceResult or type(y_result) is InstanceResult
        if type(x_result) is InstanceResult:
            return get_nested_attr(x_result, attrs[1:])
        elif type(y_result) is InstanceResult:
            return get_nested_attr(y_result, attrs[1:])
    elif attrs[0] == "semantic":
        assert (
            type(x_result) is SemanticSegmentationResult
            or type(y_result) is SemanticSegmentationResult
        )
        if type(x_result) is SemanticSegmentationResult:
            return get_nested_attr(x_result, attrs[1:])
        elif type(y_result) is SemanticSegmentationResult:
            return get_nested_attr(y_result, attrs[1:])
    return get_nested_attr(pb, attrs)


def get_hover(pb, x, y, x_title, y_title, x_result, y_result):
    hover_elements = [
        f"Model: {pb.backbone.name}",
        f"Pretrain: {get_pretrain_string(pb)}",
    ]
    if x_result == y_result:
        if isinstance(x_result, ClassificationResult):
            hover_elements.append(f"Finetune: {get_finetune_string(x_result)}")
        else:
            hover_elements.append(f"Train: {get_train_string(x_result)}")
            hover_elements.append(f"Head: {x_result.head.name}")
    else:
        if isinstance(x_result, ClassificationResult):
            hover_elements.append(f"X Finetune: {get_finetune_string(x_result)}")
        else:
            hover_elements.append(f"X Train: {get_train_string(x_result)}")
            hover_elements.append(f"X Head: {x_result.head.name}")
        if isinstance(y_result, ClassificationResult):
            hover_elements.append(f"Y Finetune: {get_finetune_string(y_result)}")
        else:
            hover_elements.append(f"Y Train: {get_train_string(y_result)}")
            hover_elements.append(f"Y Head: {y_result.head.name}")
    hover_elements += [
        f"{y_title}: {y}",
        f"{x_title}: {x}",
    ]
    hover = "<br>".join(hover_elements)
    hover = hover.replace("&mdash;", "---")
    hover = hover.replace("&rarr;", "→")
    return hover


def get_marker_configs(names):
    if not names:
        return {}
    hue_increment = 360 / len(names)
    hue = random.randint(0, 360)
    marker_configs = {}
    for name in names:
        color = f"hsla({hue},80%,40%,0.8)"
        marker_configs[name] = dict(
            size=7, line=dict(width=0, color="black"), color=color
        )
        hue += hue_increment
        hue %= 360
    return marker_configs


def get_plot_div(title, x_title, y_title, data):
    font_settings = dict(
        family="Inter, sans-serif",
        color="black",
    )
    xaxis = dict(
        gridcolor="rgba(0,0,0,1)",
        linecolor="black",
        showline=True,
        zerolinecolor="black",
        zerolinewidth=1,
    )
    yaxis = xaxis.copy()
    if y_title in ["Images / Second", "Parameters (M)", "GFLOPs"]:
        yaxis["type"] = "log"
    if x_title in ["Images / Second", "Parameters (M)", "GFLOPs"]:
        xaxis["type"] = "log"
    x_axis = dict(title=dict(text=x_title), **xaxis)
    y_axis = dict(title=dict(text=y_title), **yaxis)

    layout = go.Layout(
        title=title,
        xaxis=x_axis,
        yaxis=y_axis,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=font_settings,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="black",
            borderwidth=1,
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            activecolor="#005080",
            color="black",
            remove=["lasso", "reset", "select"],
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    for trace in fig.data:
        if "name" in trace:
            trace.name = f'<span style="letter-spacing: 0.05em;">{trace.name}</span>'

    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line=dict(color="black", width=1),
    )

    return plot(
        fig, output_type="div", include_plotlyjs=True, config={"responsive": True}
    )

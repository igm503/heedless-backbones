import random
from dataclasses import dataclass
from collections import defaultdict

from django.db.models import Prefetch, Exists, OuterRef, F, Value, Subquery, FloatField
from django.db.models.functions import Coalesce
from django.urls import reverse
from plotly.offline import plot
import plotly.graph_objs as go

from .constants import (
    CLASSIFICATION_METRICS,
    INSTANCE_METRICS,
    AXIS_CHOICES,
    INSTANCE_SEG_METRICS,
    DETECTION_METRICS,
)
from .models import (
    Dataset,
    DownstreamHead,
    Task,
    ClassificationResult,
    InstanceResult,
    PretrainedBackbone,
    FPSMeasurement,
    TaskType,
    PretrainMethod,
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
        gpu: str | None = None
        precision: str | None = None
        fps: bool = False
        resolution: int | None = None

        def __post_init__(self):
            self.fps = self.gpu is not None and self.precision is not None

    @dataclass
    class PlotArgs:
        x_attr: str
        y_attr: str
        x_task: str | None = None
        y_task: str | None = None
        dataset: Dataset | None = None
        x_dataset: Dataset | None = None
        y_dataset: Dataset | None = None
        group_by: str | None = None
        group_by_second: str | None = None

    def __init__(self, args):
        self.y_type = args["y_axis"]
        self.x_type = args["x_axis"]

        self.x_dataset = args.get("x_dataset")
        self.x_task = args.get("x_task")
        self.x_metric = args.get("x_metric")
        self.x_head = args.get("x_head")
        self.x_gpu = args.get("x_gpu")
        self.x_precision = args.get("x_precision")
        self.x_resolution = args.get("x_resolution")
        self.x_resolution = int(self.x_resolution) if self.x_resolution else None

        self.y_dataset = args.get("y_dataset")
        self.y_task = args.get("y_task")
        self.y_metric = args.get("y_metric")
        self.y_head = args.get("y_head")
        self.y_gpu = args.get("y_gpu")
        self.y_precision = args.get("y_precision")
        self.y_resolution = args.get("y_resolution")
        self.y_resolution = int(self.y_resolution) if self.y_resolution else None

        self.pretrain_dataset = args.get("_pretrain_dataset")
        self.pretrain_method = args.get("_pretrain_method")

        self.query_type = None
        self.data_args = None
        self.plot_args = None
        self.x_args = None
        self.y_args = None

        if self.y_type == "results" and self.x_type == "results":
            if (
                self.x_dataset == self.y_dataset
                and self.x_task == self.y_task
                and self.x_head == self.y_head
                and self.x_resolution == self.y_resolution
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

        self.plot_args.group_by = args.get("legend_attribute")
        self.plot_args.group_by_second = args.get("legend_attribute_(second)")

    def init_multi(self):
        self.plot_args = PlotRequest.PlotArgs(
            x_attr=self.x_metric,
            y_attr=self.y_metric,
            x_dataset=self.x_dataset,
            y_dataset=self.y_dataset,
            x_task=self.x_task,
            y_task=self.y_task,
        )
        self.x_args = PlotRequest.DataArgs(
            dataset=self.x_dataset,
            task=self.x_task,
            head=self.x_head,
        )
        if self.x_task.name == TaskType.CLASSIFICATION.value:
            self.x_args.resolution = self.x_resolution
        self.y_args = PlotRequest.DataArgs(
            dataset=self.y_dataset,
            task=self.y_task,
            head=self.y_head,
        )
        if self.y_task.name == TaskType.CLASSIFICATION.value:
            self.y_args.resolution = self.y_resolution

    def init_single(self):
        if self.x_type == "results":
            self.data_args = PlotRequest.DataArgs(
                dataset=self.x_dataset,
                task=self.x_task,
                head=self.x_head,
                resolution=self.x_resolution,
                gpu=self.y_gpu,
                precision=self.y_precision,
            )
            self.plot_args = PlotRequest.PlotArgs(
                x_attr=self.x_metric,
                y_attr=self.y_type,
                dataset=self.x_dataset,
                x_task=self.x_task,
            )
        else:
            self.data_args = PlotRequest.DataArgs(
                dataset=self.y_dataset,
                task=self.y_task,
                head=self.y_head,
                resolution=self.y_resolution,
                gpu=self.x_gpu,
                precision=self.x_precision,
            )
            self.plot_args = PlotRequest.PlotArgs(
                x_attr=self.x_type,
                y_attr=self.y_metric,
                dataset=self.y_dataset,
                y_task=self.y_task,
            )

    def init_single_two_metric(self):
        self.data_args = PlotRequest.DataArgs(
            dataset=self.x_dataset,
            task=self.x_task,
            head=self.x_head,
            resolution=self.x_resolution,
        )
        self.plot_args = PlotRequest.PlotArgs(
            x_attr=self.x_metric,
            y_attr=self.y_metric,
            dataset=self.x_dataset,
            x_task=self.x_task,
            y_task=self.y_task,
        )

    def init_none(self):
        self.data_args = PlotRequest.DataArgs(
            gpu=self.x_gpu or self.y_gpu,
            precision=self.x_precision or self.y_precision,
        )
        self.plot_args = PlotRequest.PlotArgs(x_attr=self.x_type, y_attr=self.y_type)


def get_head_instance_table(head_name, instance_type):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    queryset = queryset.prefetch_related(
        Prefetch(
            "instance_results",
            queryset=InstanceResult.objects.select_related("dataset").filter(
                instance_type__name=instance_type.value,
                head__name=head_name,
            ),
            to_attr="results",
        )
    )

    eval_datasets = set()
    for pb in queryset:
        for result in pb.results:
            eval_datasets.add(f"{result.dataset.name}")

    result_keys = (
        DETECTION_METRICS
        if instance_type.value == TaskType.DETECTION.value
        else INSTANCE_SEG_METRICS
    )

    data = []
    for eval_dataset in eval_datasets:
        rows = []
        links = []
        for pb in queryset:
            pt_dataset = pb.pretrain_dataset.name
            if "ImageNet" in pt_dataset:
                pt_dataset = pt_dataset.replace("ImageNet", "IN")
            pretrain_method = pb.pretrain_method
            if pretrain_method == PretrainMethod.SUPERVISED.value:
                pretrain_method = "Sup."
            pretraining = f"{pt_dataset} : {pretrain_method} : {pb.pretrain_epochs}"
            for result in pb.results:
                if eval_dataset != result.dataset.name:
                    continue
                row = {
                    "family": pb.family.name,
                    "backbone": pb.backbone.name,
                    "pretraining": pretraining,
                    "head": result.head.name,
                    "epochs": result.train_epochs,
                    "gflops": result.gflops,
                    result_keys["mAP"]: result.mAP if result.mAP else "&mdash;",
                    result_keys["AP50"]: result.AP50 if result.AP50 else "&mdash;",
                    result_keys["AP75"]: result.AP75 if result.AP75 else "&mdash;",
                    result_keys["mAPs"]: result.mAPs if result.mAPs else "&mdash;",
                    result_keys["mAPm"]: result.mAPm if result.mAPm else "&mdash;",
                    result_keys["mAPl"]: result.mAPl if result.mAPl else "&mdash;",
                }
                rows.append(row)
                row_links = {
                    "family": reverse("family", args=[pb.family.name]),
                    "head": result.head.github,
                }
                if result.mAP:
                    row_links[result_keys["mAP"]] = result.paper
                if result.AP50:
                    row_links[result_keys["AP50"]] = result.paper
                if result.AP75:
                    row_links[result_keys["AP75"]] = result.paper
                if result.mAPs:
                    row_links[result_keys["mAPs"]] = result.paper
                if result.mAPm:
                    row_links[result_keys["mAPm"]] = result.paper
                if result.mAPl:
                    row_links[result_keys["mAPl"]] = result.paper
                links.append(row_links)

        headers = list(rows[0].keys()) if rows else []

        data.append(
            {
                "name": eval_dataset,
                "headers": headers,
                "rows": rows,
                "links": links,
            }
        )

    return data


def get_dataset_instance_table(dataset_name, instance_type):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    queryset = queryset.prefetch_related(
        Prefetch(
            "instance_results",
            queryset=InstanceResult.objects.select_related("dataset").filter(
                instance_type__name=instance_type.value,
                dataset__name=dataset_name,
            ),
            to_attr="results",
        )
    )

    result_keys = (
        DETECTION_METRICS
        if instance_type.value == TaskType.DETECTION.value
        else INSTANCE_SEG_METRICS
    )

    rows = []
    links = []
    for pb in queryset:
        pt_dataset = pb.pretrain_dataset.name
        if "ImageNet" in pt_dataset:
            pt_dataset = pt_dataset.replace("ImageNet", "IN")
        pretrain_method = pb.pretrain_method
        if pretrain_method == PretrainMethod.SUPERVISED.value:
            pretrain_method = "Sup."
        pretraining = f"{pt_dataset} : {pretrain_method} : {pb.pretrain_epochs}"
        for result in pb.results:
            row = {
                "family": pb.family.name,
                "backbone": pb.backbone.name,
                "pretraining": pretraining,
                "head": result.head.name,
                "epochs": result.train_epochs,
                "gflops": result.gflops,
                result_keys["mAP"]: result.mAP if result.mAP else "&mdash;",
                result_keys["AP50"]: result.AP50 if result.AP50 else "&mdash;",
                result_keys["AP75"]: result.AP75 if result.AP75 else "&mdash;",
                result_keys["mAPs"]: result.mAPs if result.mAPs else "&mdash;",
                result_keys["mAPm"]: result.mAPm if result.mAPm else "&mdash;",
                result_keys["mAPl"]: result.mAPl if result.mAPl else "&mdash;",
            }
            rows.append(row)
            row_links = {
                "family": reverse("family", args=[pb.family.name]),
                "head": reverse("head", args=[result.head.name]),
            }
            if result.mAP:
                row_links[result_keys["mAP"]] = result.paper
            if result.AP50:
                row_links[result_keys["AP50"]] = result.paper
            if result.AP75:
                row_links[result_keys["AP75"]] = result.paper
            if result.mAPs:
                row_links[result_keys["mAPs"]] = result.paper
            if result.mAPm:
                row_links[result_keys["mAPm"]] = result.paper
            if result.mAPl:
                row_links[result_keys["mAPl"]] = result.paper
            links.append(row_links)

    headers = list(rows[0].keys()) if rows else []

    return {
        "headers": headers,
        "rows": rows,
        "links": links,
    }


def get_family_instance_table(family_name, instance_type):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    queryset = queryset.filter(family__name=family_name)
    queryset = queryset.prefetch_related(
        Prefetch(
            "instance_results",
            queryset=InstanceResult.objects.select_related("dataset").filter(
                instance_type__name=instance_type.value
            ),
            to_attr="results",
        )
    )

    result_keys = (
        DETECTION_METRICS
        if instance_type.value == TaskType.DETECTION.value
        else INSTANCE_SEG_METRICS
    )

    eval_datasets = set()
    for pb in queryset:
        for result in pb.results:
            eval_datasets.add(f"{result.dataset.name}")

    data = []
    for eval_dataset in eval_datasets:
        rows = []
        links = []
        for pb in queryset:
            pt_dataset = pb.pretrain_dataset.name
            if "ImageNet" in pt_dataset:
                pt_dataset = pt_dataset.replace("ImageNet", "IN")
            pretrain_method = pb.pretrain_method
            if pretrain_method == PretrainMethod.SUPERVISED.value:
                pretrain_method = "Sup."
            pretraining = f"{pt_dataset} : {pretrain_method} : {pb.pretrain_epochs}"
            for result in pb.results:
                if eval_dataset != result.dataset.name:
                    continue
                row = {
                    "backbone": pb.backbone.name,
                    "pretraining": pretraining,
                    "head": result.head.name,
                    "epochs": result.train_epochs,
                    "gflops": result.gflops,
                    result_keys["mAP"]: result.mAP if result.mAP else "&mdash;",
                    result_keys["AP50"]: result.AP50 if result.AP50 else "&mdash;",
                    result_keys["AP75"]: result.AP75 if result.AP75 else "&mdash;",
                    result_keys["mAPs"]: result.mAPs if result.mAPs else "&mdash;",
                    result_keys["mAPm"]: result.mAPm if result.mAPm else "&mdash;",
                    result_keys["mAPl"]: result.mAPl if result.mAPl else "&mdash;",
                }
                rows.append(row)
                row_links = {
                    "backbone": pb.github,
                    "head": reverse("head", args=[result.head.name]),
                }
                if result.mAP:
                    row_links[result_keys["mAP"]] = result.paper
                if result.AP50:
                    row_links[result_keys["AP50"]] = result.paper
                if result.AP75:
                    row_links[result_keys["AP75"]] = result.paper
                if result.mAPs:
                    row_links[result_keys["mAPs"]] = result.paper
                if result.mAPm:
                    row_links[result_keys["mAPm"]] = result.paper
                if result.mAPl:
                    row_links[result_keys["mAPl"]] = result.paper

                links.append(row_links)

        headers = list(rows[0].keys()) if rows else []

        data.append(
            {
                "name": eval_dataset,
                "headers": headers,
                "rows": rows,
                "links": links,
            }
        )

    return data


def get_family_classification_table(family_name):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    queryset = queryset.filter(family__name=family_name)
    queryset = queryset.prefetch_related(
        Prefetch(
            "classification_results",
            queryset=ClassificationResult.objects.select_related("dataset", "fine_tune_dataset"),
            to_attr="results",
        )
    )

    eval_datasets = set()
    for pb in queryset:
        for result in pb.results:
            dataset = result.dataset.name
            if "ImageNet" in dataset:
                dataset = dataset.replace("ImageNet", "IN")
            eval_datasets.add(f"{dataset}")

    rows = []
    links = []
    for pb in queryset:
        params = pb.backbone.m_parameters
        result_finetunes = defaultdict(lambda: defaultdict(dict))
        for result in pb.results:
            ft_dataset = result.fine_tune_dataset.name if result.fine_tune_dataset else None
            if ft_dataset is not None and "ImageNet" in ft_dataset:
                ft_dataset = ft_dataset.replace("ImageNet", "IN")
            dataset = result.dataset.name
            if "ImageNet" in dataset:
                dataset = dataset.replace("ImageNet", "IN")
            finetune = f"{ft_dataset} : {result.fine_tune_epochs} : {result.fine_tune_resolution}"
            top1 = result.top_1 if result.top_1 else "&mdash;"
            top5 = result.top_5 if result.top_5 else "&mdash;"
            result_finetunes[finetune]["row"]["gflops"] = f"{result.gflops}"
            result_finetunes[finetune]["row"][f"{dataset}"] = f"{top1}/{top5}"
            if result.top_1 or result.top_5:
                result_finetunes[finetune]["links"][f"{dataset}"] = f"{result.paper}"

        pt_dataset = pb.pretrain_dataset.name
        if "ImageNet" in pt_dataset:
            pt_dataset = pt_dataset.replace("ImageNet", "IN")
        pretrain_method = pb.pretrain_method
        if pretrain_method == PretrainMethod.SUPERVISED.value:
            pretrain_method = "Sup."
        pretraining = f"{pt_dataset} : {pretrain_method} : {pb.pretrain_epochs}"
        for finetune, results in result_finetunes.items():
            if finetune == "None : None : None":
                finetune = finetune.replace("None", "&mdash;")
            row = {
                "backbone": pb.backbone.name,
                "params (m)": params,
                "pretraining": pretraining,
                "finetuning": finetune,
            }
            row.update(results["row"])
            for dataset in eval_datasets:
                if dataset not in row:
                    row[dataset] = "&mdash;/&mdash;"
            rows.append(row)
            row_links = {"backbone": pb.github}
            row_links.update(results["links"])
            links.append(row_links)

    headers = list(rows[0].keys()) if rows else []

    return {
        "headers": headers,
        "rows": rows,
        "links": links,
    }


def get_dataset_classification_table(dataset_name):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    queryset = queryset.prefetch_related(
        Prefetch(
            "classification_results",
            queryset=ClassificationResult.objects.select_related(
                "dataset", "fine_tune_dataset"
            ).filter(dataset__name=dataset_name),
            to_attr="results",
        )
    )

    rows = []
    links = []
    for pb in queryset:
        pt_dataset = pb.pretrain_dataset.name
        if "ImageNet" in pt_dataset:
            pt_dataset = pt_dataset.replace("ImageNet", "IN")
        pretrain_method = pb.pretrain_method
        if pretrain_method == PretrainMethod.SUPERVISED.value:
            pretrain_method = "Sup."
        pretraining = f"{pt_dataset} : {pretrain_method} : {pb.pretrain_epochs}"
        params = pb.backbone.m_parameters
        for result in pb.results:
            ft_dataset = result.fine_tune_dataset.name if result.fine_tune_dataset else None
            if ft_dataset is not None and "ImageNet" in ft_dataset:
                ft_dataset = ft_dataset.replace("ImageNet", "IN")
            finetune = f"{ft_dataset} : {result.fine_tune_epochs} : {result.fine_tune_resolution}"
            if finetune == "None : None : None":
                finetune = finetune.replace("None", "&mdash;")
            dataset = result.dataset.name
            if "ImageNet" in dataset:
                dataset = dataset.replace("ImageNet", "IN")
            top1 = result.top_1 if result.top_1 else "&mdash;"
            top5 = result.top_5 if result.top_5 else "&mdash;"
            row = {
                "family": pb.family.name,
                "backbone": pb.backbone.name,
                "params (m)": params,
                "pretraining": pretraining,
                "finetuning": finetune,
                "gflops": result.gflops,
                "top-1": top1,
                "top-5": top5,
            }
            row_links = {
                "family": reverse("family", args=[pb.family.name]),
            }
            if result.top_1:
                row_links["top-1"] = result.paper
            if result.top_5:
                row_links["top-5"] = result.paper
            rows.append(row)
            links.append(row_links)

    headers = list(rows[0].keys()) if rows else []

    return {
        "headers": headers,
        "rows": rows,
        "links": links,
    }


def get_plot_data(request, family_name=None):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    if family_name:
        queryset = queryset.filter(family__name=family_name)
    if request.pretrain_dataset:
        queryset = queryset.filter(pretrain_dataset=request.pretrain_dataset)
    if request.pretrain_method:
        queryset = queryset.filter(pretrain_method=request.pretrain_method)
    if request.query_type == PlotRequest.MULTI:
        queryset = filter_and_add_results(queryset, request.x_args, "x_results")
        return filter_and_add_results(queryset, request.y_args, "y_results")
    elif request.query_type == PlotRequest.SINGLE:
        return filter_and_add_results(queryset, request.data_args, "filtered_results")
    else:
        return get_all_results_data(queryset, request.data_args)


def get_table(queryset, request, page=""):
    plot_args = request.plot_args
    x_type = plot_args.x_attr
    y_type = plot_args.y_attr
    x_task = plot_args.x_task.name if plot_args.x_task else None
    y_task = plot_args.y_task.name if plot_args.y_task else None
    x_title = get_axis_title(x_type, x_task)
    y_title = get_axis_title(y_type, y_task)
    rows = []
    links = []
    if request.query_type == PlotRequest.SINGLE:
        for pb in queryset:
            params = pb.backbone.m_parameters
            for result in pb.filtered_results:
                row = {}
                row_links = {}
                if page == "family":
                    row_links["backbone"] = pb.github
                else:
                    row["family"] = pb.family.name
                    row_links["family"] = reverse("family", args=[pb.family.name])
                row["backbone"] = pb.backbone.name
                row["parameters (m)"] = params
                row["pretrain"] = pb.pretrain_dataset.name
                if hasattr(result, "head"):
                    row["head"] = result.head.name
                    if page == "head":
                        row_links["head"] = result.head.github
                    else:
                        row_links["head"] = reverse("head", args=[result.head.name])
                if x_type != "m_parameters":
                    row[f"{x_title}"] = get_value(pb, result, x_type)
                    row_links[f"{x_title}"] = result.paper
                if y_type != "m_parameters":
                    row[f"{y_title}"] = get_value(pb, result, y_type)
                    row_links[f"{y_title}"] = result.paper
                rows.append(row)
                links.append(row_links)

    elif request.query_type == PlotRequest.MULTI:
        for pb in queryset:
            params = pb.backbone.m_parameters
            for x_result in pb.x_results:
                for y_result in pb.y_results:
                    x = getattr(x_result, x_type)
                    y = getattr(y_result, y_type)
                    if x_title == y_title:
                        if y_result.dataset.name != x_result.dataset.name:
                            x_title = f"{x_title} ({x_result.dataset.name})"
                            y_title = f"{y_title} ({y_result.dataset.name})"
                        elif y_result.head.name != x_result.head.name:
                            x_title = f"{x_title} ({x_result.head.name})"
                            y_title = f"{y_title} ({y_result.head.name})"
                        else:
                            x_title = f"{x_title} (x)"
                            y_title = f"{y_title} (y)"
                    row = {}
                    row_links = {}
                    if page == "family":
                        row_links["backbone"] = pb.github
                    else:
                        row["family"] = pb.family.name
                        row_links["family"] = reverse("family", args=[pb.family.name])
                    row["backbone"] = pb.backbone.name
                    row["parameters (m)"] = params
                    row["pretrain"] = pb.pretrain_dataset.name
                    y_head = y_result.head.name if hasattr(y_result, "head") else False
                    if hasattr(x_result, "head"):
                        head_key = "x head" if y_head else "head"
                        row[head_key] = x_result.head.name
                        if page == "head":
                            row_links[head_key] = x_result.head.github
                        else:
                            row_links[head_key] = reverse("head", args=[x_result.head.name])
                    row[f"{x_title}"] = x
                    row_links[f"{x_title}"] = x_result.paper
                    if y_head:
                        row["y head"] = y_result.head.name
                        if page == "head":
                            row_links["y head"] = y_result.head.github
                        else:
                            row_links["y head"] = reverse("head", args=[y_result.head.name])

                    row[f"{y_title}"] = y
                    row_links[f"{y_title}"] = y_result.paper
                    rows.append(row)
                    links.append(row_links)
    else:
        for pb in queryset:
            params = pb.backbone.m_parameters
            for results in [pb._instance_results, pb._classification_results]:
                for result in results:
                    row = {}
                    row_links = {}
                    if page == "family":
                        row_links["backbone"] = pb.github
                    else:
                        row["family"] = pb.family.name
                        row_links["family"] = reverse("family", args=[pb.family.name])
                    row["backbone"] = pb.backbone.name
                    row["parameters (m)"] = params
                    row["pretrain"] = pb.pretrain_dataset.name
                    if x_type != "m_parameters":
                        row[f"{x_title}"] = get_value(pb, result, x_type)
                        row_links[f"{x_title}"] = result.paper
                    if y_type != "m_parameters":
                        row[f"{y_title}"] = get_value(pb, result, y_type)
                        row_links[f"{y_title}"] = result.paper
                    rows.append(row)
                    links.append(row_links)
    headers = list(rows[0].keys()) if rows else []
    return {
        "headers": headers,
        "rows": rows,
        "links": links,
    }


def get_all_results_data(queryset, args):
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
        # "resolution": args.resolution,
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


def get_plot(queryset, request):
    plot_args = request.plot_args
    x_task = plot_args.x_task.name if plot_args.x_task else None
    y_task = plot_args.y_task.name if plot_args.y_task else None
    x_type = plot_args.x_attr
    y_type = plot_args.y_attr
    y_title = get_axis_title(y_type, y_task)
    x_title = get_axis_title(x_type, x_task)

    grouped_data = defaultdict(lambda: defaultdict(list))
    group_keys = set()

    if request.query_type == PlotRequest.SINGLE:
        title = f"{y_title} on {plot_args.dataset.name}"
        for pb in queryset:
            for result in pb.filtered_results:
                if plot_args.group_by:
                    group_key = get_group_key(pb, result, plot_args.group_by)
                    if plot_args.group_by_second:
                        group_key += " / " + get_group_key(pb, result, plot_args.group_by_second)
                else:
                    group_key = pb.family.name
                x = get_value(pb, result, x_type)
                y = get_value(pb, result, y_type)
                hover = f"Model: {pb.name}<br>{y_title}: {y}<br>{x_title}: {x}"
                grouped_data[group_key]["x"].append(x)
                grouped_data[group_key]["y"].append(y)
                grouped_data[group_key]["hovers"].append(hover)
                group_keys.add(group_key)

    elif request.query_type == PlotRequest.MULTI:
        x_title += f" ({plot_args.x_dataset})"
        y_title += f" ({plot_args.y_dataset})"
        title = (
            f"{y_title} on {plot_args.y_dataset.name} vs. {x_title} on {plot_args.x_dataset.name}"
        )
        for pb in queryset:
            for x_result in pb.x_results:
                for y_result in pb.y_results:
                    if plot_args.group_by:
                        group_key = get_two_result_group_key(
                            pb, x_result, y_result, plot_args.group_by
                        )
                        if plot_args.group_by_second:
                            group_key += " / " + get_two_result_group_key(
                                pb, x_result, y_result, plot_args.group_by_second
                            )
                    else:
                        group_key = pb.family.name
                    x = get_value(pb, x_result, x_type)
                    y = get_value(pb, y_result, y_type)
                    hover = f"Model: {pb.name}<br>{y_title}: {y}<br>{x_title}: {x}"
                    grouped_data[group_key]["x"].append(x)
                    grouped_data[group_key]["y"].append(y)
                    grouped_data[group_key]["hovers"].append(hover)
                    group_keys.add(group_key)
    else:
        title = f"{y_title} against {x_title}"
        for pb in queryset:
            for results in [pb._instance_results, pb._classification_results]:
                for result in results:
                    if plot_args.group_by:
                        group_key = get_group_key(pb, result, plot_args.group_by)
                        if plot_args.group_by_second:
                            group_key += " / " + get_group_key(
                                pb, result, plot_args.group_by_second
                            )
                    else:
                        group_key = pb.family.name
                    x = get_value(pb, result, x_type)
                    y = get_value(pb, result, y_type)
                    hover = f"Model: {pb.name}<br>{y_title}: {y}<br>{x_title}: {x}"
                    grouped_data[group_key]["x"].append(x)
                    grouped_data[group_key]["y"].append(y)
                    grouped_data[group_key]["hovers"].append(hover)
                    group_keys.add(group_key)

    marker_configs = get_marker_configs(group_keys)

    data = [
        go.Scatter(
            x=group_data["x"],
            y=group_data["y"],
            mode="markers",
            name=str(group_key),
            text=group_data["hovers"],
            hoverinfo="text",
            marker=marker_configs[group_key],
        )
        for group_key, group_data in grouped_data.items()
    ]

    return get_plot_div(title, x_title, y_title, data)


def get_value(pb, result, attr):
    if attr == "m_parameters":
        return pb.backbone.m_parameters
    elif attr == "pub_date":
        return pb.family.pub_date
    else:
        return getattr(result, attr)


def get_two_result_group_key(pb, x_result, y_result, attr):
    attrs = attr.split(".")
    if attrs[0] == "classification":
        assert type(x_result) is ClassificationResult or type(y_result) is ClassificationResult
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
    return get_nested_attr(pb, attrs)


def get_group_key(pb, result, attr):
    attrs = attr.split(".")
    if attrs[0] == "classification":
        assert type(result) is ClassificationResult
        return get_nested_attr(result, attrs[1:])
    elif attrs[0] == "instance":
        assert type(result) is InstanceResult
        return get_nested_attr(result, attrs[1:])
    return get_nested_attr(pb, attrs)


def get_nested_attr(obj, attr_path):
    for attr in attr_path:
        obj = getattr(obj, attr)
    return str(obj)


def get_axis_title(db_name, instance_type=None):
    if db_name in INSTANCE_METRICS:
        if instance_type is None:
            title = INSTANCE_METRICS[db_name]
        elif instance_type == TaskType.DETECTION.value:
            title = DETECTION_METRICS[db_name]
        elif instance_type == TaskType.INSTANCE_SEG.value:
            title = INSTANCE_SEG_METRICS[db_name]
        else:
            title = db_name  # this shouldn't happen
    else:
        title = AXIS_CHOICES.get(db_name, "") or CLASSIFICATION_METRICS.get(db_name, "")
    return title


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
    font_settings = dict(
        family="Inter, sans-serif",
        color="black",
    )

    layout = go.Layout(
        title=dict(
            text=f'<span style="letter-spacing: 0.05em;">{title}</span>',
            font=dict(size=24, **font_settings),
        ),
        xaxis=dict(
            title=dict(text=f'<span style="letter-spacing: 0.05em;">{x_title}</span>'),
            gridcolor="rgba(0,0,0,1)",
            linecolor="black",
            showline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            titlefont=font_settings,
            tickfont=font_settings,
        ),
        yaxis=dict(
            title=dict(text=f'<span style="letter-spacing: 0.05em;">{y_title}</span>'),
            gridcolor="rgba(0,0,0,1)",
            linecolor="black",
            showline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            titlefont=font_settings,
            tickfont=font_settings,
        ),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=font_settings,
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            bordercolor="black",
            borderwidth=1,
            font=font_settings,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
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
        line=dict(
            color="black",
            width=1,
        ),
    )

    plot_div = plot(fig, output_type="div", include_plotlyjs=True, config={"responsive": True})
    return plot_div


def get_plot_and_table(plot_request, page=""):
    queryset = get_plot_data(plot_request)
    plot = get_plot(queryset, plot_request)
    table = get_table(queryset, plot_request, page=page)
    return plot, table

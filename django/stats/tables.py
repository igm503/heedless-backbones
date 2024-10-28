from collections import defaultdict
from urllib.parse import urlencode

from django.db.models import Prefetch
from django.urls import reverse

from .request import PlotRequest
from .data_utils import (
    get_value,
    get_pretrain_string,
    get_finetune_string,
    get_train_string,
)
from .constants import INSTANCE_SEG_METRICS, DETECTION_METRICS, SEMANTIC_SEG_METRICS
from .models import (
    ClassificationResult,
    InstanceResult,
    PretrainedBackbone,
    SemanticSegmentationResult,
    TaskType,
)

INSTANCE_TYPES = [TaskType.DETECTION.value, TaskType.INSTANCE_SEG.value]


def get_plot_table(queryset, request, page=""):
    if request.query_type == PlotRequest.SINGLE:
        return get_plot_table_single(queryset, request.plot_args, page)
    elif request.query_type == PlotRequest.MULTI:
        return get_plot_table_multi(queryset, request.plot_args, page)
    else:
        return get_plot_table_none(queryset, request.plot_args, page)


def get_plot_table_none(queryset, args, page):
    rows = []
    links = []
    for pb in queryset:
        for results in [pb._instance_results, pb._classification_results]:
            for result in results:
                if not check_axis_values(pb, result, args):
                    continue
                row, row_links = get_base_row(pb, page)
                row, row_links = add_axis_values(pb, result, args, row, row_links)
                rows.append(row)
                links.append(row_links)
    return package_table(rows, links)


def get_plot_table_single(queryset, args, page):
    rows = []
    links = []
    for pb in queryset:
        for result in pb.filtered_results:
            if not check_axis_values(pb, result, args):
                continue
            row, row_links = get_base_row(pb, page)
            if hasattr(result, "head"):
                add_head(result.head, row, row_links, "head", page)
                row["train"] = get_train_string(result)
            else:
                row["finetune"] = get_finetune_string(result)
            row, row_links = add_axis_values(pb, result, args, row, row_links)
            rows.append(row)
            links.append(row_links)
    return package_table(rows, links)


def get_plot_table_multi(queryset, args, page):
    x_title = args.x_title
    y_title = args.y_title
    rows = []
    links = []
    for pb in queryset:
        for x_result in pb.x_results:
            for y_result in pb.y_results:
                if hasattr(x_result, args.x_attr) is None or hasattr(y_result, args.y_attr) is None:
                    continue
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

                row, row_links = get_base_row(pb, page)

                x_head = x_result.head.name if hasattr(x_result, "head") else False
                y_head = y_result.head.name if hasattr(y_result, "head") else False
                if x_head:
                    head_key = "x head" if y_head else "head"
                    add_head(x_result.head, row, row_links, head_key, page)

                row[f"{x_title}"] = getattr(x_result, args.x_attr)
                row_links[f"{x_title}"] = get_paper(x_result, pb)

                if y_head:
                    head_key = "y head" if x_head else "head"
                    add_head(y_result.head, row, row_links, head_key, page)

                row[f"{y_title}"] = getattr(y_result, args.y_attr)
                row_links[f"{y_title}"] = get_paper(y_result, pb)

                rows.append(row)
                links.append(row_links)

    return package_table(rows, links)


def get_head_downstream_table(head_name, downstream_type):
    queryset = get_downstream_data(downstream_type, head_name=head_name)
    dataset_names = {result.dataset.name for pb in queryset for result in pb.results}
    data = []
    for dataset_name in dataset_names:
        table = get_downstream_table(queryset, dataset_name, downstream_type, page="head")
        table["name"] = dataset_name
        table["dataset_link"] = (
            reverse("dataset", args=[dataset_name])
            + f"?{urlencode({'task': downstream_type.value})}"
        )
        data.append(table)

    return data


def get_family_downstream_table(family_name, downstream_type):
    queryset = get_downstream_data(downstream_type, family_name=family_name)
    dataset_names = {result.dataset.name for pb in queryset for result in pb.results}
    data = []
    for dataset_name in dataset_names:
        table = get_downstream_table(queryset, dataset_name, downstream_type, page="family")
        table["name"] = dataset_name
        print(downstream_type.value)
        table["dataset_link"] = (
            reverse("dataset", args=[dataset_name])
            + f"?{urlencode({'task': downstream_type.value})}"
        )
        print(table["dataset_link"])
        data.append(table)

    return data


def get_dataset_downstream_table(dataset_name, downstream_type):
    queryset = get_downstream_data(downstream_type, dataset_name=dataset_name)
    return get_downstream_table(queryset, dataset_name, downstream_type)


def get_downstream_data(downstream_type, head_name=None, family_name=None, dataset_name=None):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    if family_name:
        queryset = queryset.filter(family__name=family_name)
    if downstream_type.value in INSTANCE_TYPES:
        prefetch_queryset = InstanceResult.objects.select_related("dataset").filter(
            instance_type__name=downstream_type.value
        )
    else:
        prefetch_queryset = SemanticSegmentationResult.objects.select_related("dataset")
    if head_name:
        prefetch_queryset = prefetch_queryset.filter(head__name=head_name)
    if dataset_name:
        prefetch_queryset = prefetch_queryset.filter(dataset__name=dataset_name)

    if downstream_type.value in INSTANCE_TYPES:
        queryset = queryset.prefetch_related(
            Prefetch("instanceresult_set", prefetch_queryset, "results")
        )
    else:
        queryset = queryset.prefetch_related(
            Prefetch("semanticsegmentationresult_set", prefetch_queryset, "results")
        )
    return queryset


def get_downstream_table(queryset, dataset_name, downstream_type, page=""):
    if downstream_type.value == TaskType.INSTANCE_SEG.value:
        result_keys = INSTANCE_SEG_METRICS
    elif downstream_type.value == TaskType.DETECTION.value:
        result_keys = DETECTION_METRICS
    else:
        result_keys = SEMANTIC_SEG_METRICS

    result_strs = list(result_keys.keys())
    result_strs.remove("gflops")

    rows = []
    links = []
    for pb in queryset:
        for result in pb.results:
            if dataset_name != result.dataset.name:
                continue
            row, row_links = get_base_row(pb, page)
            del row["params (m)"]
            row["head"] = result.head.name
            row["train"] = get_train_string(result)
            row["gflops"] = result.gflops
            row_links["head"] = reverse("head", args=[result.head.name])
            if page == "head":
                del row["head"]
            for result_str in result_strs:
                result_key = result_keys[result_str]
                result_val = getattr(result, result_str)
                if result_val is not None:
                    row[result_key] = result_val
                    row_links[result_key] = get_paper(result, pb)
                else:
                    row[result_key] = "&mdash;"
            rows.append(row)
            links.append(row_links)
    return package_table(rows, links)


def get_family_classification_table(family_name):
    queryset = get_classification_data(family_name=family_name)
    eval_datasets = {result.dataset.name for pb in queryset for result in pb.results}

    rows = []
    links = []
    for pb in queryset:
        result_finetunes = defaultdict(lambda: defaultdict(dict))
        for result in pb.results:
            finetune = get_finetune_string(result)
            top1 = result.top_1 if result.top_1 else "&mdash;"
            top5 = result.top_5 if result.top_5 else "&mdash;"
            dataset = result.dataset.name
            result_finetunes[finetune]["row"]["gflops"] = result.gflops
            result_finetunes[finetune]["row"][f"{dataset}"] = f"{top1}/{top5}"
            if result.top_1 or result.top_5:
                result_finetunes[finetune]["links"][f"{dataset}"] = get_paper(result, pb)

        for finetune, results in result_finetunes.items():
            row, row_links = get_base_row(pb, page="family")
            row["finetune"] = finetune
            row.update(results["row"])
            for dataset in eval_datasets:
                if dataset not in row:
                    row[dataset] = "&mdash;/&mdash;"
            row_links.update(results["links"])
            rows.append(row)
            links.append(row_links)

    for row in rows:
        for key in list(row.keys()):
            if "ImageNet" in key:
                new_key = key.replace("ImageNet", "IN")
                row[new_key] = row.pop(key)
                if "-C" in new_key:
                    row[new_key + "&darr;"] = row.pop(new_key)

    return package_table(rows, links)


def get_dataset_classification_table(dataset_name):
    queryset = get_classification_data(dataset_name=dataset_name)
    if "ImageNet-C" in dataset_name:
        top1_str = "top-1&darr;"
        top5_str = "top-5&darr;"
    else:
        top1_str = "top-1"
        top5_str = "top-5"

    rows = []
    links = []
    for pb in queryset:
        for result in pb.results:
            finetune = get_finetune_string(result)
            row, row_links = get_base_row(pb, "dataset")
            row["finetune"] = finetune
            row["gflops"] = result.gflops
            row[top1_str] = result.top_1 if result.top_1 else "&mdash;"
            row[top5_str] = result.top_5 if result.top_5 else "&mdash;"
            if result.top_1:
                row_links[top1_str] = get_paper(result, pb)
            if result.top_5:
                row_links[top5_str] = get_paper(result, pb)
            rows.append(row)
            links.append(row_links)

    return package_table(rows, links)


def get_classification_data(dataset_name=None, family_name=None):
    queryset = PretrainedBackbone.objects.select_related("family", "backbone")
    if family_name is not None:
        queryset = queryset.filter(family__name=family_name)
    prefetch_queryset = ClassificationResult.objects.select_related("dataset", "fine_tune_dataset")
    if dataset_name is not None:
        prefetch_queryset = prefetch_queryset.filter(dataset__name=dataset_name)
    return queryset.prefetch_related(
        Prefetch("classificationresult_set", prefetch_queryset, "results")
    )


def package_table(rows, links):
    if rows:
        headers = list(rows[0].keys())
        return {"headers": headers, "rows": rows, "links": links}
    else:
        return None


def get_base_row(pb, page):
    row = {
        "family": pb.family.name,
        "model": pb.backbone.name,
        "params (m)": pb.backbone.m_parameters,
        "pretrain": get_pretrain_string(pb),
    }
    if page == "family":
        del row["family"]
        row_links = {"model": get_github(pb)}
    else:
        row_links = {"family": reverse("family", args=[pb.family.name])}

    return row, row_links


def check_axis_values(pb, result, args):
    if args.x_attr != "m_parameters":
        if get_value(pb, result, args.x_attr) is None:
            return False
    if args.y_attr != "m_parameters":
        if get_value(pb, result, args.y_attr) is None:
            return False
    return True


def add_axis_values(pb, result, args, row, row_links):
    if args.x_attr != "m_parameters":
        row[f"{args.x_title}"] = get_value(pb, result, args.x_attr)
        row_links[f"{args.x_title}"] = get_paper(result, pb)
    if args.y_attr != "m_parameters":
        row[f"{args.y_title}"] = get_value(pb, result, args.y_attr)
        row_links[f"{args.y_title}"] = get_paper(result, pb)
    return row, row_links


def add_head(head, row, links, head_key, page):
    row[head_key] = head.name
    if page == "head":
        links[head_key] = head.github
    else:
        links[head_key] = reverse("head", args=[head.name])


def get_paper(result, pb):
    if result.paper:
        return result.paper
    elif pb.paper:
        return pb.paper
    elif pb.github:
        return pb.github
    elif pb.backbone.paper:
        return pb.backbone.paper
    elif pb.backbone.github:
        return pb.backbone.github
    else:
        return pb.backbone.family.paper


def get_github(pb):
    if pb.github:
        return pb.github
    elif pb.backbone.github:
        return pb.backbone.github
    else:
        return pb.backbone.family.github

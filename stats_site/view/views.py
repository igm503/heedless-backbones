from django import forms
from django.shortcuts import get_object_or_404, render

from .lists import get_family_list, get_head_lists, get_dataset_lists
from .plot import (
    PlotRequest,
    get_plot_and_table,
    get_family_classification_table,
    get_family_instance_table,
    get_head_instance_table,
    get_dataset_classification_table,
    get_dataset_instance_table,
)
from .constants import (
    AXIS_CHOICES,
    AXIS_WITH_GFLOPS,
    CLASSIFICATION_METRICS,
    INSTANCE_METRICS,
    FIELDS,
    RESOLUTIONS,
    LIMITED_LEGEND_ATTRIBUTES,
    CLASSIFICATION_LEGEND_ATTRIBUTES,
    INSTANCE_LEGEND_ATTRIBUTES,
)
from .models import (
    BackboneFamily,
    Backbone,
    Dataset,
    DownstreamHead,
    Task,
    TaskType,
    GPU,
    Precision,
    PretrainMethod,
)

plot, table = None, None
family_plot, family_table = None, None
head_plot, head_table = None, None
dataset_plot, dataset_table = None, None

dataset_lists = None
head_lists = None
family_list = None


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


class PlotForm(forms.Form):
    y_axis = forms.ChoiceField(choices=AXIS_CHOICES, initial="results")
    x_axis = forms.ChoiceField(choices=AXIS_WITH_GFLOPS, initial="gflops")

    def __init__(self, *args, **kwargs):
        if "head" in kwargs:
            self.head = kwargs["head"]
            del kwargs["head"]
        else:
            self.head = None
        if "dataset" in kwargs:
            self.dataset = kwargs["dataset"]
            del kwargs["dataset"]
        else:
            self.dataset = None

        super().__init__(*args, **kwargs)

        args = args[0]
        if args:
            if args["y_axis"] == "results":
                self.fields["y_axis"] = forms.ChoiceField(choices=AXIS_WITH_GFLOPS)
            else:
                self.fields["x_axis"] = forms.ChoiceField(choices=AXIS_CHOICES)
            if args["x_axis"] == "results":
                self.fields["y_axis"] = forms.ChoiceField(choices=AXIS_WITH_GFLOPS)
            else:
                self.fields["y_axis"] = forms.ChoiceField(choices=AXIS_CHOICES)
            for axis in ["x", "y"]:
                if args[f"{axis}_axis"] == "results":
                    self.init_result_extras(args, axis)
                elif args[f"{axis}_axis"] == "fps":
                    self.init_fps_extras(axis)
            self.init_legend(args)
        self.init_filters()

        self.reorder_fields()
        self.init_graph = False

    def init_result_extras(self, args, axis):
        if self.head:
            tasks = self.head.tasks.all()
            self.fields[f"{axis}_task"] = forms.ModelChoiceField(queryset=tasks, required=False)
        elif self.dataset:
            tasks = self.dataset.tasks.all()
            self.fields[f"{axis}_task"] = forms.ModelChoiceField(queryset=tasks, required=False)
        else:
            self.fields[f"{axis}_task"] = forms.ModelChoiceField(
                queryset=Task.objects.all(), required=False
            )
        if args.get(f"{axis}_task"):
            task_name = Task.objects.get(pk=args[f"{axis}_task"]).name
            if not self.dataset:
                self.fields[f"{axis}_dataset"] = forms.ModelChoiceField(
                    queryset=Dataset.objects.filter(tasks__name=task_name, eval=True),
                    required=False,
                )
            if task_name == TaskType.CLASSIFICATION.value:
                self.fields[f"{axis}_metric"] = forms.ChoiceField(
                    choices=CLASSIFICATION_METRICS, required=False
                )
                self.fields[f"{axis}_resolution"] = forms.ChoiceField(
                    choices=RESOLUTIONS, required=False
                )
            else:
                self.fields[f"{axis}_metric"] = forms.ChoiceField(
                    choices=INSTANCE_METRICS, required=False
                )
                if not self.head:
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

    def init_filters(self):
        self.fields["_pretrain_dataset"] = forms.ModelChoiceField(
            queryset=Dataset.objects.filter(pretrainedbackbone__isnull=False).distinct(),
            required=False,
        )
        pretrain_methods = {name.value: name.value for name in PretrainMethod}
        pretrain_methods[""] = "----------"
        self.fields["_pretrain_method"] = forms.ChoiceField(
            choices=pretrain_methods, required=False
        )

    def init_legend(self, args):
        group_attrs = LIMITED_LEGEND_ATTRIBUTES.copy()
        if args.get("x_task") or args.get("y_task"):
            x_task = Task.objects.get(pk=args["x_task"]).name if args.get("x_task") else None
            y_task = Task.objects.get(pk=args["y_task"]).name if args.get("y_task") else None
            if TaskType.CLASSIFICATION.value in [x_task, y_task]:
                group_attrs.update(CLASSIFICATION_LEGEND_ATTRIBUTES)
            if TaskType.INSTANCE_SEG.value in [x_task, y_task]:
                group_attrs.update(INSTANCE_LEGEND_ATTRIBUTES)
            elif TaskType.DETECTION.value in [x_task, y_task]:
                group_attrs.update(INSTANCE_LEGEND_ATTRIBUTES)
            if args.get("legend_attribute"):
                second_group_attrs = group_attrs.copy()
                del second_group_attrs[args["legend_attribute"]]
                self.fields["legend_attribute_(second)"] = forms.ChoiceField(
                    choices=second_group_attrs, required=False
                )
            if args.get("legend_attribute_(second)"):
                del group_attrs[args["legend_attribute_(second)"]]
        self.fields["legend_attribute"] = forms.ChoiceField(choices=group_attrs, required=False)

    def reorder_fields(self):
        field_names = list(self.fields.keys())
        self.order_fields(field_order=sorted(field_names, key=lambda x: FIELDS.index(x)))

    def is_ready(self):
        values = [
            v
            for k, v in self.cleaned_data.items()
            if k
            not in [
                "_pretrain_dataset",
                "_pretrain_method",
                "x_resolution",
                "y_resolution",
                "x_head",
                "y_head",
                "legend_attribute",
                "legend_attribute_(second)",
            ]
        ]
        return self.init_graph or (None not in values and "" not in values)


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
        family_plot, family_table = get_plot_and_table(plot_request, page="family")
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


def list_datasets(request):
    global dataset_lists
    if dataset_lists is None:
        dataset_lists = get_dataset_lists()
    print(dataset_lists)
    return render(request, "view/all_datasets.html", {"datasets": dataset_lists})


def list_families(request):
    global family_list
    if family_list is None:
        family_list = get_family_list()
    return render(request, "view/all_families.html", {"families": family_list})


def list_heads(request):
    global head_lists
    if head_lists is None:
        head_lists = get_head_lists()
    return render(request, "view/all_heads.html", {"heads": head_lists})

from django import forms
from django.shortcuts import get_object_or_404, render

from .plot import (
    PlotRequest,
    get_plot_data,
    get_plot,
    get_table,
    get_defaults,
    get_family_classification_table,
    get_family_instance_table,
    get_head_instance_table,
    get_dataset_classification_table,
    get_dataset_instance_table,
)
from .constants import (
    AXIS_CHOICES,
    CLASSIFICATION_METRICS,
    INSTANCE_METRICS,
    FIELDS,
    RESOLUTIONS,
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
)


plot, table = None, None
family_plot, family_table = None, None
head_plot, head_table = None, None
dataset_plot, dataset_table = None, None


class PlotForm(forms.Form):
    y_axis = forms.ChoiceField(choices=AXIS_CHOICES, initial="results")
    x_axis = forms.ChoiceField(choices=AXIS_CHOICES, initial="gflops")

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
        if not args:
            self.init_defaults()
        else:
            for axis in ["x", "y"]:
                if args[f"{axis}_axis"] == "results":
                    self.init_result_extras(args, axis)
                elif args[f"{axis}_axis"] == "fps":
                    self.init_fps_extras(axis)
        self.init_filters()

        self.reorder_fields()
        self.init_graph = False

    def init_defaults(self):
        if self.head:
            tasks = self.head.tasks.all()
            first_task = tasks.first()
            datasets = Dataset.objects.filter(tasks__in=tasks, eval=True).distinct().all()
            first_dataset = datasets.filter(tasks=first_task).all()[1]
            self.fields["y_task"] = forms.ModelChoiceField(queryset=tasks, initial=first_task.pk)
            self.fields["y_dataset"] = forms.ModelChoiceField(
                queryset=datasets,
                initial=first_dataset.pk,
            )
            self.fields["y_metric"] = forms.ChoiceField(choices=INSTANCE_METRICS, initial="mAP")
            self.init_graph = True
        elif self.dataset:
            tasks = self.dataset.tasks.all()
            first_task = tasks.first()
            self.fields["y_task"] = forms.ModelChoiceField(queryset=tasks, initial=first_task.pk)
            if first_task.name == TaskType.CLASSIFICATION.value:
                self.fields["y_metric"] = forms.ChoiceField(
                    choices=CLASSIFICATION_METRICS, initial="top_1"
                )
            else:
                self.fields["y_metric"] = forms.ChoiceField(choices=INSTANCE_METRICS, initial="mAP")
                self.fields["y_head"] = forms.ModelChoiceField(
                    queryset=DownstreamHead.objects.filter(tasks__name=first_task.name),
                    required=False,
                )
        else:
            self.fields["y_task"] = forms.ModelChoiceField(queryset=Task.objects.all(), initial=4)
            self.fields["y_dataset"] = forms.ModelChoiceField(
                queryset=Dataset.objects.filter(eval=True), initial=1
            )
            self.fields["y_metric"] = forms.ChoiceField(
                choices=CLASSIFICATION_METRICS, initial="top_1"
            )
            self.init_graph = True

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
        self.fields["_resolution"] = forms.ChoiceField(choices=RESOLUTIONS, required=False)
        self.fields["_pretrain_dataset"] = forms.ModelChoiceField(
            queryset=Dataset.objects.filter(pretrainedbackbone__isnull=False).distinct(),
            required=False,
        )

    def reorder_fields(self):
        field_names = list(self.fields.keys())
        self.order_fields(field_order=sorted(field_names, key=lambda x: FIELDS.index(x)))

    def is_ready(self):
        values = [
            v
            for k, v in self.cleaned_data.items()
            if k not in ["_pretrain_dataset", "_resolution", "x_head", "y_head"]
        ]
        return self.init_graph or (None not in values and "" not in values)


def show_all(request):
    global plot, table, headers
    form = PlotForm(request.GET or None)
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        queryset = get_plot_data(plot_request)
        plot = get_plot(queryset, plot_request)
        table = get_table(queryset, plot_request)
    else:
        plot, table = get_defaults()
    return render(
        request,
        "view/all.html",
        {
            "form": form,
            "plot": plot,
            "table": table,
        },
    )


def show_family(request, family_name):
    global family_plot, family_table
    try:
        family = BackboneFamily.objects.get(name=family_name)
    except BackboneFamily.DoesNotExist:
        backbone = get_object_or_404(Backbone, name=family_name)
        family = backbone.family
    form = PlotForm(request.GET or None)
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        queryset = get_plot_data(plot_request, family_name=family.name)
        family_plot = get_plot(queryset, plot_request)
        family_table = get_table(queryset, plot_request, page="backbone_family")
    else:
        family_plot, family_table = get_defaults(family_name=family.name)
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


def show_downstream_head(request, downstream_head_name):
    global head_plot, head_table
    head = get_object_or_404(DownstreamHead, name=downstream_head_name)
    form = PlotForm(request.GET or None, head=head)
    if form.is_valid() and form.is_ready():
        form.cleaned_data["x_head"] = head
        form.cleaned_data["y_head"] = head
        plot_request = PlotRequest(form.cleaned_data)
        queryset = get_plot_data(plot_request)
        head_plot = get_plot(queryset, plot_request)
        head_table = get_table(queryset, plot_request, page="downstream_head")
    else:
        head_plot, head_table = get_defaults(head=head)
    head_tasks = [task.name for task in head.tasks.all()]
    det_tables = None
    instance_tables = None
    if TaskType.DETECTION.value in head_tasks:
        det_tables = get_head_instance_table(head.name, TaskType.DETECTION)
    if TaskType.INSTANCE_SEG.value in head_tasks:
        instance_tables = get_head_instance_table(head.name, TaskType.INSTANCE_SEG)

    return render(
        request,
        "view/downstream_head.html",
        {
            "downstream_head": head,
            "form": form,
            "plot": head_plot,
            "table": head_table,
            "detection_tables": det_tables,
            "instance_tables": instance_tables,
        },
    )


def show_dataset(request, dataset_name):
    global dataset_plot, dataset_table
    dataset = get_object_or_404(Dataset, name=dataset_name)
    form = PlotForm(request.GET or None, dataset=dataset)
    if form.is_valid() and form.is_ready():
        form.cleaned_data["x_dataset"] = dataset
        form.cleaned_data["y_dataset"] = dataset
        plot_request = PlotRequest(form.cleaned_data)
        queryset = get_plot_data(plot_request)
        dataset_plot = get_plot(queryset, plot_request)
        dataset_table = get_table(queryset, plot_request, page="dataset")
    else:
        dataset_plot, dataset_table = get_defaults(dataset=dataset)
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

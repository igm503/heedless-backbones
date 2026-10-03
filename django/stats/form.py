from collections import Counter

from django import forms

from .categories import choices as category_choices

from .models import (
    Task,
    InstanceResult,
    TASK_TO_TABLE,
    TASK_TO_FIRST_METRIC,
    Dataset,
    DownstreamHead,
    FPSMeasurement,
    TaskType,
    GPU,
    Precision,
)
from .constants import (
    AXIS_CHOICES,
    AXIS_WITH_GFLOPS,
    CLASSIFICATION_METRICS,
    IMAGENET_C_METRICS,
    IMAGENET_C_BAR_METRICS,
    INSTANCE_METRICS,
    SEMANTIC_SEG_METRICS,
    FIELDS,
    CLASSIFICATION_RESOLUTIONS,
    SEMANTIC_SEG_RESOLUTIONS,
    LIMITED_LEGEND_ATTRIBUTES,
    CLASSIFICATION_LEGEND_ATTRIBUTES,
    INSTANCE_LEGEND_ATTRIBUTES,
    SEMANTIC_SEG_LEGEND_ATTRIBUTES,
)


class PlotForm(forms.Form):
    y_axis = forms.ChoiceField(choices=AXIS_CHOICES, initial="results")
    x_axis = forms.ChoiceField(choices=AXIS_WITH_GFLOPS, initial="gflops")

    def __init__(self, *args, **kwargs):
        self.head = kwargs.pop("head", None)
        self.dataset = kwargs.pop("dataset", None)

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
            self.fields[f"{axis}_task"] = forms.ModelChoiceField(
                queryset=self.head.tasks.all(), required=False
            )
        elif self.dataset:
            tasks = self.dataset.tasks
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
                dataset_pk = args.get(f"{axis}_dataset")
                dataset_name = Dataset.objects.get(pk=dataset_pk).name if dataset_pk else None
                metrics = {
                    "ImageNet-C": IMAGENET_C_METRICS,
                    "ImageNet-C-bar": IMAGENET_C_BAR_METRICS,
                }.get(dataset_name, CLASSIFICATION_METRICS)
                self.fields[f"{axis}_metric"] = forms.ChoiceField(
                    choices=metrics, required=False
                )
                self.fields[f"{axis}_resolution"] = forms.ChoiceField(
                    choices=CLASSIFICATION_RESOLUTIONS, required=False
                )
            else:
                if task_name == TaskType.SEMANTIC_SEG.value:
                    self.fields[f"{axis}_metric"] = forms.ChoiceField(
                        choices=SEMANTIC_SEG_METRICS, required=False
                    )
                    self.fields[f"{axis}_resolution"] = forms.ChoiceField(
                        choices=SEMANTIC_SEG_RESOLUTIONS, required=False
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
        pretrain_methods = {"": "----------"}
        pretrain_methods.update(category_choices("pretrain_method"))
        self.fields["_pretrain_method"] = forms.ChoiceField(
            choices=pretrain_methods, required=False
        )
        self.fields["_show_spiking"] = forms.BooleanField(
            label="Show spiking networks", required=False
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
            if TaskType.DETECTION.value in [x_task, y_task]:
                group_attrs.update(INSTANCE_LEGEND_ATTRIBUTES)
            if TaskType.SEMANTIC_SEG.value in [x_task, y_task]:
                group_attrs.update(SEMANTIC_SEG_LEGEND_ATTRIBUTES)
            if args.get("legend_attribute"):
                second_group_attrs = group_attrs.copy()
                del second_group_attrs[args["legend_attribute"]]
                self.fields["legend_attribute_(second)"] = forms.ChoiceField(
                    choices=second_group_attrs, required=False
                )
            if second := args.get("legend_attribute_(second)"):
                if second in group_attrs:
                    del group_attrs[args["legend_attribute_(second)"]]
        del group_attrs[""]
        self.fields["legend_attribute"] = forms.ChoiceField(choices=group_attrs, required=True)

    def reorder_fields(self):
        field_names = list(self.fields.keys())
        self.order_fields(field_order=sorted(field_names, key=lambda x: FIELDS.index(x)))

    def is_ready(self):
        values = [v for k, v in self.cleaned_data.items() if k not in OPTIONAL_FIELDS]
        return self.init_graph or (None not in values and "" not in values)


# Fields a plot does not need; every other field must have a value.
OPTIONAL_FIELDS = {
    "_pretrain_dataset",
    "_pretrain_method",
    "_show_spiking",
    "x_resolution",
    "y_resolution",
    "x_head",
    "y_head",
    "legend_attribute",
    "legend_attribute_(second)",
}


def complete_plot_form(args, **form_kwargs):
    """A PlotForm for args, with required options that are blank or no longer valid with defaults, and clear
    invalid optional ones, so every combination of options draws a plot. Changing one option
    can invalidate another (GFLOPs is only an x axis against results; a new task has
    different datasets and metrics), and the plot would otherwise wait for it silently."""
    data = {key: value for key, value in args.items()}
    for _ in range(len(FIELDS)):
        form = PlotForm(data, **form_kwargs)
        if form.is_valid() and form.is_ready():
            return form
        changed = False
        for name, field in form.fields.items():
            if isinstance(field, forms.BooleanField):
                continue
            valid = field_values(field)
            value = str(data.get(name) or "")
            if value in valid:
                continue
            if name in OPTIONAL_FIELDS and name != "legend_attribute":
                if value:
                    data[name] = ""
                    changed = True
                continue
            default = default_value(name, valid, data)
            if default is not None and default != value:
                data[name] = default
                changed = True
        if not changed:
            break
    return PlotForm(data, **form_kwargs)


def field_values(field):
    if isinstance(field, forms.ModelChoiceField):
        return [str(pk) for pk in field.queryset.values_list("pk", flat=True)]
    return [str(value) for value, _ in field.choices if value != ""]


def default_value(name, valid, data):
    if not valid:
        return None
    axis, _, kind = name.partition("_")
    if axis not in ("x", "y"):
        kind = name
    other = "y" if axis == "x" else "x"
    if kind == "axis":
        preferred = ["gflops"] if data.get(f"{other}_axis") == "results" else ["results"]
    elif kind == "task":
        preferred = [str(pk) for pk in Task.objects.filter(name=TaskType.CLASSIFICATION.value).values_list("pk", flat=True)]
    elif kind == "dataset":
        preferred = [str(pk) for pk in Dataset.objects.filter(name="ImageNet-1k").values_list("pk", flat=True)]
    elif kind == "gpu":
        preferred = [gpu for gpu, _ in Counter(FPSMeasurement.objects.values_list("gpu", flat=True)).most_common()]
    elif kind == "precision":
        measurements = FPSMeasurement.objects.filter(gpu=data.get(f"{axis}_gpu"))
        preferred = [precision for precision, _ in Counter(measurements.values_list("precision", flat=True)).most_common()]
    elif kind == "legend_attribute":
        preferred = ["family.name"]
    else:
        preferred = []
    return next((value for value in preferred if value in valid), valid[0])


def get_default_request(family=None, head=None, dataset=None, task_query=None, dataset_query=None):
    if family and task_query is not None:
        task = Task.objects.get(name=task_query)
        metric = TASK_TO_FIRST_METRIC[task.name]
        dataset_pk = get_family_task_dataset(family, task)
        return {
            "y_axis": "results",
            "y_task": task.pk,
            "y_dataset": dataset_pk,
            "y_metric": metric,
            "x_axis": "gflops",
            "legend_attribute": "pretrain_dataset.name",
            "legend_attribute_(second)": "pretrain_method",
        }
    elif head:
        if task_query is None:
            task_query = get_first_task(head)
        task = Task.objects.get(name=task_query)
        metric = TASK_TO_FIRST_METRIC[task.name]
        dataset_pk = get_head_task_dataset(head, task)
        return {
            "y_axis": "results",
            "y_task": task.pk,
            "y_dataset": dataset_pk,
            "y_head": head.pk,
            "y_metric": metric,
            "x_axis": "gflops",
            "legend_attribute": "family.name",
        }
    elif dataset:
        if task_query is None:
            task_query = get_first_task(dataset)
        task = Task.objects.get(name=task_query)
        metric = TASK_TO_FIRST_METRIC[task.name]
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


def get_first_task(model):
    tasks = model.tasks.all()
    task_counts = {}
    for task in tasks:
        if task.name not in TASK_TO_TABLE:
            continue
        result_model = TASK_TO_TABLE[task.name]
        related_name = result_model._meta.model_name + "_set"
        if result_model == InstanceResult:
            count = getattr(model, related_name).filter(instance_type__name=task.name).count()
        else:
            count = getattr(model, related_name).count()

        task_counts[task.name] = count
    return max(task_counts, key=task_counts.get)


def get_head_task_dataset(head, task):
    result_model = TASK_TO_TABLE[task.name]
    results = result_model.objects.filter(head=head)
    return get_most_common_dataset(results, task)


def get_family_task_dataset(family, task):
    result_model = TASK_TO_TABLE[task.name]
    results = result_model.objects.filter(pretrained_backbone__family=family)
    return get_most_common_dataset(results, task)


def get_most_common_dataset(results, task):
    if results.model == InstanceResult:
        results = results.filter(instance_type__name=task.name)
    # Keep first-seen tie breaking, without loading entire results and datasets.
    counts = Counter(results.values_list("dataset_id", flat=True))
    return max(counts, key=counts.get)

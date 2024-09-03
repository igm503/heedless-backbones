from django import forms
from django.shortcuts import render

from .plot import get_defaults, get_plot, PlotRequest, get_table_data, get_plot_data
from .constants import (
    AXIS_CHOICES,
    CLASSIFICATION_METRICS,
    INSTANCE_METRICS,
    FIELDS,
    RESOLUTIONS,
)
from .models import (
    Dataset,
    DownstreamHead,
    Task,
    TaskType,
    GPU,
    Precision,
)


last_plot, last_table, last_headers = get_defaults()


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
        self.init_filters()

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
            v for k, v in self.cleaned_data.items() if k not in ["_pretrain_dataset", "_resolution"]
        ]
        return self.init_graph or (None not in values and "" not in values)


def plot_view(request):
    global last_plot, last_table, last_headers
    plot = last_plot
    table_data = last_table
    table_headers = last_headers
    form = DatasetTaskForm(request.GET or None)
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        queryset = get_plot_data(plot_request)
        plot = get_plot(queryset, plot_request)
        table_data, table_headers = get_table_data(queryset, plot_request)
        print(table_data)
        last_plot = plot
    return render(
        request,
        "view/all.html",
        {
            "form": form,
            "plot": plot,
            "table_data": table_data,
            "table_headers": table_headers,
        },
    )


def show_family(request, family):
    pass


def show_dataset(request, dataset):
    pass


def show_downstream_head(request, downstream_head):
    pass

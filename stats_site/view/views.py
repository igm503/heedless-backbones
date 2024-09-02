from django import forms
from django.shortcuts import render

from .plot import get_default_plot, get_plot, PlotRequest
from .constants import AXIS_CHOICES, CLASSIFICATION_METRICS, INSTANCE_METRICS, FIELDS
from .models import (
    Dataset,
    DownstreamHead,
    Task,
    TaskType,
    GPU,
    Precision,
)


last_plot = get_default_plot()


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
        self.order_fields(field_order=sorted(field_names, key=lambda x: FIELDS.index(x)))

    def is_ready(self):
        values = list(self.cleaned_data.values())
        return self.init_graph or (None not in values and "" not in values)


def plot_view(request):
    global last_plot
    plot = last_plot
    form = DatasetTaskForm(request.GET or None)
    if form.is_valid() and form.is_ready():
        plot_request = PlotRequest(form.cleaned_data)
        plot = get_plot(plot_request)
        last_plot = plot
    return render(request, "view/all.html", {"form": form, "plot": plot})


# def get_single_plot(pretrained_backbones, x_type, y_type, dataset_name):
#     y_title = get_axis_title(y_type)
#     x_title = get_axis_title(x_type)
#     data = []
#     families = {pb.family.name for pb in pretrained_backbones}
#     marker_configs = get_marker_configs(families)
#     seen_families = set()
#
#     for pb in pretrained_backbones:
#         x_values = []
#         y_values = []
#         hovers = []
#         params = pb.backbone.m_parameters
#         for result in pb.filtered_results:
#             x = params if x_type == "m_parameters" else getattr(result, x_type)
#             y = params if y_type == "m_parameters" else getattr(result, y_type)
#             hover = (
#                 f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
#             )
#             x_values.append(x)
#             y_values.append(y)
#             hovers.append(hover)
#
#         show_legend = pb.family.name not in seen_families
#         seen_families.add(pb.family.name)
#
#         data.append(
#             go.Scatter(
#                 x=x_values,
#                 y=y_values,
#                 mode="markers",
#                 name=pb.family.name,
#                 text=hovers,
#                 hoverinfo="text",
#                 marker=marker_configs[pb.family.name],
#                 showlegend=show_legend,
#             )
#         )
#
#     title = f"{y_title} on {dataset_name}"
#     return get_plot_div(title, x_title, y_title, data)
#
#
# def get_multi_plot(pretrained_backbones, x_type, y_type, x_dataset, y_dataset):
#     y_title = get_axis_title(y_type)
#     x_title = get_axis_title(x_type)
#     data = []
#     families = {pb.family.name for pb in pretrained_backbones}
#     marker_configs = get_marker_configs(families)
#     seen_families = set()
#
#     for pb in pretrained_backbones:
#         x_values = []
#         y_values = []
#         hovers = []
#         for x_result in pb.x_results:
#             for y_result in pb.y_results:
#                 x = getattr(x_result, x_type)
#                 y = getattr(y_result, y_type)
#                 hover = f"<b>{pb.name}</b><br>Family: {pb.family.name}<br>{y_title}: {y:.2f}<br>{x_title}: {x:.2f}"
#                 x_values.append(x)
#                 y_values.append(y)
#                 hovers.append(hover)
#
#         show_legend = pb.family.name not in seen_families
#         seen_families.add(pb.family.name)
#
#         data.append(
#             go.Scatter(
#                 x=x_values,
#                 y=y_values,
#                 mode="markers",
#                 name=pb.family.name,
#                 text=hovers,
#                 hoverinfo="text",
#                 marker=marker_configs[pb.family.name],
#                 showlegend=show_legend,
#             )
#         )
#     title = f"{y_title} on {y_dataset.name} vs. {x_title} on {x_dataset.name}"
#     return get_plot_div(title, x_title, y_title, data)
#
#
# def get_all_plot(pretrained_backbones, x_type, y_type):
#     y_title = get_axis_title(y_type)
#     x_title = get_axis_title(x_type)
#     data = []
#     families = {pb.family.name for pb in pretrained_backbones}
#     marker_configs = get_marker_configs(families)
#     seen_families = set()
#
#     for pb in pretrained_backbones:
#         x_values = []
#         y_values = []
#         hovers = []
#         params = pb.backbone.m_parameters
#         for results in [pb._instance_results, pb._classification_results]:
#             for result in results:
#                 x = params if x_type == "m_parameters" else getattr(result, x_type)
#                 y = params if y_type == "m_parameters" else getattr(result, y_type)
#                 hover = f"Model: {pb.name}<br>Family: {pb.family.name}<br>{y_title}: {y}<br>{x_title}: {x}"
#                 x_values.append(x)
#                 y_values.append(y)
#                 hovers.append(hover)
#
#         show_legend = pb.family.name not in seen_families
#         seen_families.add(pb.family.name)
#
#         data.append(
#             go.Scatter(
#                 x=x_values,
#                 y=y_values,
#                 mode="markers",
#                 name=pb.family.name,
#                 text=hovers,
#                 hoverinfo="text",
#                 marker=marker_configs[pb.family.name],
#                 showlegend=show_legend,
#             )
#         )
#
# title = f"{y_title} against {x_title}"
# return get_plot_div(title, x_title, y_title, data)


def show_family(request, family):
    pass


def show_dataset(request, dataset):
    pass


def show_downstream_head(request, downstream_head):
    pass

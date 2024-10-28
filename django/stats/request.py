from dataclasses import dataclass

from .data_utils import get_axis_title
from .models import Dataset, DownstreamHead


class PlotRequest:
    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"

    @dataclass
    class DataArgs:
        dataset: Dataset | None = None
        task: str | None = None
        head: DownstreamHead | None = None
        resolution: int | None = None
        gpu: str | None = None
        precision: str | None = None
        fps: bool = False

        def __post_init__(self):
            self.fps = self.gpu is not None and self.precision is not None
            self.resolution = int(self.resolution) if self.resolution else None

    @dataclass
    class PlotArgs:
        x_attr: str
        y_attr: str
        x_task: str | None = None
        y_task: str | None = None
        dataset: Dataset | None = None
        x_dataset: Dataset | None = None
        y_dataset: Dataset | None = None
        group_by: str = "family.name"
        group_by_second: str | None = None
        x_title: str = ""
        y_title: str = ""

        def __post_init__(self):
            self.x_title = get_axis_title(self.x_attr, self.x_task, self.x_dataset or self.dataset)
            self.y_title = get_axis_title(self.y_attr, self.y_task, self.y_dataset or self.dataset)
            if self.group_by_second and not self.group_by:
                self.group_by = self.group_by_second
                self.group_by_second = None

    def __init__(self, args):
        self.pretrain_dataset = args.get("_pretrain_dataset")
        self.pretrain_method = args.get("_pretrain_method")

        x_args = PlotRequest.DataArgs(
            dataset=args.get("x_dataset"),
            task=args.get("x_task"),
            head=args.get("x_head"),
            resolution=args.get("x_resolution"),
            gpu=args.get("x_gpu"),
            precision=args.get("x_precision"),
        )
        y_args = PlotRequest.DataArgs(
            dataset=args.get("y_dataset"),
            task=args.get("y_task"),
            head=args.get("y_head"),
            resolution=args.get("y_resolution"),
            gpu=args.get("y_gpu"),
            precision=args.get("y_precision"),
        )
        self.data_args = PlotRequest.DataArgs(
            dataset=x_args.dataset or y_args.dataset,
            task=x_args.task or y_args.task,
            head=x_args.head or y_args.head,
            resolution=x_args.resolution or y_args.resolution,
            gpu=x_args.gpu or y_args.gpu,
            precision=x_args.precision or y_args.precision,
        )
        self.plot_args = PlotRequest.PlotArgs(
            x_attr=args.get("x_metric") or args.get("x_axis"),
            y_attr=args.get("y_metric") or args.get("y_axis"),
            dataset=y_args.dataset or x_args.dataset,
            x_dataset=x_args.dataset,
            y_dataset=y_args.dataset,
            x_task=x_args.task,
            y_task=y_args.task,
            group_by=args.get("legend_attribute"),
            group_by_second=args.get("legend_attribute_(second)"),
        )

        if x_args.dataset and y_args.dataset and x_args.task and y_args.task:
            if (
                x_args.dataset == y_args.dataset
                and x_args.task == y_args.task
                and x_args.head == y_args.head
            ):
                self.query_type = PlotRequest.SINGLE
            else:
                self.query_type = PlotRequest.MULTI
                self.x_args = x_args
                self.y_args = y_args
        else:
            if x_args.dataset or y_args.dataset:
                self.query_type = PlotRequest.SINGLE
            else:
                self.query_type = PlotRequest.NONE

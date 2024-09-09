AXIS_CHOICES = {
    "results": "Results",
    "m_parameters": "Parameters (M)",
    "fps": "Images / Second",
    "pub_date": "Publication Date",
}

AXIS_WITH_GFLOPS = {
    "results": "Results",
    "m_parameters": "Parameters (M)",
    "fps": "Images / Second",
    "gflops": "GFLOPs",
    "pub_date": "Publication Date",
}

CLASSIFICATION_METRICS = {
    "": "----------",
    "top_1": "Top-1",
    "top_5": "Top-5",
    "gflops": "GFLOPs",
}

INSTANCE_METRICS = {
    "": "----------",
    "mAP": "mAP",
    "AP50": "AP<sub>50</sub>",
    "AP75": "AP<sub>75</sub>",
    "mAPs": "mAP<sub>s</sub>",
    "mAPm": "mAP<sub>m</sub>",
    "mAPl": "mAP<sub>l</sub>",
    "gflops": "GFLOPs",
}

INSTANCE_SEG_METRICS = {
    "mAP": "mAP<sup>m</sup>",
    "AP50": "AP<span class='supsub'><sup>m</sup><sub>50</sub></span>",
    "AP75": "AP<span class='supsub'><sup>m</sup><sub>75</sub></span>",
    "mAPs": "mAP<span class='supsub'><sup>m</sup><sub>s</sub></span>",
    "mAPm": "mAP<span class='supsub'><sup>m</sup><sub>m</sub></span>",
    "mAPl": "mAP<span class='supsub'><sup>m</sup><sub>l</sub></span>",
    "gflops": "GFLOPs",
}

DETECTION_METRICS = {
    "mAP": "mAP<sup>b</sup>",
    "AP50": "AP<span class='supsub'><sup>b</sup><sub>50</sub></span>",
    "AP75": "AP<span class='supsub'><sup>b</sup><sub>75</sub></span>",
    "mAPs": "mAP<span class='supsub'><sup>b</sup><sub>s</sub></span>",
    "mAPm": "mAP<span class='supsub'><sup>b</sup><sub>m</sub></span>",
    "mAPl": "mAP<span class='supsub'><sup>b</sup><sub>l</sub></span>",
    "gflops": "GFLOPs",
}

RESOLUTIONS = {
    "": "----------",
    "224": "224x224",
    "384": "384x384",
}

LIMITED_LEGEND_ATTRIBUTES = {
    "": "----------",
    "family.name": "Family",
    "pretrain_dataset.name": "Pretrain Dataset",
    "pretrain_method": "Pretrain Method",
}

CLASSIFICATION_LEGEND_ATTRIBUTES = {
    "classification.resolution": "Classification Resolution",
}

INSTANCE_LEGEND_ATTRIBUTES = {
    "instance.head.name": "Downstream Head",
    "instance.train_epochs": "Downstream Training Epochs",
}

FIELDS = [
    "y_axis",
    "y_task",
    "y_dataset",
    "y_metric",
    "y_head",
    "y_resolution",
    "y_gpu",
    "y_precision",
    "x_axis",
    "x_task",
    "x_dataset",
    "x_metric",
    "x_head",
    "x_resolution",
    "x_gpu",
    "x_precision",
    "_pretrain_dataset",
    "_pretrain_method",
    "legend_attribute",
    "legend_attribute_(second)",
]

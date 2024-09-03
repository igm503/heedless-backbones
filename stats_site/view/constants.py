AXIS_CHOICES = {
    "results": "Results",
    "gflops": "GFLOPs",
    "m_parameters": "Parameters (M)",
    "fps": "Images / Second",
}

CLASSIFICATION_METRICS = {
    "": "----------",
    "top_1": "Top-1 Accuracy",
    "top_5": "Top-5 Accuracy",
}

INSTANCE_METRICS = {
    "": "----------",
    "mAP": "mAP",
    "AP50": "AP@50",
    "AP75": "AP@75",
    "mAPs": "mAP Small",
    "mAPm": "mAP Medium",
    "mAPl": "mAP Large",
}

RESOLUTIONS = {
   "": "----------",
   "224": "224x224",
   "384": "384x384",
}

FIELDS = [
    "y_axis",
    "y_task",
    "y_dataset",
    "y_metric",
    "y_head",
    "y_gpu",
    "y_precision",
    "x_axis",
    "x_task",
    "x_dataset",
    "x_metric",
    "x_head",
    "x_gpu",
    "x_precision",
    "_resolution",
    "_pretrain_dataset",
]

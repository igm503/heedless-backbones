from .models import TaskType, PretrainMethod
from .constants import (
    IMAGENET_C_METRICS,
    IMAGENET_C_BAR_METRICS,
    CLASSIFICATION_METRICS,
    INSTANCE_METRICS,
    SEMANTIC_SEG_METRICS,
    AXIS_CHOICES,
    INSTANCE_SEG_METRICS,
    DETECTION_METRICS,
)


def get_axis_title(db_name, instance_type=None, dataset=None):
    if db_name in INSTANCE_METRICS:
        if instance_type is None:
            title = INSTANCE_METRICS[db_name]
        elif instance_type == TaskType.DETECTION.value:
            title = DETECTION_METRICS[db_name]
        elif instance_type == TaskType.INSTANCE_SEG.value:
            title = INSTANCE_SEG_METRICS[db_name]
        else:
            title = db_name  # this shouldn't happen
    elif db_name in CLASSIFICATION_METRICS:
        title = CLASSIFICATION_METRICS[db_name]
        if dataset is not None:
            if dataset.name == "ImageNet-C":
                title = IMAGENET_C_METRICS[db_name]
            elif dataset.name == "ImageNet-C-bar":
                title = IMAGENET_C_BAR_METRICS[db_name]
    elif db_name in SEMANTIC_SEG_METRICS:
        title = SEMANTIC_SEG_METRICS[db_name]
    else:
        title = AXIS_CHOICES.get(db_name, "")
    return title


def get_value(pb, result, attr):
    if attr == "m_parameters":
        return pb.backbone.m_parameters
    elif attr == "pub_date":
        return pb.family.pub_date
    else:
        return getattr(result, attr)


def get_nested_attr(obj, attr_path):
    for attr in attr_path:
        obj = getattr(obj, attr)
    return str(obj)


def get_train_string(result):
    dataset = result.train_dataset.name if result.train_dataset else None
    training = f"{dataset} : {result.train_epochs}"
    if result.intermediate_train_dataset is not None:
        int_dataset = result.intermediate_train_dataset.name
        intermediate = f"{int_dataset} : {result.intermediate_train_epochs}"
        training = intermediate + " &rarr; " + training
    training = training.replace("None", "&mdash;")
    return training


def get_finetune_string(result):
    ft_dataset = result.fine_tune_dataset.name if result.fine_tune_dataset else None
    finetune = f"{ft_dataset} : {result.fine_tune_epochs} : {result.fine_tune_resolution}"
    if result.intermediate_fine_tune_dataset is not None:
        int_dataset = result.intermediate_fine_tune_dataset.name
        intermediate = f"{int_dataset} : {result.intermediate_fine_tune_epochs} : {result.intermediate_fine_tune_resolution}"
        finetune = intermediate + " &rarr; " + finetune
    finetune = finetune.replace("ImageNet", "IN")
    finetune = finetune.replace("None", "&mdash;")
    return finetune


def get_pretrain_string(pb):
    pt_dataset = pb.pretrain_dataset.name
    if "ImageNet" in pt_dataset:
        pt_dataset = pt_dataset.replace("ImageNet", "IN")
    pretrain_method = pb.pretrain_method
    if pretrain_method == PretrainMethod.SUPERVISED.value:
        pretrain_method = "Sup."
    return f"{pt_dataset} : {pretrain_method} : {pb.pretrain_epochs}"

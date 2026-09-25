"""Related records needed when rendering benchmark plots and tables."""

from .models import ClassificationResult, InstanceResult, PretrainedBackbone


def backbone_queryset():
    # Keep the parent query's joins unchanged; additional joins can reorder plot
    # points and table rows. These two batch lookups avoid per-backbone queries.
    return PretrainedBackbone.objects.select_related(
        "family", "backbone"
    ).prefetch_related("backbone__family", "pretrain_dataset")


def result_queryset(model):
    if model is ClassificationResult:
        related = ("fine_tune_dataset", "intermediate_fine_tune_dataset")
    else:
        related = ("head", "train_dataset", "intermediate_train_dataset")
        if model is InstanceResult:
            related += ("instance_type",)
    return model.objects.select_related("dataset", *related)

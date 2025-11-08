from enum import Enum

from django.db import models
from django.core.exceptions import ValidationError


class PretrainMethod(Enum):
    SUPERVISED = "Supervised"
    TOKEN_LABELLING = "Sup. + TL"
    FCMAE = "FCMAE"
    MAE = "MAE"
    CONTRASTIVE = "CL"
    MAP = "MAP"


class TokenMixer(Enum):
    ATTN = "Attention"
    CONV = "Convolution"
    LSTM = "LSTM"
    SSM = "State Space Model"
    ATTN_CONV = "Attn + Conv"
    CONV_SSM = "Conv + SSM"
    RANDOM = "Random"
    IDENTITY = "Identity"


class TaskType(Enum):
    CLASSIFICATION = "Classification"
    DETECTION = "Object Detection"
    INSTANCE_SEG = "Instance Segmentation"
    SEMANTIC_SEG = "Semantic Segmentation"
    PANOPTIC_SEG = "Panoptic Segmentation"


class GPU(Enum):
    V100 = "V100"
    A100 = "A100"
    H100 = "H100"
    T4 = "T4"
    RTX2080TI = "2080Ti"
    RTX3080 = "3080"
    RTX3090 = "3090"
    RTX4090 = "4090"
    L40S = "L40S"


class Precision(Enum):
    FP16 = "FP16"
    FP32 = "FP32"
    INT8 = "INT8"
    AMP = "AMP"
    TF32 = "TF32"
    BF16 = "BF16"


INSTANCE_TASKS = [TaskType.DETECTION.value, TaskType.INSTANCE_SEG.value]


class Task(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100, choices={name.value: name.value for name in TaskType})

    def __str__(self):
        return str(self.name)


class Dataset(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100)
    tasks = models.ManyToManyField(Task)
    eval = models.BooleanField()
    website = models.URLField()

    def __str__(self):
        return str(self.name)


class DownstreamHead(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100)
    tasks = models.ManyToManyField(Task)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        return str(self.name)


class FPSMeasurement(models.Model):
    objects = models.Manager()

    backbone_name = models.CharField(max_length=100)
    resolution = models.IntegerField()
    fps = models.FloatField()
    gpu = models.CharField(max_length=100, choices={name.value: name.value for name in GPU})
    precision = models.CharField(
        choices={name.value: name.value for name in Precision},
        max_length=20,
        null=True,
        blank=True,
    )
    batch_size = models.IntegerField(blank=True, null=True)
    source = models.URLField(blank=True)

    def __str__(self):
        string = ""
        for item in [
            self.backbone_name,
            self.gpu,
            self.resolution,
            self.precision,
        ]:
            if item is not None:
                string += str(item)
                string += "/"
        return string


class BackboneFamily(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100, unique=True)
    model_type = models.CharField(
        max_length=100, choices={name.value: name.value for name in TokenMixer}
    )
    hierarchical = models.BooleanField()
    pretrain_method = models.CharField(
        max_length=100, choices={name.value: name.value for name in PretrainMethod}
    )
    pub_date = models.DateField()
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def clean(self):
        if not (self.paper or self.github):
            raise ValidationError("Either paper or github must be provided")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.name)


class Backbone(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100, unique=True)
    family = models.ForeignKey(BackboneFamily, on_delete=models.CASCADE)
    m_parameters = models.FloatField()
    fps_measurements = models.ManyToManyField(FPSMeasurement, blank=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        return str(self.name)


class PretrainedBackbone(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100, unique=True)
    backbone = models.ForeignKey(Backbone, on_delete=models.CASCADE)
    family = models.ForeignKey(BackboneFamily, on_delete=models.CASCADE)
    pretrain_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        limit_choices_to={"tasks__name": TaskType.CLASSIFICATION.value},
    )
    pretrain_method = models.CharField(
        max_length=100, choices={name.value: name.value for name in PretrainMethod}
    )
    pretrain_resolution = models.IntegerField()
    pretrain_epochs = models.IntegerField()
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        return str(self.name)


class ClassificationResult(models.Model):
    objects = models.Manager()

    pretrained_backbone = models.ForeignKey(PretrainedBackbone, on_delete=models.CASCADE)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        limit_choices_to={"tasks__name": TaskType.CLASSIFICATION.value},
    )
    resolution = models.IntegerField()
    fine_tune_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        related_name="classification_fine_tune",
        limit_choices_to={"tasks__name": TaskType.CLASSIFICATION.value},
        blank=True,
        null=True,
    )
    fine_tune_epochs = models.IntegerField(blank=True, null=True)
    fine_tune_resolution = models.IntegerField(blank=True, null=True)
    intermediate_fine_tune_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        related_name="intermediate_classification_fine_tune",
        limit_choices_to={"tasks__name": TaskType.CLASSIFICATION.value},
        blank=True,
        null=True,
    )
    intermediate_fine_tune_epochs = models.IntegerField(blank=True, null=True)
    intermediate_fine_tune_resolution = models.IntegerField(blank=True, null=True)
    # MCE for Imagenet-C, CE for Imagenet-C-bar
    top_1 = models.FloatField()
    top_5 = models.FloatField(null=True, blank=True)
    gflops = models.FloatField(blank=True, null=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def clean(self):
        if self.fine_tune_dataset:
            if not self.fine_tune_resolution:
                raise ValidationError(
                    "fine tune resolution is required when fine tune dataset is provided"
                )
        else:
            if self.fine_tune_resolution:
                raise ValidationError(
                    "fine tune dataset is required when fine tune resolution is provided"
                )
            if self.fine_tune_epochs:
                raise ValidationError(
                    "fine tune dataset is required when fine tune epochs is provided"
                )
        if self.intermediate_fine_tune_dataset:
            if not self.intermediate_fine_tune_resolution:
                raise ValidationError(
                    "int fine tune resolution is required when int fine tune dataset is provided"
                )
        else:
            if self.intermediate_fine_tune_resolution:
                raise ValidationError(
                    "int fine tune dataset is required when int fine tune resolution is provided"
                )
            if self.intermediate_fine_tune_epochs:
                raise ValidationError(
                    "int fine tune dataset is required when int fine tune epochs is provided"
                )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        string = ""
        for item in [
            self.pretrained_backbone.name,
            self.dataset,
            self.fine_tune_dataset,
            self.fine_tune_resolution,
        ]:
            if item is not None:
                string += str(item)
                string += "/"
        return string


class InstanceResult(models.Model):
    objects = models.Manager()

    pretrained_backbone = models.ForeignKey(PretrainedBackbone, on_delete=models.CASCADE)
    head = models.ForeignKey(DownstreamHead, on_delete=models.RESTRICT)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        limit_choices_to={"tasks__name__in": INSTANCE_TASKS},
    )
    instance_type = models.ForeignKey(Task, on_delete=models.RESTRICT)
    train_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        related_name="instance_train",
        limit_choices_to={"tasks__name__in": INSTANCE_TASKS},
    )
    train_epochs = models.IntegerField()
    intermediate_train_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        related_name="intermediate_instance_train",
        limit_choices_to={"tasks__name__in": INSTANCE_TASKS},
        blank=True,
        null=True,
    )
    intermediate_train_epochs = models.IntegerField(blank=True, null=True)
    mAP = models.FloatField()
    AP50 = models.FloatField(null=True, blank=True)
    AP75 = models.FloatField(null=True, blank=True)
    mAPs = models.FloatField(null=True, blank=True)
    mAPm = models.FloatField(null=True, blank=True)
    mAPl = models.FloatField(null=True, blank=True)
    gflops = models.FloatField(blank=True, null=True)
    fps_measurements = models.ManyToManyField(FPSMeasurement, blank=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def clean(self):
        if (self.intermediate_train_epochs is None) != (self.intermediate_train_dataset is None):
            raise ValidationError(
                "int train dataset and int train epochs must be provided together"
            )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        string = ""
        for item in [
            self.pretrained_backbone.name,
            self.head,
            self.instance_type,
            self.dataset,
            self.train_dataset,
            self.train_epochs,
        ]:
            if item is not None:
                string += str(item)
                string += "/"
        return string


class SemanticSegmentationResult(models.Model):
    objects = models.Manager()
    pretrained_backbone = models.ForeignKey(PretrainedBackbone, on_delete=models.CASCADE)
    head = models.ForeignKey(DownstreamHead, on_delete=models.RESTRICT)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        limit_choices_to={"tasks__name": TaskType.SEMANTIC_SEG.value},
    )
    train_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        related_name="semantic_seg_train",
        limit_choices_to={"tasks__name": TaskType.SEMANTIC_SEG.value},
    )
    train_epochs = models.IntegerField()
    intermediate_train_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.RESTRICT,
        related_name="intermediate_semantic_seg_train",
        limit_choices_to={"tasks__name": TaskType.SEMANTIC_SEG.value},
        blank=True,
        null=True,
    )
    intermediate_train_epochs = models.IntegerField(blank=True, null=True)
    crop_size = models.IntegerField()

    ms_m_iou = models.FloatField(blank=True, null=True)
    ms_pixel_accuracy = models.FloatField(blank=True, null=True)
    ms_mean_accuracy = models.FloatField(blank=True, null=True)
    ss_m_iou = models.FloatField(blank=True, null=True)
    ss_pixel_accuracy = models.FloatField(blank=True, null=True)
    ss_mean_accuracy = models.FloatField(blank=True, null=True)
    flip_test = models.BooleanField(default=False, help_text="horizontal flip testing")
    gflops = models.FloatField(blank=True, null=True)
    fps_measurements = models.ManyToManyField("FPSMeasurement", blank=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def clean(self):
        if not self.ms_m_iou and not self.ss_m_iou:
            raise ValidationError("at least one of ms_m_iou and ss_m_iou must be provided")
        if (self.intermediate_train_epochs is None) != (self.intermediate_train_dataset is None):
            raise ValidationError(
                "int train dataset and int train epochs must be provided together"
            )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        string = ""
        for item in [
            self.pretrained_backbone.name,
            self.head,
            self.dataset,
            self.train_dataset,
            self.train_epochs,
        ]:
            if item is not None:
                string += str(item)
                string += "/"
        return string


TASK_TO_TABLE = {
    TaskType.CLASSIFICATION.value: ClassificationResult,
    TaskType.INSTANCE_SEG.value: InstanceResult,
    TaskType.DETECTION.value: InstanceResult,
    TaskType.SEMANTIC_SEG.value: SemanticSegmentationResult,
}

TASK_TO_FIRST_METRIC = {
    TaskType.CLASSIFICATION.value: "top_1",
    TaskType.INSTANCE_SEG.value: "mAP",
    TaskType.DETECTION.value: "mAP",
    TaskType.SEMANTIC_SEG.value: "ms_m_iou",
}

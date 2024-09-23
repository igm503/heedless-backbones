from enum import Enum

from django.db import models


class PretrainMethod(Enum):
    SUPERVISED = "Supervised"
    FCMAE = "FCMAE"
    MAE = "MAE"
    CONTRASTIVE = "CL"


class TokenMixer(Enum):
    ATTN = "Attention"
    CONV = "Convolution"
    LSTM = "LSTM"
    SSM = "State Space Model"
    ATTN_CONV = "Attn + Conv"
    CONV_SSM = "Conv + SSM"


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


class Precision(Enum):
    FP16 = "FP16"
    FP32 = "FP32"
    INT8 = "INT8"
    AMP = "AMP"
    TF32 = "TF32"
    BF16 = "BF16"


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
    batch_size = models.IntegerField("batch size", blank=True, null=True)
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


class InstanceResult(models.Model):
    objects = models.Manager()

    pretrained_backbone_name = models.CharField(max_length=100)
    head = models.ForeignKey(DownstreamHead, on_delete=models.CASCADE)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    instance_type = models.ForeignKey(Task, on_delete=models.CASCADE)
    train_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="instance_train",
    )
    train_epochs = models.IntegerField("training epochs", blank=True, null=True)
    intermediate_train_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="intermediate_instance_train",
        blank=True,
        null=True,
    )
    intermediate_train_epochs = models.IntegerField("intermediate training epochs", blank=True, null=True)
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

    def __str__(self):
        string = ""
        for item in [
            self.pretrained_backbone_name,
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


class ClassificationResult(models.Model):
    objects = models.Manager()

    pretrained_backbone_name = models.CharField(max_length=100)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    resolution = models.IntegerField()
    fine_tune_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="classification_fine_tune",
        blank=True,
        null=True,
    )
    fine_tune_epochs = models.IntegerField("fine-tuning epochs", blank=True, null=True)
    fine_tune_resolution = models.IntegerField("fine-tuning resolution", blank=True, null=True)
    intermediate_fine_tune_dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="intermediate_classification_fine_tune",
        blank=True,
        null=True,
    )
    intermediate_fine_tune_epochs = models.IntegerField(
        "intermediate fine-tuning epochs", blank=True, null=True
    )
    intermediate_fine_tune_resolution = models.IntegerField(
        "intermediate fine-tuning resolution", blank=True, null=True
    )
    # MCE for Imagenet-C, CE for Imagenet-C-bar
    top_1 = models.FloatField("top-1", null=True, blank=True)
    top_5 = models.FloatField("top-5", null=True, blank=True)
    gflops = models.FloatField(blank=True, null=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        string = ""
        for item in [
            self.pretrained_backbone_name,
            self.dataset,
            self.fine_tune_dataset,
            self.fine_tune_resolution,
        ]:
            if item is not None:
                string += str(item)
                string += "/"
        return string


class BackboneFamily(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100)
    model_type = models.CharField(
        "model type",
        max_length=100,
        choices={name.value: name.value for name in TokenMixer},
    )
    hierarchical = models.BooleanField()
    pretrain_method = models.CharField(
        "pretraining method",
        max_length=100,
        choices={name.value: name.value for name in PretrainMethod},
    )
    pub_date = models.DateField("publication date")
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        return str(self.name)


class Backbone(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100)
    family = models.ForeignKey(BackboneFamily, on_delete=models.CASCADE)
    m_parameters = models.FloatField("parameters (M)")
    fps_measurements = models.ManyToManyField(FPSMeasurement, blank=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        return str(self.name)


class PretrainedBackbone(models.Model):
    objects = models.Manager()

    name = models.CharField(max_length=100)
    backbone = models.ForeignKey(Backbone, on_delete=models.CASCADE)
    family = models.ForeignKey(BackboneFamily, on_delete=models.CASCADE)
    pretrain_dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    pretrain_method = models.CharField(
        "pretraining method",
        max_length=100,
        choices={name.value: name.value for name in PretrainMethod},
    )
    pretrain_resolution = models.IntegerField(blank=True, null=True)
    pretrain_epochs = models.IntegerField(blank=True, null=True)
    classification_results = models.ManyToManyField(ClassificationResult, blank=True)
    instance_results = models.ManyToManyField(InstanceResult, blank=True)
    paper = models.URLField(blank=True)
    github = models.URLField(blank=True)

    def __str__(self):
        return str(self.name)

import os
import yaml

from django.db import transaction
from django.core.management.base import BaseCommand
from django.core.exceptions import ValidationError

from .llm_gen import LLM_OUTPUT_DIR
from ...models import (
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    ClassificationResult,
    InstanceResult,
    Dataset,
    Task,
    DownstreamHead,
    FPSMeasurement,
)


class Command(BaseCommand):
    help = "Updates the database with information about a new AI model family"

    def add_arguments(self, parser):
        parser.add_argument(
            "yaml_file",
            type=str,
            help="name of the yaml file describing the new model family",
        )

    def handle(self, *args, **options):
        try:
            if ".yml" in options["yaml_file"]:
                file_path = os.path.join(LLM_OUTPUT_DIR, options["yaml_file"])
            else:
                file_path = os.path.join(LLM_OUTPUT_DIR, options["yaml_file"] + ".yml")
            parsed_data = load_yaml(file_path)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Validation error: {str(e)}"))
            return

        try:
            result = update_database(parsed_data)
            self.stdout.write(self.style.SUCCESS(result))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error updating database: {str(e)}"))


def load_yaml(path):
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML: {str(e)}")
    return data


@transaction.atomic
def update_database(data):
    family = BackboneFamily(
        name=data["name"],
        model_type=data["model_type"],
        hierarchical=data["hierarchical"],
        pretrain_method=data["pretrain_method"],
        pub_date=data["pub_date"],
        paper=data["paper"],
        github=data["github"],
    )
    family.full_clean()
    family.save()

    for backbone_data in data["backbones"]:
        backbone = Backbone(
            name=backbone_data["name"],
            family=family,
            m_parameters=backbone_data["m_parameters"],
        )
        backbone.full_clean()
        backbone.save()

        for fps_data in backbone_data.get("fps_measurements", []):
            fps_measurement = FPSMeasurement(
                backbone_name=backbone.name,
                resolution=fps_data["resolution"],
                gpu=fps_data["gpu"],
                precision=fps_data["precision"],
                fps=fps_data["fps"],
            )
            fps_measurement.full_clean()
            fps_measurement.save()
            backbone.fps_measurements.add(fps_measurement)
        for pretrained_data in backbone_data.get("pretrained_backbones", []):
            pretrained_backbone = PretrainedBackbone(
                name=pretrained_data["name"],
                backbone=backbone,
                family=family,
                pretrain_dataset=Dataset.objects.get(name=pretrained_data["pretrain_dataset"]),
                pretrain_method=pretrained_data["pretrain_method"],
                pretrain_resolution=pretrained_data["pretrain_resolution"],
                pretrain_epochs=pretrained_data["pretrain_epochs"],
            )
            pretrained_backbone.full_clean()
            pretrained_backbone.save()

            for classification_data in pretrained_data.get("classification_results", []):
                classification_result = ClassificationResult(
                    pretrained_backbone=pretrained_backbone,
                    dataset=Dataset.objects.get(name=classification_data["dataset"]),
                    resolution=classification_data["resolution"],
                    top_1=classification_data["top_1"],
                    top_5=classification_data.get("top_5"),
                    gflops=classification_data["gflops"],
                )
                if "fine_tune_dataset" in classification_data:
                    ft_dataset = Dataset.objects.get(name=classification_data["fine_tune_dataset"])
                    classification_result.fine_tune_dataset = ft_dataset
                    classification_result.fine_tune_epochs = classification_data["fine_tune_epochs"]
                    classification_result.fine_tune_resolution = classification_data[
                        "fine_tune_resolution"
                    ]
                if "intermediate_fine_tune_dataset" in classification_data:
                    inter_dataset = Dataset.objects.get(
                        name=classification_data["intermediate_fine_tune_dataset"]
                    )
                    classification_result.intermediate_fine_tune_dataset = inter_dataset
                    classification_result.intermediate_fine_tune_epochs = classification_data[
                        "intermediate_fine_tune_epochs"
                    ]
                    classification_result.intermediate_fine_tune_resolution = classification_data[
                        "intermediate_fine_tune_resolution"
                    ]
                classification_result.full_clean()
                classification_result.save()

            for instance_data in pretrained_data.get("instance_results", []):
                instance_result = InstanceResult(
                    pretrained_backbone=pretrained_backbone,
                    head=DownstreamHead.objects.get(name=instance_data["head"]),
                    dataset=Dataset.objects.get(name=instance_data["dataset"]),
                    instance_type=Task.objects.get(name=instance_data["instance_type"]),
                    train_dataset=Dataset.objects.get(name=instance_data["train_dataset"]),
                    train_epochs=instance_data["train_epochs"],
                    mAP=instance_data["mAP"],
                    AP50=instance_data.get("AP50"),
                    AP75=instance_data.get("AP75"),
                    mAPs=instance_data.get("mAPs"),
                    mAPm=instance_data.get("mAPm"),
                    mAPl=instance_data.get("mAPl"),
                    gflops=instance_data["gflops"],
                )
                if "intermediate_train_dataset" in instance_data:
                    instance_result.intermediate_train_dataset = Dataset.objects.get(
                        name=instance_data["intermediate_train_dataset"]
                    )
                    instance_result.intermediate_train_epochs = instance_data[
                        "intermediate_train_epochs"
                    ]
                instance_result.full_clean()
                instance_result.save()
                for fps_data in instance_data.get("fps_measurements", []):
                    fps_measurement = FPSMeasurement(
                        backbone_name=backbone.name,
                        resolution=fps_data["resolution"],
                        gpu=fps_data["gpu"],
                        precision=fps_data["precision"],
                        fps=fps_data["fps"],
                    )
                    fps_measurement.full_clean()
                    fps_measurement.save()
                    instance_result.fps_measurements.add(fps_measurement)
    return f"Success adding family {data['name']}"

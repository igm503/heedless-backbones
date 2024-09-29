import os
import yaml

from django.core.exceptions import ObjectDoesNotExist
from django.core.management.base import BaseCommand

from .llm_gen import YAML_DIR
from ...models import (
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    ClassificationResult,
    InstanceResult,
    FPSMeasurement,
)


class Command(BaseCommand):
    help = "Dumps model family information into a YAML file"

    def add_arguments(self, parser):
        parser.add_argument(
            "family",
            type=str,
            help="Name of the model family to dump",
        )

    def handle(self, *args, **options):
        try:
            family_data = generate_yaml_from_db(options["family"])
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error generating YAML: {str(e)}"))
            return

        try:
            yaml_name = f"{options['family']}.yml"
            yaml_path = os.path.join(YAML_DIR, yaml_name)
            with open(yaml_path, "w") as f:
                yaml.dump(family_data, f, default_flow_style=False, sort_keys=False)
            self.stdout.write(self.style.SUCCESS(f"YAML written to {yaml_path}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error writing to yaml file: {str(e)}"))
            return


def generate_yaml_from_db(family_name):
    family = BackboneFamily.objects.get(name=family_name)

    data = {
        "name": family.name,
        "model_type": family.model_type,
        "hierarchical": family.hierarchical,
        "pretrain_method": family.pretrain_method,
        "pub_date": str(family.pub_date),
        "paper": family.paper,
        "github": family.github,
        "backbones": [],
    }

    for backbone in Backbone.objects.filter(family=family):
        backbone_data = {
            "name": backbone.name,
            "m_parameters": backbone.m_parameters,
            "fps_measurements": [],
            "pretrained_backbones": [],
        }

        for fps in backbone.fps_measurements.all():
            backbone_data["fps_measurements"].append(
                {
                    "resolution": fps.resolution,
                    "gpu": fps.gpu,
                    "precision": fps.precision,
                    "fps": fps.fps,
                }
            )

        for pretrained in PretrainedBackbone.objects.filter(backbone=backbone):
            pretrained_data = {
                "name": pretrained.name,
                "pretrain_dataset": pretrained.pretrain_dataset.name,
                "pretrain_method": pretrained.pretrain_method,
                "pretrain_resolution": pretrained.pretrain_resolution,
                "pretrain_epochs": pretrained.pretrain_epochs,
                "classification_results": [],
                "instance_results": [],
            }

            for classification in ClassificationResult.objects.filter(
                pretrained_backbone=pretrained
            ):
                classification_data = {
                    "dataset": classification.dataset.name,
                    "resolution": classification.resolution,
                    "top_1": classification.top_1,
                    "top_5": classification.top_5,
                    "gflops": classification.gflops,
                }
                if classification.fine_tune_dataset:
                    classification_data.update(
                        {
                            "fine_tune_dataset": classification.fine_tune_dataset.name,
                            "fine_tune_epochs": classification.fine_tune_epochs,
                            "fine_tune_resolution": classification.fine_tune_resolution,
                        }
                    )
                if classification.intermediate_fine_tune_dataset:
                    classification_data.update(
                        {
                            "intermediate_fine_tune_dataset": classification.intermediate_fine_tune_dataset.name,
                            "intermediate_fine_tune_epochs": classification.intermediate_fine_tune_epochs,
                            "intermediate_fine_tune_resolution": classification.intermediate_fine_tune_resolution,
                        }
                    )
                pretrained_data["classification_results"].append(classification_data)

            for instance in InstanceResult.objects.filter(pretrained_backbone=pretrained):
                instance_data = {
                    "head": instance.head.name,
                    "dataset": instance.dataset.name,
                    "instance_type": instance.instance_type.name,
                    "train_dataset": instance.train_dataset.name,
                    "train_epochs": instance.train_epochs,
                    "mAP": instance.mAP,
                    "AP50": instance.AP50,
                    "AP75": instance.AP75,
                    "mAPs": instance.mAPs,
                    "mAPm": instance.mAPm,
                    "mAPl": instance.mAPl,
                    "gflops": instance.gflops,
                    "fps_measurements": [],
                }
                if instance.intermediate_train_dataset:
                    instance_data.update(
                        {
                            "intermediate_train_dataset": instance.intermediate_train_dataset.name,
                            "intermediate_train_epochs": instance.intermediate_train_epochs,
                        }
                    )
                for fps in instance.fps_measurements.all():
                    instance_data["fps_measurements"].append(
                        {
                            "resolution": fps.resolution,
                            "gpu": fps.gpu,
                            "precision": fps.precision,
                            "fps": fps.fps,
                        }
                    )
                pretrained_data["instance_results"].append(instance_data)

            backbone_data["pretrained_backbones"].append(pretrained_data)

        data["backbones"].append(backbone_data)

    return data

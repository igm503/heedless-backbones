import os
import yaml

from django.core.management.base import BaseCommand

from .llm_gen import YAML_DIR
from ...models import (
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    ClassificationResult,
    InstanceResult,
    SemanticSegmentationResult,
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
        if options["family"] == "all":
            for family in BackboneFamily.objects.all():
                self.dump_yaml(family.name)
        else:
            self.dump_yaml(options["family"])

    def dump_yaml(self, family_name):
        try:
            family_data = generate_yaml_from_db(family_name)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error generating YAML: {str(e)}"))
            return

        try:
            yaml_name = f"{family_name}.yml"
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
                "semantic_seg_results": [],    
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

            for semantic in SemanticSegmentationResult.objects.filter(
                pretrained_backbone=pretrained
            ):
                semantic_data = {
                    "head": semantic.head.name,
                    "dataset": semantic.dataset.name,
                    "train_dataset": semantic.train_dataset.name,
                    "train_epochs": semantic.train_epochs,
                    "train_resolution": semantic.train_resolution,
                    "ms_m_iou": semantic.ms_m_iou,
                    "ms_pixel_accuracy": semantic.ms_pixel_accuracy,
                    "ms_mean_accuracy": semantic.ms_mean_accuracy,
                    "ss_m_iou": semantic.ss_m_iou,
                    "ss_pixel_accuracy": semantic.ss_pixel_accuracy,
                    "ss_mean_accuracy": semantic.ss_mean_accuracy,
                    "flip_test": semantic.flip_test,
                    "gflops": semantic.gflops,
                    "fps_measurements": [],
                }
                if semantic.intermediate_train_dataset:
                    semantic_data.update(
                        {
                            "intermediate_train_dataset": semantic.intermediate_train_dataset.name,
                            "intermediate_train_epochs": semantic.intermediate_train_epochs,
                            "intermediate_train_resolution": semantic.intermediate_train_resolution,
                        }
                    )
                for fps in semantic.fps_measurements.all():
                    semantic_data["fps_measurements"].append(
                        {
                            "resolution": fps.resolution,
                            "gpu": fps.gpu,
                            "precision": fps.precision,
                            "fps": fps.fps,
                        }
                    )
                pretrained_data["semantic_seg_results"].append(semantic_data)

            backbone_data["pretrained_backbones"].append(pretrained_data)

        data["backbones"].append(backbone_data)

    return data

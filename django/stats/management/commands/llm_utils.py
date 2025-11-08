import requests
import yaml
import io
import os
import re

from django.db.models.fields.related import ForeignKey, ManyToManyField
from django.core.exceptions import ValidationError
from dotenv import load_dotenv
from pypdf import PdfReader

from ...models import (
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    ClassificationResult,
    InstanceResult,
    SemanticSegmentationResult,
    FPSMeasurement,
    Dataset,
    Task,
    DownstreamHead,
    GPU,
    TokenMixer,
    Precision,
    PretrainMethod,
)

LLM_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        os.path.pardir,
        "llm_output",
    )
)
if not os.path.exists(LLM_OUTPUT_DIR):
    os.makedirs(LLM_OUTPUT_DIR)

YAML_DIR = os.path.abspath(os.path.join(LLM_OUTPUT_DIR, os.path.pardir, "family_data"))

PROMPT = """
Please analyze the provided paper and generate a YAML 
structure describing the new model family, its backbones, 
and its results, and related fps measurements. Since this
YAML will be used to update a database, you should be
aware of the following information:

Table definitions:

{model_definitions}

Available Datasets, Tasks, Downstream Heads, and GPUs:

{static_data}

However, please ignore Panoptic Segmentation.

The YAML you will generate will be used to populate to 
create new entries in the tables listed above. An example
of the desired YAML structure to generate the tables is
given below. 

A few notes:

- Foreign key and many to many relationships are indicated by nesting.
- Classification results and instance results (this means instance
  segmentation or object detection) can take more than just one metric. 
  For example, instance results  can take mAP, AP50, AP75, mAPs, mAPm, 
  mAPl, and gflops.
- If some of these metrics are not included for a given experiment, you 
  may indicate this by using the keyword "null" in the YAML. 
- Please include all metrics that are listed.
- Please include relevant gflop measurements along with each result.
- Be careful when reporting train epochs for semantic segmentation results.
  Often the training settings report iterations instead of epochs. Make sure
  to convert the iterations to epochs if this is the case.

The following example shows the desired YAML structure:

{example_yaml}

YAML Structure:

model_family [object]
├─ name [string, required]
├─ model_type [string, required]
├─ hierarchical [boolean, required]
├─ pretrain_method [string, required]
├─ pub_date [date, required]
├─ paper [url, required]
├─ github [url, optional]
└─ backbones [list, required, min 1]
   ├─ name [string, required]
   ├─ m_parameters [float, required]
   ├─ fps_measurements [list, optional]
   │  ├─ resolution [int, required]
   │  ├─ fps [float, required]
   │  ├─ gpu [string, required]
   │  └─ precision [string, required]
   └─ pretrained_backbones [list, required, min 1]
      ├─ name [string, required]
      ├─ pretrain_dataset [string, required]
      ├─ pretrain_method [string, required]
      ├─ pretrain_resolution [int, required]
      ├─ pretrain_epochs [int, required]
      ├─ classification_results [list, optional]
      │  ├─ dataset [string, required]
      │  ├─ resolution [int, required]
      │  ├─ top_1 [float, required]
      │  ├─ top_5 [float, optional]
      │  ├─ gflops [float, required]
      │  ├─ fine_tune_dataset [string, conditional]
      │  ├─ fine_tune_epochs [int, conditional]
      │  ├─ fine_tune_resolution [int, conditional]
      │  ├─ intermediate_fine_tune_dataset [string, optional]
      │  ├─ intermediate_fine_tune_epochs [int, optional]
      │  └─ intermediate_fine_tune_resolution [int, optional]
      ├─ instance_results [list, optional]
      │  ├─ head [string, required]
      │  ├─ dataset [string, required]
      │  ├─ instance_type [enum: "Object Detection"|"Instance Segmentation", required]
      │  ├─ train_dataset [string, required]
      │  ├─ train_epochs [int, required]
      │  ├─ mAP [float, required]
      │  ├─ AP50 [float, optional]
      │  ├─ AP75 [float, optional]
      │  ├─ mAPs [float, optional]
      │  ├─ mAPm [float, optional]
      │  ├─ mAPl [float, optional]
      │  ├─ gflops [float, required]
      │  ├─ intermediate_train_dataset [string, optional]
      │  ├─ intermediate_train_epochs [int, optional]
      │  └─ fps_measurements [list, optional]
      │     └─ [same structure as backbone fps_measurements]
      └─ semantic_seg_results [list, optional]
         ├─ head [string, required]
         ├─ dataset [string, required]
         ├─ train_dataset [string, required]
         ├─ train_epochs [int, required]
         ├─ crop_size [int, required]
         ├─ ms_m_iou [float, required if ss_m_iou absent]
         ├─ ms_pixel_accuracy [float, optional]
         ├─ ms_mean_accuracy [float, optional]
         ├─ ss_m_iou [float, required if ms_m_iou absent]
         ├─ ss_pixel_accuracy [float, optional]
         ├─ ss_mean_accuracy [float, optional]
         ├─ flip_test [boolean, required]
         ├─ gflops [float, required]
         └─ fps_measurements [list, optional]
            └─ [same structure as backbone fps_measurements]

Field Semantics and Rules:

model_type: Describes the channel mixing approach used by the model architecture

backbones: Each backbone represents a different size/scale variant of the model family introduced in the paper. You must include every backbone size mentioned in the paper. The backbone name generally follow the convention FamilyName-SizeAbbrev such as ConvNeXt V2-T or ConvNeXt V2-B.

m_parameters: The number of parameters in millions for the backbone model.

pretrained_backbones: Different training runs of the same backbone architecture. Include every distinct pretraining configuration reported in the paper. A pretrained backbone represents a specific instance of a backbone that has been trained with particular hyperparameters and datasets.

pretrain_resolution: The resolution in pixels at which the backbone was pretrained. This is typically a square resolution, so 224 means 224x224 pixels.

pretrain_epochs: The total number of training epochs used during pretraining.

classification_results - Scope: For each pretrained backbone, include every image classification result reported in the paper. This includes results on different datasets, at different resolutions, and with different training configurations.

classification_results - Fine-tuning conditions: A result requires fine-tuning fields when any of these conditions hold:
- The pretraining dataset differs from the evaluation dataset (e.g., pretrained on ImageNet-22k, evaluated on ImageNet-1k)
- The pretraining resolution differs from the evaluation resolution (e.g., pretrained at 224px, evaluated at 384px)
- The pretraining method differs from the training method used for classification (e.g., pretrained with FCMAE, fine-tuned with supervised learning)

When fine-tuning occurs, populate fine_tune_dataset, fine_tune_epochs, and fine_tune_resolution.

classification_results - Intermediate fine-tuning: Some papers report a three-stage training process: pretraining then intermediate fine-tuning then final fine-tuning. For example, a model might be pretrained with FCMAE on ImageNet-22k at 224px, then supervised fine-tuned on ImageNet-22k at 224px (intermediate), then finally supervised fine-tuned on ImageNet-1k at 224px (final). When this occurs, populate the intermediate_fine_tune_dataset, intermediate_fine_tune_epochs, and intermediate_fine_tune_resolution fields with the intermediate stage's parameters.

gflops: The computational cost in giga floating-point operations for a forward pass through the model at the specified resolution. For classification results, this is just the backbone. For instance and semantic segmentation results, this includes both the backbone and the downstream head.

instance_results - Scope: For each pretrained backbone, include every object detection and instance segmentation result reported in the paper.

instance_results - Instance type: Must be either Object Detection or Instance Segmentation. These are different tasks that may use the same head architecture but produce different outputs.

instance_results - Head: The name of the detection/segmentation head used, such as Mask R-CNN, Cascade R-CNN, RetinaNet, etc.

instance_results - Training datasets: The train_dataset field specifies the dataset used for training the full model (backbone plus head). This is typically the same as the evaluation dataset but may differ in some cases.

instance_results - Intermediate training: Some papers train object detectors in two stages: first training the backbone plus head on a large dataset like Objects365, then training on the target dataset like COCO. When this occurs, populate intermediate_train_dataset and intermediate_train_epochs.

instance_results - AP metrics: mAP is always required. AP50, AP75, mAPs (small objects), mAPm (medium objects), and mAPl (large objects) are optional but should be included when reported in the paper.

semantic_seg_results - Scope: For each pretrained backbone, include every semantic segmentation result reported in the paper.

semantic_seg_results - Head: The name of the segmentation head used, such as UPerNet, Panoptic FPN, etc. NOTE: Semantic FPN should be replaced with Panoptic FPN.

semantic_seg_results - Crop size: The size of image crops used during training. Semantic segmentation models are typically trained on crops rather than full images.

semantic_seg_results - Multi-scale vs single-scale: Papers report either multi-scale evaluation (ms_m_iou) or single-scale evaluation (ss_m_iou), or sometimes both. At least one must be present. Multi-scale evaluation typically yields higher scores as it averages predictions across multiple image scales. The ms_ prefix indicates multi-scale metrics, the ss_ prefix indicates single-scale metrics.

semantic_seg_results - Flip test: A boolean indicating whether the model was evaluated with horizontal flip augmentation (averaging predictions from the original and horizontally flipped images).

fps_measurements - Context: FPS measurements can appear at three levels: on the backbone itself, on instance results (backbone plus detection/segmentation head), and on semantic segmentation results (backbone plus segmentation head). Include these when reported in the paper, provided all required information is available.

fps_measurements - GPU: The specific GPU model used for the measurement, such as A100, V100, RTX 3090, etc.

fps_measurements - Precision: The numerical precision used, such as FP32, FP16, INT8, etc.

The following is the paper content:

{pdf_content}

Please enclose your yaml in tags as follows:
```yaml 
# yaml content 
```
"""


def get_prompt_with_example(pdf_content, example_name):
    return PROMPT.format(
        pdf_content=pdf_content,
        static_data=get_static_data(),
        model_definitions=get_model_definitions(),
        example_yaml=get_example_yaml(example_name),
    )


def get_static_data():
    return {
        "Model Types": [mixer.value for mixer in TokenMixer],
        "Pretrain Methods": [method.value for method in PretrainMethod],
        "Datasets": [dataset.name for dataset in Dataset.objects.all()],
        "Tasks": [task.name for task in Task.objects.all()],
        "Downstream Heads": [head.name for head in DownstreamHead.objects.all()],
        "GPUs": [gpu.value for gpu in GPU],
        "GPU Precisions": [precision.value for precision in Precision],
    }




def get_model_definitions():
    model_definitions = {}

    for model in [
        BackboneFamily,
        Backbone,
        PretrainedBackbone,
        ClassificationResult,
        SemanticSegmentationResult,
        InstanceResult,
        FPSMeasurement,
    ]:
        fields = []
        for field in model._meta.get_fields():
            if isinstance(field, ForeignKey):
                fields.append(f"{field.name} (FK to {field.related_model.__name__})")
            elif isinstance(field, ManyToManyField):
                fields.append(f"{field.name} (M2M to {field.related_model.__name__})")
            else:
                fields.append(field.name)

        model_definitions[model.__name__] = ", ".join(fields)

    return "\n".join([f"{model}: {fields}" for model, fields in model_definitions.items()])


def get_example_yamls(k, family_name):
    example_yamls = ""
    i = 1
    for file in os.listdir(YAML_DIR):
        if ".yml" not in file:
            continue
        name = file.split(".")[0]
        if "_" in name or name == family_name:
            continue
        example_yamls += get_example_yaml(file)
        i += 1
        if i > k:
            break

    return example_yamls


def get_example_yaml(filename):
    path = os.path.join(YAML_DIR, filename)
    example_yaml = ""
    with open(path, "r") as f:
        example_yaml += "Example: \n\n```yaml\n"
        example_yaml += f.read()
        example_yaml += "```\n\n"
    return example_yaml


def get_pdf_content(url):
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    content = "\n".join(page.extract_text() for page in pdf_reader.pages)

    content = re.sub(r"/uni\d+/uni\d+[^\s]*", "", content)

    return content.strip()


def call_openai_api(prompt):
    import openai

    load_dotenv()
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    message = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.choices[0].message.content


def call_anthropic_api(prompt):
    import anthropic

    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_together_api(prompt):
    from together import Together

    load_dotenv()
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def parse_model_output(model_output):
    try:
        assert "```yaml" in model_output
        yaml_text = model_output.split("```yaml")[1].split("```")[0]
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML: {str(e)}")
    return data

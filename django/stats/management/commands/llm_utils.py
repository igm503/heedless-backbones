import requests
import yaml
import io
import os

from django.db.models.fields.related import ForeignKey, ManyToManyField
from dotenv import load_dotenv
from pypdf import PdfReader
import anthropic
import openai

from ...models import (
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    ClassificationResult,
    InstanceResult,
    FPSMeasurement,
    Dataset,
    Task,
    DownstreamHead,
    GPU,
    TokenMixer,
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

However, please ignore Panoptic and Semantic Segmentation.

The YAML you will generate will be used to populate to 
create new entries in the tables listed above. An example
of the desired YAML structure to generate the tables is
given below. Note that foreign key and many to many
relationships are indicated by nesting. Note also that
classification results and instance (this means instance
segmentation or object detection) results can take
more than just one metric. For example, instance results 
can take mAP, AP50, AP75, mAPs, mAPm, mAPl, and gflops.
If some of these metrics are not included for a given
experiment, you are welcome to leave the keys out of the 
YAML. However, please include all metrics that are listed.
Finally, please include relevant gflop measurements along
with each result.

The following examples show the desired YAML structure:

{example_yamls}


- Include every backbone from the model family introduced in the paper. 
- For each bacbkbone, include every pretrained backbone introduced in the 
paper.
- Include every image classification result for the pretrained backbones 
- Include every object detection and instance segmentation result for 
the pretrained backbones, provided the downstream head used in the task
exists in the list given above. 
- Sometimes not all of the pretrained backbones will have instance results,
or different sizes will use different heads and training schemes. Keep this
in mind, and include all and only those instance results found in the paper.
- Include every FPS result available for the models, including for 
instance tasks (instance segmentation and object detection). 
- FPS measurements are not always available, so the fps fields are optional. Only
include measurements if they are available, and if you can find the gpu + float 
precision used to take the measurement.

The following is the paper content:

{pdf_content}

Please enclose your yaml in tags as follows:
```yaml 
# yaml content 
```
"""


def get_prompt_with_examples(pdf_content, num_examples, exclude_name=None):
    return PROMPT.format(
        pdf_content=pdf_content,
        static_data=get_static_data(),
        model_definitions=get_model_definitions(),
        example_yamls=get_example_yamls(num_examples, exclude_name),
    )


def get_static_data():
    return {
        "Model Types": [mixer.name for mixer in TokenMixer],
        "Datasets": [dataset.name for dataset in Dataset.objects.all()],
        "Tasks": [task.name for task in Task.objects.all()],
        "Downstream Heads": [head.name for head in DownstreamHead.objects.all()],
        "GPUs": [gpu.value for gpu in GPU],
    }


def get_model_definitions():
    model_definitions = {}

    for model in [
        BackboneFamily,
        Backbone,
        PretrainedBackbone,
        ClassificationResult,
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
        yaml_path = os.path.join(YAML_DIR, file)
        with open(yaml_path, "r") as f:
            example_yamls += f"Example {i}:\n\n```yaml\n"
            example_yamls += f.read()
            example_yamls += "```\n\n"
            i += 1
        if i > k:
            break

    return example_yamls


def get_pdf_content(url):
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    content = "\n".join(page.extract_text() for page in pdf_reader.pages)
    return content




def call_openai_api(prompt):
    load_dotenv()
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    message = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.choices[0].message.content


def call_anthropic_api(prompt):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def parse_model_output(model_output):
    try:
        assert "```yaml" in model_output
        yaml_text = model_output.split("```yaml")[1].split("```")[0]
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML: {str(e)}")
    return data

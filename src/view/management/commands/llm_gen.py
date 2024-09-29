import requests
import os
import yaml
import io

from dotenv import load_dotenv
from pypdf import PdfReader
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand
from django.db.models.fields.related import ForeignKey, ManyToManyField
import anthropic

from .prompt import PROMPT
from ...models import (
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    ClassificationResult,
    InstanceResult,
    Dataset,
    Task,
    GPU,
    DownstreamHead,
    FPSMeasurement,
    TokenMixer,
)

LLM_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir, "llm_entries"
)


STATIC_DATA = {
    "Model Types": [mixer.name for mixer in TokenMixer],
    "Datasets": [dataset.name for dataset in Dataset.objects.all()],
    "Tasks": [task.name for task in Task.objects.all()],
    "Downstream Heads": [head.name for head in DownstreamHead.objects.all()],
    "GPUs": [gpu.value for gpu in GPU],
}


class Command(BaseCommand):
    help = "Generates a YAML file with information about a new AI model family using Claude 3.5"

    def add_arguments(self, parser):
        parser.add_argument(
            "paper_url",
            type=str,
            help="URL of the paper describing the new model family",
        )
        parser.add_argument("name", type=str, help="Name of the new model family")

    def handle(self, *args, **options):
        try:
            pdf_content = get_pdf_content(options["paper_url"])
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error downloading PDF: {str(e)}"))
            return

        try:
            model_definitions = get_model_definitions()
            prompt = PROMPT.format(
                static_data=STATIC_DATA,
                pdf_content=pdf_content,
                model_definitions=model_definitions,
            )
            llm_output = call_anthropic_api(prompt)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error calling Anthropic API: {str(e)}"))
            return

        try:
            os.makedirs(LLM_OUTPUT_DIR, exist_ok=True)
            file_name = get_llm_output_filename(options["name"])
            file_path = os.path.join(LLM_OUTPUT_DIR, file_name)
            with open(file_path, "w") as f:
                f.write(llm_output)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error writing to text file: {str(e)}"))
            return

        try:
            parsed_data = parse_model_output(llm_output)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Validation error: {str(e)}"))
            return

        try:
            yaml_path = file_path.replace(".txt", ".yml")
            with open(yaml_path, "w") as f:
                yaml.dump(parsed_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error writing to yaml file: {str(e)}"))
            return


def get_pdf_content(url):
    response = requests.get(url)
    response.raise_for_status()
    pdf_file = io.BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    content = "\n".join(page.extract_text() for page in pdf_reader.pages)
    return content


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


def call_anthropic_api(prompt):
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def get_llm_output_filename(name):
    existing_files = os.listdir(LLM_OUTPUT_DIR)
    return f"{name}_{len(existing_files)}.txt"


def parse_model_output(model_output):
    try:
        assert "```yaml" in model_output
        yaml_text = model_output.split("```yaml")[1].split("```")[0]
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML: {str(e)}")
    return data

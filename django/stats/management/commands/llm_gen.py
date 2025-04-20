import os
import yaml

from django.core.management.base import BaseCommand

from .llm_utils import (
    LLM_OUTPUT_DIR,
    YAML_DIR,
    get_pdf_content,
    get_prompt_with_examples,
    call_anthropic_api,
    call_openai_api,
    call_together_api,
    parse_model_output,
)


class Command(BaseCommand):
    help = "Generates a YAML file with information about a new AI model family using Claude 3.7"

    def add_arguments(self, parser):
        parser.add_argument(
            "paper_url",
            type=str,
            help="URL of the paper describing the new model family",
        )
        parser.add_argument("name", type=str, help="Name of the new model family")

    def handle(self, *args, **options):
        assert f"{options['name']}.yml" not in os.listdir(
            YAML_DIR
        ), "Model family yaml already exists"
        assert f"{options['name']}.txt" not in os.listdir(
            LLM_OUTPUT_DIR
        ), "Model family txt already exists"
        try:
            pdf_content = get_pdf_content(options["paper_url"])
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error downloading PDF: {str(e)}"))
            return

        try:
            prompt = get_prompt_with_examples(pdf_content, 3)
            print(prompt)
            llm_output = call_anthropic_api(prompt)
            # Deepseek R1
            # llm_output = call_together_api(prompt)
            # GPT-4o
            # llm_output = call_openai_api(prompt)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error calling LLM API: {str(e)}"))
            return

        try:
            file_name = f"{options['name']}.txt"
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
            yaml_name = file_name.replace(".txt", ".yml")
            yaml_path = os.path.join(YAML_DIR, yaml_name)
            with open(yaml_path, "w") as f:
                yaml.dump(parsed_data, f, default_flow_style=False, sort_keys=False)
            self.stdout.write(self.style.SUCCESS(f"YAML written to {yaml_path}"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error writing to yaml file: {str(e)}"))
            return

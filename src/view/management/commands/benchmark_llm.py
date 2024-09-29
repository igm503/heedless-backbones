import difflib
import os
import yaml
from collections.abc import MutableMapping, MutableSequence

from django.core.management.base import BaseCommand

from .llm_utils import (
    YAML_DIR,
    get_pdf_content,
    get_prompt_with_examples,
    call_anthropic_api,
    parse_model_output,
)
from ...models import BackboneFamily


class Command(BaseCommand):
    help = "Benchmarks data entry performance of Claude 3.5 tool"

    def add_arguments(self, parser):
        parser.add_argument(
            "paper_url",
            type=str,
            help="URL of the paper to benchmark on",
        )
        parser.add_argument("name", type=str, help="Name of the new model family")

    def handle(self, *args, **options):
        family = BackboneFamily.objects.get(name=options["name"])

        try:
            pdf_content = get_pdf_content(options["paper_url"])
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error downloading PDF: {str(e)}"))
            return

        try:
            prompt = get_prompt_with_examples(pdf_content, 5, family.name)
            llm_output = call_anthropic_api(prompt)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error calling Anthropic API: {str(e)}"))
            return

        try:
            generated = parse_model_output(llm_output)
            # with open("generated.txt", "r") as file:
            #     generated = yaml.safe_load(file)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Validation error: {str(e)}"))
            return

        yaml_name = options["name"] + ".yml"
        reference_path = os.path.join(YAML_DIR, yaml_name)
        with open(reference_path, "r") as file:
            reference = yaml.safe_load(file)
        results = compare_yamls(generated, reference)
        self.stdout.write(self.style.SUCCESS(f"Results: \n {results}"))


def sort_nested_structures(item):
    if isinstance(item, MutableMapping):
        return {k: sort_nested_structures(v) for k, v in sorted(item.items())}
    elif isinstance(item, MutableSequence):
        return sorted([sort_nested_structures(i) for i in item], key=dict_sort)
    else:
        return item


def dict_sort(item):
    if "name" in item:
        return item["name"]
    elif "gpu" in item:
        return str(item["gpu"]) + str(item["precision"]) + str(item["resolution"])
    elif "mAP" in item:
        return str(item["dataset"]) + str(item["head"]) + str(item["mAP"])
    else:
        return (
            item["dataset"]
            + str(item["resolution"])
            + str(item.get("fine_tune_dataset"))
            + str(item["top_1"])
        )


def compare_yamls(generated_yaml, reference_yaml):
    sorted_generated_yaml = sort_nested_structures(generated_yaml)
    sorted_reference_yaml = sort_nested_structures(reference_yaml)

    generated_yaml_str = yaml.dump(sorted_generated_yaml, default_flow_style=False, sort_keys=False)
    reference_yaml_str = yaml.dump(sorted_reference_yaml, default_flow_style=False, sort_keys=False)

    generated_lines = generated_yaml_str.splitlines(keepends=True)
    reference_lines = reference_yaml_str.splitlines(keepends=True)

    with open("generated.txt", "w") as f:
        f.writelines(generated_lines)

    with open("reference.txt", "w") as f:
        f.writelines(reference_lines)

    differ = difflib.Differ()
    diff = list(differ.compare(reference_lines, generated_lines))

    different_lines = [line for line in diff if line.startswith("+ ") or line.startswith("- ")]
    num_different_lines = len(different_lines)

    total_lines = max(len(reference_lines), len(generated_lines))
    percent_different = (num_different_lines / total_lines) * 100

    with open("diff.txt", "w") as f:
        f.writelines(different_lines)

    return {
        "num_different_lines": num_different_lines,
        "total_lines": total_lines,
        "percent_different": percent_different,
    }

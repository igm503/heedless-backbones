import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.apps import apps
from django.test import TestCase

from ingestion.evaluate import (
    DatabaseWrite, Evaluation, eval_papers, hidden_vocabulary, no_database_writes, score_eval,
)
from ingestion import agent
from ingestion.family_yaml import family_to_dict, yaml_to_entries
from ingestion.tests.test_clients import selection
from stats.models import BackboneFamily, ClassificationResult, InstanceResult


def extraction_of(family):
    """What a perfect extraction of the family would return."""
    records = []
    for key, kind, data in yaml_to_entries(family_to_dict(family)):
        data = {name: value for name, value in data.items() if kind == "family" or name not in {"paper", "github"}}
        records.append({"key": key, "kind": kind, "fields": [{"name": n, "value": v} for n, v in data.items()],
                        "evidence": [], "uncertain": []})
    return {"decision": dict(selection(), qualifies=True), "records": records}


class EvaluateTests(TestCase):
    fixtures = [str(Path(__file__).resolve().parents[3] / "db.json")]

    def setUp(self):
        self.family = BackboneFamily.objects.get(name="Swin")
        self.directory = Path(tempfile.mkdtemp())

    def evaluate(self, output):
        """Run the eval with the agent session replaced by one that writes `output`."""
        prepared = {}

        def prepare(folder, identifier, families_hidden=(), read_only=False):
            prepared.update(hidden=list(families_hidden), read_only=read_only)
            agent.write_json(folder / "pages.json", ["page text"])
            agent.write_json(folder / "metadata.json", {"arxiv_id": identifier, "version": 2})
            (folder / "paper.pdf").write_bytes(b"%PDF-test")
            (folder / "sources").mkdir(exist_ok=True)
            return folder

        def run_agent(folder, model=None, timeout=3600):
            agent.write_json(folder / agent.FINAL, output)
            return {"seconds": 12.0, "turns": 30, "is_error": False, "summary": "done", "cost_usd": 1.25,
                    "usage": {"input_tokens": 1000, "cache_read_input_tokens": 50000, "output_tokens": 20000},
                    "auth": "none"}

        counts = {model: model.objects.count() for model in apps.get_models()}
        with no_database_writes(), patch("ingestion.agent.prepare", side_effect=prepare), \
                patch("ingestion.agent.run_agent", side_effect=run_agent):
            Evaluation(self.directory).run({"2103.14030": [self.family]})
            summary = score_eval(self.directory)
        self.assertEqual({model: model.objects.count() for model in apps.get_models()}, counts)
        self.assertEqual(prepared, {"hidden": ["Swin"], "read_only": True})
        return summary

    def test_eval_writes_organized_files_and_nothing_to_the_database(self):
        summary = self.evaluate(extraction_of(self.family))
        folder = self.directory / "papers" / "2103.14030"
        for name in ["paper.pdf", "pages.json", "metadata.json", "agent.json", "decision.json", "records.json",
                     "reference.yml", "score.json"]:
            self.assertTrue((folder / name).exists(), name)
        for name in ["config.json", "summary.json", "summary.md"]:
            self.assertTrue((self.directory / name).exists(), name)
        overall = summary["overall"]
        self.assertEqual((overall["result_recall"], overall["result_precision"]), (1, 1))
        self.assertEqual((overall["metric_accuracy"], overall["field_accuracy"]), (1, 1))
        self.assertEqual(summary["cost_usd"], 1.25)
        self.assertEqual((summary["input_tokens"], summary["output_tokens"]), (51000, 20000))
        self.assertNotIn("Warning", (self.directory / "summary.md").read_text())

    def test_wrong_and_missing_values_are_reported(self):
        output = extraction_of(self.family)
        results = [record for record in output["records"] if record["kind"] == "classification"]
        next(f for f in results[0]["fields"] if f["name"] == "top_1")["value"] += 1
        output["records"].remove(results[1])
        summary = self.evaluate(output)
        score = json.loads((self.directory / "papers" / "2103.14030" / "score.json").read_text())
        detail = score["detail"]["classification"]
        count = ClassificationResult.objects.filter(pretrained_backbone__family=self.family).count()
        self.assertEqual(detail["matched"], count - 1)
        self.assertEqual([error["field"] for error in detail["errors"]], ["top_1"])
        self.assertLess(summary["overall"]["metric_accuracy"], 1)

    def test_results_from_other_papers_do_not_count_toward_recall(self):
        output = extraction_of(self.family)
        output["records"] = [record for record in output["records"] if record["kind"] != "instance"]
        summary = self.evaluate(output)
        detail = json.loads((self.directory / "papers" / "2103.14030" / "score.json").read_text())["detail"]["instance"]
        linked = InstanceResult.objects.filter(pretrained_backbone__family=self.family).exclude(paper="").count()
        total = InstanceResult.objects.filter(pretrained_backbone__family=self.family).count()
        own = total - InstanceResult.objects.filter(pretrained_backbone__family=self.family).exclude(paper="").exclude(
            paper__contains="2103.14030").count()
        self.assertGreater(linked, 0)
        self.assertEqual((detail["truth"], len(detail["missed"])), (own, own))

    def test_results_for_other_models_are_scored_as_baselines(self):
        output = extraction_of(self.family)
        stored = {"key": "convnext-stored", "kind": "classification", "evidence": [], "uncertain": [],
                  "fields": [{"name": n, "value": v} for n, v in [("pretrained_backbone", "ConvNeXt-T-IN1k"),
                                                                  ("dataset", "ImageNet-1k"), ("resolution", 224),
                                                                  ("top_1", 82.1)]]}
        new = {**stored, "key": "convnext-new", "fields": [{"name": n, "value": v} for n, v in
                                                          [("pretrained_backbone", "ConvNeXt-T-IN1k"),
                                                           ("dataset", "ImageNet-1k"), ("resolution", 999), ("top_1", 80.0)]]}
        output["records"] += [stored, new]
        summary = self.evaluate(output)
        paper = summary["per_paper"]["2103.14030"]
        self.assertEqual((paper["baselines"]["extracted"], paper["baselines"]["already_stored"], paper["baselines"]["new"]),
                         (2, 1, 1))
        # They do not count against the paper's own precision.
        self.assertEqual(summary["overall"]["result_precision"], 1)

    def test_database_writes_are_rejected(self):
        with no_database_writes(), self.assertRaises(DatabaseWrite):
            BackboneFamily.objects.filter(name="Swin").update(github="")

    def test_papers_group_families_and_hide_their_names(self):
        papers, skipped = eval_papers(20)
        self.assertEqual(sum(len(families) for families in papers.values()) + len(skipped), 20)
        self.assertIn("2103.14030", papers)  # stored as a doi.org link
        words = hidden_vocabulary([self.family])
        self.assertNotIn("Swin", words["family"])
        self.assertIn("ConvNeXt", words["family"])

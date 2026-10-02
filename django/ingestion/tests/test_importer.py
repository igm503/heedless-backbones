from io import StringIO
from pathlib import Path

import yaml
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from stats.models import (
    Backbone, BackboneFamily, Category, ClassificationResult, FPSMeasurement, InstanceResult,
    PretrainedBackbone, SemanticSegmentationResult,
)
from ingestion.evidence import derive
from ingestion.family_yaml import family_to_dict, yaml_to_entries
from ingestion.importer import apply_entries, apply_run, approve_run
from ingestion.models import ExtractedRecord, ImportChange, IngestionRun, PaperVersion

TEXT = """FixtureNet introduces a Convolution backbone with Supervised pretraining on ImageNet-1k.
FixtureNet-T has 28 parameters, 224 resolution, 300 epochs, 83.1 top-1, 95.4 top-5 and 4.5 GFLOPs.
Mask R-CNN on COCO (train) / COCO (val), Object Detection, 12 epochs, 45.0 mAP and 100 GFLOPs.
We train for 80000 iterations using an effective batch size of 16 on 20000 images, or 64 epochs.
Novel mixer and Novel pretraining are proposed. Classification and object detection are evaluated.
"""


def citation(quote=TEXT):
    return {"page": 1, "location": "Table 1, FixtureNet-T row", "quote": quote.strip()}


def decision():
    return {"qualifies": True, "architecture_or_pretraining_contribution": True,
            "imagenet_1k_results": True, "downstream_results": True,
            "reason": "New backbone with both evaluations", "evidence": [citation()]}


class ImportTests(TestCase):
    fixtures = [str(Path(__file__).resolve().parents[3] / "db.json")]

    def setUp(self):
        self.paper = PaperVersion.objects.create(arxiv_id="2609.12345", revision="2026-09-01", version=1,
                                                title="FixtureNet", pages=[TEXT], metadata={"created": "2026-09-01T00:00:00Z"})
        self.run = IngestionRun.objects.create(paper=self.paper, provider="claude-code", model="test", decision=decision())
        self.add("family", "family", name="FixtureNet", model_type="Convolution", hierarchical=True, pretrain_method="Supervised")
        self.add("backbone", "backbone", name="FixtureNet-T", family="$family", m_parameters=28)
        self.add("pretrain", "pretrained_backbone", name="FixtureNet-T-IN1k", family="$family", backbone="$backbone",
                 pretrain_dataset="ImageNet-1k", pretrain_method="Supervised", pretrain_resolution=224, pretrain_epochs=300)
        self.add("classification", "classification", pretrained_backbone="$pretrain", dataset="ImageNet-1k",
                 resolution=224, top_1=83.1, top_5=95.4, gflops=4.5)
        self.add("detection", "instance", pretrained_backbone="$pretrain", head="Mask R-CNN", dataset="COCO (val)",
                 instance_type="Object Detection", train_dataset="COCO (train)", train_epochs=12, mAP=45.0, gflops=100)

    def add(self, key, kind, **data):
        return ExtractedRecord.objects.create(run=self.run, key=key, kind=kind, data=data,
            evidence=[{"field": field, "citation": citation(), "derivation": None} for field in data])

    def test_dry_run_then_publish_is_atomic_and_idempotent(self):
        self.assertEqual(apply_run(self.run), [])
        self.assertFalse(BackboneFamily.objects.filter(name="FixtureNet").exists())
        self.assertFalse(ImportChange.objects.exists())
        self.assertEqual(self.run.status, "ready")
        self.assertEqual(apply_run(self.run, publish=True), [])
        family = BackboneFamily.objects.get(name="FixtureNet")
        self.assertEqual(family.source_record.run_id, self.run.pk)
        self.assertEqual(ImportChange.objects.count(), 5)
        self.assertEqual(apply_run(self.run, publish=True), [])
        self.assertEqual(ImportChange.objects.count(), 5)
        self.assertEqual(ClassificationResult.objects.filter(pretrained_backbone__family=family).count(), 1)

    def test_rerun_does_not_duplicate_benchmarks(self):
        self.assertEqual(apply_run(self.run, publish=True), [])
        old_records = list(self.run.records.all())
        self.run = IngestionRun.objects.create(paper=self.paper, decision=decision())
        for old in old_records:
            old.pk, old.run = None, self.run
            old.status = "pending"
            old.save()
        self.assertEqual(apply_run(self.run, publish=True), [])
        self.assertEqual(ClassificationResult.objects.filter(pretrained_backbone__family__name="FixtureNet").count(), 1)
        self.assertEqual(ImportChange.objects.count(), 10)

    def test_late_validation_failure_rolls_back_all_live_changes(self):
        record = self.run.records.get(key="detection")
        record.data["head"] = "Does not exist"
        record.save()
        self.assertTrue(apply_run(self.run, publish=True))
        self.assertFalse(BackboneFamily.objects.filter(name="FixtureNet").exists())
        self.assertFalse(ImportChange.objects.exists())
        record.refresh_from_db()
        self.assertTrue(record.issues)
        self.assertEqual(self.run.status, "review")

    def test_fabricated_quote_and_number_are_blocked(self):
        record = self.run.records.get(key="classification")
        record.data["top_1"] = 99.9
        record.save()
        self.assertIn("not supported", apply_run(self.run, publish=True)[0])
        record.data["top_1"] = 83.1
        record.evidence[0]["citation"]["quote"] = "This quotation does not appear in the PDF"
        record.save()
        self.assertIn("does not occur", apply_run(self.run, publish=True)[0])

    def test_new_category_requires_approve_category(self):
        record = self.run.records.get(key="family")
        record.data["model_type"] = "Novel mixer"
        record.save()
        self.add("category", "category", scope="model_type", value="Novel mixer")
        self.assertIn("approve_category", apply_run(self.run, publish=True)[0])
        # Reviewing the run does not approve the category.
        approve_run(self.run, "reviewer", "Checked the architecture definition")
        self.assertIn("approve_category", apply_run(self.run, publish=True)[0])
        out = StringIO()
        call_command("approve_category", "--pending", stdout=out)
        self.assertIn("Novel mixer", out.getvalue())
        with self.assertRaises(CommandError):
            call_command("approve_category", "model_type", "Novel mixer")
        call_command("approve_category", "model_type", "Novel mixer", actor="reviewer",
                     note="Distinct mixing operator", stdout=StringIO())
        self.assertEqual(apply_run(self.run, publish=True), [])
        self.assertEqual(BackboneFamily.objects.get(name="FixtureNet").model_type, "Novel mixer")

    def test_unapproved_category_without_proposal_is_rejected(self):
        record = self.run.records.get(key="family")
        record.data["model_type"] = "Novel mixer"
        record.save()
        self.assertIn("not an approved category", apply_run(self.run, publish=True)[0])

    def test_missing_required_metadata_goes_to_review(self):
        record = self.run.records.get(key="pretrain")
        record.data.update(pretrain_epochs=None)
        record.save()
        self.assertIn("pretrain_epochs", apply_run(self.run, publish=True)[0])
        self.assertEqual(self.run.status, "review")
        self.assertFalse(BackboneFamily.objects.filter(name="FixtureNet").exists())

    def test_uncertain_field_goes_to_review_even_after_approval(self):
        record = self.run.records.get(key="backbone")
        record.uncertain = [{"field": "m_parameters", "reason": "Table reports detector parameters only"}]
        record.save()
        self.assertIn("Uncertain: m_parameters", apply_run(self.run, publish=True)[0])
        approve_run(self.run, "reviewer", "Looked at it")
        self.assertIn("Uncertain", apply_run(self.run, publish=True)[0])
        self.assertFalse(BackboneFamily.objects.filter(name="FixtureNet").exists())

    def test_review_draft_corrected_and_added_with_reference_yaml(self):
        record = self.run.records.get(key="backbone")
        record.uncertain = [{"field": "m_parameters", "reason": "Table reports detector parameters only"}]
        record.save()
        self.add("fps", "fps", owner="$backbone", backbone_name="FixtureNet-T", resolution=224,
                 fps=300, gpu="V100", precision="AMP", batch_size=16)
        apply_run(self.run, publish=True)
        out = StringIO()
        call_command("review_ingestion", "--list", stdout=out)
        self.assertIn(str(self.run.pk), out.getvalue())
        call_command("review_ingestion", str(self.run.pk), "--export", stdout=StringIO())
        draft_path = Path(settings.FAMILY_DATA_DIR) / "review" / f"FixtureNet-run{self.run.pk}.yml"
        draft = yaml.safe_load(draft_path.read_text())
        item = draft["review"]["items"][0]
        self.assertEqual((item["field"], item["evidence"]["page"]), ("m_parameters", 1))
        backbone = draft["backbones"][0]
        self.assertEqual(backbone["fps_measurements"][0]["fps"], 300)
        self.assertEqual(backbone["pretrained_backbones"][0]["instance_results"][0]["head"], "Mask R-CNN")
        with self.assertRaisesMessage(CommandError, "review block"):
            call_command("add_yaml", str(draft_path), run=self.run.pk)
        del draft["review"]
        backbone["m_parameters"] = 27.5
        draft_path.write_text(yaml.dump(draft, sort_keys=False))
        call_command("add_yaml", str(draft_path), run=self.run.pk, actor="reviewer", stdout=StringIO())
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "imported")
        self.assertEqual(Backbone.objects.get(name="FixtureNet-T").m_parameters, 27.5)
        result = InstanceResult.objects.get(pretrained_backbone__name="FixtureNet-T-IN1k")
        self.assertEqual(result.paper, self.paper.url)
        change = ImportChange.objects.get(model="stats.backbone", object_id=result.pretrained_backbone.backbone_id, before__isnull=True)
        self.assertEqual((change.run_id, change.actor, change.record), (self.run.pk, "reviewer", None))
        self.assertTrue(change.origin.endswith(draft_path.name))
        reference = yaml.safe_load((Path(settings.FAMILY_DATA_DIR) / "FixtureNet.yml").read_text())
        self.assertEqual(reference["backbones"][0]["m_parameters"], 27.5)
        with self.assertRaisesMessage(CommandError, "already been imported"):
            call_command("add_yaml", str(draft_path), run=self.run.pk)

    def test_conflicting_existing_result_is_not_overwritten(self):
        self.assertEqual(apply_run(self.run, publish=True), [])
        result = InstanceResult.objects.get(pretrained_backbone__name="FixtureNet-T-IN1k")
        result.mAP = 44
        result.save()
        self.run.status = "ready"
        self.run.save()
        problem = apply_run(self.run, publish=True)[0]
        self.assertIn("mAP (44.0 stored, 45.0 in this paper)", problem)
        self.assertIn("Proposed correction", problem)
        result.refresh_from_db()
        self.assertEqual(result.mAP, 44)
        approve_run(self.run, "reviewer", "Correcting transcription to Table 1")
        self.assertEqual(apply_run(self.run, publish=True, actor="reviewer", allow_updates=True), [])
        change = ImportChange.objects.filter(model="stats.instanceresult").latest("pk")
        self.assertEqual(change.before["mAP"], 44)
        self.assertEqual(change.after["mAP"], 45)

    def test_cycles_and_missing_references_do_not_import(self):
        record = self.run.records.get(key="backbone")
        record.data["family"] = "$missing"
        record.save()
        self.assertIn("Missing or cyclic", apply_run(self.run, publish=True)[0])
        self.assertFalse(BackboneFamily.objects.filter(name="FixtureNet").exists())

    def test_fractional_epoch_derivation_is_preserved(self):
        record = self.run.records.get(key="detection")
        record.data["train_epochs"] = 80000 * 16 / 300
        item = next(e for e in record.evidence if e["field"] == "train_epochs")
        item["derivation"] = {"method": "iterations_to_epochs", "assumptions": [], "inputs": [
            {"name": name, "value": value, "citation": citation()}
            for name, value in [("iterations", 80000), ("effective_batch_size", 16), ("dataset_size", 300)]
        ]}
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])
        self.assertAlmostEqual(InstanceResult.objects.get(pretrained_backbone__name="FixtureNet-T-IN1k").train_epochs, 4266.666666666667)
        item["derivation"]["inputs"][2]["value"] = 300
        self.assertAlmostEqual(derive(item["derivation"], self.paper.pages), 4266.666666666667)

    def test_judgment_on_cited_inputs_publishes(self):
        record = self.run.records.get(key="detection")
        record.data["train_epochs"] = 64
        record.note = "Dataset size interpreted as the training split"
        item = next(e for e in record.evidence if e["field"] == "train_epochs")
        item["derivation"] = {"method": "iterations_to_epochs", "assumptions": ["Dataset size interpreted as training split"], "inputs": [
            {"name": name, "value": value, "citation": citation()}
            for name, value in [("iterations", 80000), ("effective_batch_size", 16), ("dataset_size", 20000)]
        ]}
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_guide_dataset_size_needs_no_citation(self):
        record = self.run.records.get(key="detection")
        record.data["train_epochs"] = 80000 * 16 / 117000
        item = next(e for e in record.evidence if e["field"] == "train_epochs")
        item["derivation"] = {"method": "iterations_to_epochs", "assumptions": ["COCO train2017 by convention"], "inputs": [
            {"name": "iterations", "value": 80000, "citation": citation()},
            {"name": "effective_batch_size", "value": 16, "citation": citation()},
            {"name": "dataset_size", "value": 117000, "citation": None, "source": "convention: COCO"},
        ]}
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_standard_batch_size_convention(self):
        record = self.run.records.get(key="detection")
        record.data["train_epochs"] = 128
        item = next(e for e in record.evidence if e["field"] == "train_epochs")
        item["derivation"] = {"method": "iterations_to_epochs", "assumptions": ["Standard ConvNeXt setup"], "inputs": [
            {"name": "iterations", "value": 160000, "citation": citation("We train for 160000 iterations")},
            {"name": "effective_batch_size", "value": 16, "citation": None, "source": "convention: standard batch size"},
            {"name": "dataset_size", "value": 20000, "citation": None, "source": "convention: ADE20K"},
        ]}
        self.paper.pages = [TEXT + " We train for 160000 iterations."]
        self.paper.save()
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])
        item["derivation"]["inputs"][1]["value"] = 32  # Not the convention
        record.data["train_epochs"] = 256
        record.save()
        self.run.status = "ready"
        self.run.save()
        self.assertIn("need review", apply_run(self.run)[0])

    def test_named_schedule_supports_epochs(self):
        schedule = "Mask R-CNN on COCO with the 1x schedule reaches 45.0 mAP."
        self.paper.pages = [TEXT + schedule]
        self.paper.save()
        record = self.run.records.get(key="detection")
        item = next(e for e in record.evidence if e["field"] == "train_epochs")
        item["citation"] = citation(schedule)
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_references_names_and_repeated_citations_are_accepted(self):
        record = self.run.records.get(key="classification")
        record.evidence = [e for e in record.evidence if e["field"] != "pretrained_backbone"]
        record.evidence.append({"field": "top_1", "citation": citation(), "derivation": None})
        record.save()
        backbone = self.run.records.get(key="backbone")
        backbone.evidence = [e for e in backbone.evidence if e["field"] != "name"]
        backbone.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_abbreviated_numbers_and_judgment_fields(self):
        from ingestion.evidence import contains_number
        self.assertTrue(contains_number("ConvNeXt-T   224^2 29M 4.5G", 224))
        self.assertTrue(contains_number("trained for 160K iterations", 160000))
        self.assertTrue(contains_number("4.5G FLOPs at 224px", 4.5) and contains_number("224px", 224))
        record = self.run.records.get(key="pretrain")
        record.evidence = [e for e in record.evidence if e["field"] != "pretrain_method"]
        record.save()
        self.assertIn("pretrain_method has no quotation", apply_run(self.run, publish=True)[0])
        record.inferred = [{"field": "pretrain_method", "source": "trained with labels only (Sec. 3)"}]
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_quote_may_end_at_a_line_break_hyphen(self):
        from ingestion.evidence import normalized
        page = normalized("When training from scratch with a 224^2 input, we em-\nploy an AdamW optimizer")
        self.assertIn(normalized("with a 224^2 input, we em-"), page)
        self.assertIn(normalized("ploy an AdamW optimizer"), page)

    def test_quotes_match_across_ligatures_and_hyphenation(self):
        self.paper.pages = [TEXT + "We propose a pure CNN architec-\nture, finetuned using Cascade Mask-RCNN."]
        self.paper.save()
        record = self.run.records.get(key="family")
        record.evidence[0]["citation"]["quote"] = "a pure CNN architec-ture, \ufb01netuned using Cascade Mask-RCNN."
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_reference_without_quote_needs_a_note(self):
        record = self.run.records.get(key="detection")
        record.evidence = [e for e in record.evidence if e["field"] != "head"]
        record.save()
        self.assertIn("head has no quotation", apply_run(self.run, publish=True)[0])
        record.inferred = [{"field": "head", "source": "follows the standard Mask R-CNN setup (Sec. 4)"}]
        record.save()
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_new_head_needs_approval(self):
        self.add("head", "head", name="Novel Head", tasks=["Object Detection"])
        record = self.run.records.get(key="detection")
        record.data["head"] = "$head"
        record.save()
        self.assertIn("New head 'Novel Head' needs approval", apply_run(self.run, publish=True)[0])
        approve_run(self.run, "reviewer", "Head described in Section 3")
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_imagenet_1k_required_and_downstream_optional(self):
        self.run.records.filter(key="detection").delete()
        self.assertEqual(apply_run(self.run), [])
        record = self.run.records.get(key="classification")
        record.data["dataset"] = "ImageNet-V2"
        record.save()
        self.assertIn("ImageNet-1k classification", apply_run(self.run)[0])

    def test_screening_alone_cannot_publish(self):
        self.run.decision["evidence"] = []
        self.assertIn("selection evidence", apply_run(self.run, publish=True)[0])
        self.assertFalse(ImportChange.objects.exists())

    def test_conventional_input_requires_source_and_review(self):
        record = self.run.records.get(key="detection")
        record.data["train_epochs"] = 80000 * 16 / 118287
        item = next(e for e in record.evidence if e["field"] == "train_epochs")
        item["derivation"] = {"method": "iterations_to_epochs", "assumptions": ["COCO train2017 split"], "inputs": [
            {"name": "iterations", "value": 80000, "citation": citation()},
            {"name": "effective_batch_size", "value": 16, "citation": citation()},
            {"name": "dataset_size", "value": 118287, "citation": None, "source": "COCO train2017 image count"},
        ]}
        record.save()
        self.assertIn("need review", apply_run(self.run)[0])
        approve_run(self.run, "reviewer", "Verified training split size against dataset documentation")
        self.assertEqual(apply_run(self.run, publish=True), [])

    def test_incomplete_metadata_does_not_duplicate_result(self):
        self.assertEqual(apply_run(self.run, publish=True), [])
        record = self.run.records.get(key="detection")
        record.data["train_epochs"] = None
        record.save()
        self.run.status = "ready"
        self.run.save()
        self.assertIn("may duplicate", apply_run(self.run, publish=True)[0])
        self.assertEqual(InstanceResult.objects.filter(pretrained_backbone__name="FixtureNet-T-IN1k").count(), 1)

    def test_throughput_attachment_has_its_own_change_record(self):
        self.add("fps", "fps", owner="$backbone", backbone_name="FixtureNet-T", resolution=224,
                 fps=300, gpu="V100", precision="AMP", batch_size=16)
        self.assertEqual(apply_run(self.run, publish=True), [])
        changes = ImportChange.objects.filter(record__key="fps")
        self.assertEqual(changes.count(), 2)
        owner_change = changes.get(model="stats.backbone")
        self.assertEqual(owner_change.before["fps_measurements"], [])
        self.assertEqual(len(owner_change.after["fps_measurements"]), 1)

    def test_admin_correction_preserves_before_and_after(self):
        from django.contrib.auth import get_user_model
        from django.urls import reverse
        self.assertEqual(apply_run(self.run, publish=True), [])
        family = BackboneFamily.objects.get(name="FixtureNet")
        user = get_user_model().objects.create_superuser(username="reviewer", password="test")
        self.client.force_login(user)
        response = self.client.post(reverse("admin:stats_backbonefamily_change", args=[family.pk]), {
            "name": family.name, "model_type": family.model_type, "pretrain_method": family.pretrain_method,
            "pub_date": "2026-09-01", "paper": family.paper, "github": "", "_save": "Save",
        })
        self.assertEqual(response.status_code, 302)
        change = ImportChange.objects.filter(model="stats.backbonefamily", actor="reviewer").get()
        self.assertTrue(change.before["hierarchical"])
        self.assertFalse(change.after["hierarchical"])

    def test_unknown_finetuning_resolution_goes_to_review(self):
        record = self.run.records.get(key="classification")
        record.data.update(fine_tune_dataset="ImageNet-1k", fine_tune_resolution=None)
        record.evidence.append({"field": "fine_tune_dataset", "citation": citation(), "derivation": None})
        record.save()
        self.assertIn("fine tune resolution is required", apply_run(self.run, publish=True)[0])
        self.assertFalse(ClassificationResult.objects.filter(pretrained_backbone__name="FixtureNet-T-IN1k").exists())


class FamilyYamlTests(TestCase):
    fixtures = [str(Path(__file__).resolve().parents[3] / "db.json")]

    def test_every_family_round_trips_through_yaml(self):
        for family in BackboneFamily.objects.all():
            with self.subTest(family.name):
                before = yaml.safe_load(yaml.dump(family_to_dict(family), sort_keys=False))
                # Re-importing the unchanged file matches every existing object...
                counts = [model.objects.count() for model in (Backbone, PretrainedBackbone, InstanceResult)]
                if family.name not in AMBIGUOUS:
                    apply_entries(yaml_to_entries(before), actor="test", origin="round trip")
                    self.assertEqual([model.objects.count() for model in (Backbone, PretrainedBackbone, InstanceResult)], counts)
                # ...and rebuilding the family from it reproduces the same file.
                family.delete()
                apply_entries(yaml_to_entries(before), actor="test", origin="round trip")
                after = family_to_dict(BackboneFamily.objects.get(name=family.name))
                self.assertEqual(before, yaml.safe_load(yaml.dump(after, sort_keys=False)))

    def test_result_source_links_survive_the_round_trip(self):
        swin = BackboneFamily.objects.get(name="Swin")
        linked = InstanceResult.objects.filter(pretrained_backbone__family=swin).exclude(paper="").count()
        self.assertGreater(linked, 0)
        data = yaml.safe_load(yaml.dump(family_to_dict(swin), sort_keys=False))
        swin.delete()
        apply_entries(yaml_to_entries(data), actor="test", origin="round trip")
        self.assertEqual(InstanceResult.objects.filter(pretrained_backbone__family__name="Swin").exclude(paper="").count(), linked)

    def test_spiking_is_written_only_when_true(self):
        swin = BackboneFamily.objects.get(name="Swin")
        data = yaml.safe_load(yaml.dump(family_to_dict(swin), sort_keys=False))
        self.assertNotIn("spiking", data)
        swin.delete()
        apply_entries(yaml_to_entries(data), actor="test", origin="round trip")
        self.assertFalse(BackboneFamily.objects.get(name="Swin").spiking)

        BackboneFamily.objects.filter(name="Swin").update(spiking=True)
        data = yaml.safe_load(yaml.dump(family_to_dict(BackboneFamily.objects.get(name="Swin")), sort_keys=False))
        self.assertEqual(list(data)[:4], ["name", "model_type", "hierarchical", "spiking"])
        self.assertIs(data["spiking"], True)
        BackboneFamily.objects.get(name="Swin").delete()
        apply_entries(yaml_to_entries(data), actor="test", origin="round trip")
        self.assertTrue(BackboneFamily.objects.get(name="Swin").spiking)


# Families with results whose recorded settings coincide (e.g. two FocalNet-T-SRF Mask R-CNN
# rows); re-importing them into an existing family cannot tell which row is which.
AMBIGUOUS = {"FocalNet", "InternImage", "UniConvNet"}

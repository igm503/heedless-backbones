import io
import tempfile

import pdfplumber

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings
from django.urls import reverse

from ingestion import review
from ingestion.importer import apply_run
from ingestion.models import ExtractedRecord, IngestionRun
from ingestion.tests import test_importer
from ingestion.tests.test_importer import TEXT
from stats.models import BackboneFamily, ClassificationResult


def pdf_with_text(lines):
    """A minimal one-page PDF showing each line of text (Helvetica, 10pt)."""
    body = "BT /F1 10 Tf 40 760 Td 12 TL " + " ".join(
        "(" + line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*" for line in lines) + " ET"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
               f"<< /Length {len(body)} >>\nstream\n{body}\nendstream",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = "%PDF-1.4\n", []
    for number, content in enumerate(objects, 1):
        offsets.append(len(out.encode()))
        out += f"{number} 0 obj\n{content}\nendobj\n"
    xref = len(out.encode())
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ReviewTests(TestCase):
    fixtures = test_importer.ImportTests.fixtures
    add = test_importer.ImportTests.add

    def setUp(self):
        test_importer.ImportTests.setUp(self)
        self.paper.pdf.save("fixture.pdf", ContentFile(pdf_with_text(TEXT.splitlines())), save=True)
        user = get_user_model().objects.create_superuser(username="reviewer", password="test")
        self.client.force_login(user)
        self.url = reverse("admin:ingestion_review_run", args=[self.run.pk])

    def test_quote_is_located_and_cropped(self):
        with pdfplumber.open(io.BytesIO(self.paper.pdf.open("rb").read())) as pdf:
            self.assertIsNotNone(review.quote_box(pdf.pages[0], "83.1 top-1, 95.4 top-5"))
        record = self.run.records.get(key="classification")
        response = self.client.get(reverse("admin:ingestion_review_crop", args=[self.run.pk, record.pk, 0]))
        self.assertEqual((response.status_code, response["Content-Type"]), (200, "image/png"))
        self.assertTrue(response.content.startswith(b"\x89PNG"))

    def test_flagged_view_and_full_view(self):
        record = self.run.records.get(key="backbone")
        record.uncertain = [{"field": "m_parameters", "reason": "Table reports detector parameters"}]
        record.save()
        apply_run(self.run)
        flagged = self.client.get(self.url)
        self.assertContains(flagged, "Table reports detector parameters")
        self.assertNotContains(flagged, "FixtureNet-T-IN1k")
        self.assertContains(self.client.get(self.url + "?show=all"), "FixtureNet-T-IN1k")
        self.assertContains(self.client.get(reverse("admin:ingestion_review")), f">{self.run.pk}<")

    def test_correct_then_approve_publishes_with_the_reviewer_as_source(self):
        record = self.run.records.get(key="backbone")
        record.uncertain = [{"field": "m_parameters", "reason": "unclear"}]
        record.evidence = [e for e in record.evidence if e["field"] != "m_parameters"]
        record.save()
        self.assertTrue(apply_run(self.run))
        self.client.post(self.url, {"action": "correct", "record": record.pk, "field": "m_parameters",
                                    "value": "27.5", "note": "Table 3 lists 27.5M for the backbone"})
        record.refresh_from_db()
        self.assertEqual((record.data["m_parameters"], record.overrides[0]["old"]), (27.5, 28))
        self.assertEqual(record.uncertain, [])
        response = self.client.post(self.url, {"action": "approve", "note": "Checked against the paper"}, follow=True)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "imported", [str(m) for m in response.context["messages"]])
        self.assertTrue(BackboneFamily.objects.filter(name="FixtureNet").exists())

    def test_uncertain_filter_shows_only_uncertain_fields_in_the_yaml_layout(self):
        record = self.run.records.get(key="backbone")
        record.uncertain = [{"field": "m_parameters", "reason": "Detector parameters only"}]
        record.save()
        detection = self.run.records.get(key="detection")
        detection.note = "Head inferred from the setup section"
        detection.save()
        page = self.client.get(self.url + "?show=uncertain")
        self.assertContains(page, "Detector parameters only")
        self.assertContains(page, "FixtureNet")               # the family, as context
        self.assertNotContains(page, "Head inferred from")    # flagged, but not uncertain
        self.assertContains(self.client.get(self.url), "Head inferred from")
        everything = self.client.get(self.url + "?show=all").content.decode()
        # Hierarchy: family, then backbone, then pretrained model, then its results.
        order = [everything.index(text) for text in ("FixtureNet-T-IN1k", "ImageNet-1k @ 224", "Mask R-CNN · COCO (val)")]
        self.assertEqual(order, sorted(order))

    def test_removing_a_section_removes_everything_under_it(self):
        pretrain = self.run.records.get(key="pretrain")
        self.client.post(self.url, {"action": "remove", "record": pretrain.pk, "note": "ablation model"})
        statuses = dict(self.run.records.values_list("key", "status"))
        self.assertEqual({statuses[k] for k in ("pretrain", "classification", "detection")}, {"rejected"})
        self.assertEqual(statuses["backbone"], "pending")
        self.run.refresh_from_db()
        self.assertIn("ImageNet-1k classification", self.run.error)  # nothing left to publish
        self.client.post(self.url, {"action": "restore", "record": pretrain.pk})
        self.assertFalse(self.run.records.filter(status="rejected").exists())
        # Removing only the detection result still publishes the rest.
        self.client.post(self.url, {"action": "remove", "record": self.run.records.get(key="detection").pk})
        self.client.post(self.url, {"action": "approve", "note": "Checked"})
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "imported", self.run.error)
        from stats.models import InstanceResult
        self.assertFalse(InstanceResult.objects.filter(pretrained_backbone__name="FixtureNet-T-IN1k").exists())

    def test_head_proposal_lists_and_removes_its_results(self):
        head = self.add("head", "head", name="Novel Head", tasks=["Object Detection"])
        detection = self.run.records.get(key="detection")
        detection.data["head"] = "$head"
        detection.save()
        page = self.client.get(self.url)
        self.assertContains(page, "Used by 1 result")
        self.assertContains(page, "FixtureNet-T-IN1k — Novel Head · COCO (val)")
        # The result is also shown in full inside the proposal, with its evidence, even though
        # nothing about it is flagged.
        html = page.content.decode()
        inside = html[html.index(f'id="proposal-{detection.pk}"'):]
        inside = inside[:inside.index('class="section-title"')]
        self.assertIn("<strong>mAP</strong>", inside)
        self.assertIn(f"/crop/{detection.pk}/", inside)
        self.assertIn("Novel Head (proposed in this run)", inside)
        self.client.post(self.url, {"action": "remove", "record": head.pk, "note": "not a real head"})
        detection.refresh_from_db()
        self.assertEqual(detection.status, "rejected")

    def test_reject_and_retry_queues_a_read_with_the_feedback(self):
        from unittest.mock import patch
        from ingestion import agent
        from ingestion.pipeline import shortlist
        from ingestion.review import retry_feedback
        backbone = self.run.records.get(key="backbone")
        self.client.post(self.url, {"action": "correct", "record": backbone.pk, "field": "m_parameters",
                                    "value": "27.5", "note": "backbone only"})
        self.client.post(self.url, {"action": "remove", "record": self.run.records.get(key="detection").pk,
                                    "note": "1x schedule is an ablation"})
        other = IngestionRun.objects.create(paper=self.paper, status="shortlisted")
        self.client.post(self.url, {"action": "reject_retry", "note": "Use Table 2 for FLOPs"})
        self.run.refresh_from_db()
        retry = self.run.retries.get()
        self.assertEqual((self.run.status, retry.status, retry.feedback), ("rejected", "shortlisted", "Use Table 2 for FLOPs"))
        self.assertEqual(shortlist(5)[0], retry)  # retries are read first
        self.assertIn(other, shortlist(5))
        feedback = retry_feedback(retry)
        for text in ("Use Table 2 for FLOPs", "1x schedule is an ablation", "m_parameters: 28 -> 27.5 (backbone only)"):
            self.assertIn(text, feedback)
        # The agent's prompt carries the feedback and the rejected records.
        seen = {}
        with patch("ingestion.agent.prepare", side_effect=lambda folder, identifier, **kw: seen.update(kw)), \
                patch("ingestion.agent.run_agent", return_value={}), patch("ingestion.agent.submit", return_value=retry):
            agent.AgentRunner().read(retry)
        self.assertIn("Use Table 2 for FLOPs", seen["feedback"])
        self.assertEqual(len(seen["previous"]), self.run.records.count())
        self.assertIn("REVIEWER FEEDBACK", agent.agent_prompt("2609.12345", 3, seen["feedback"]))
        # A note is required.
        response = self.client.post(reverse("admin:ingestion_review_run", args=[retry.pk]),
                                    {"action": "reject_retry", "note": ""}, follow=True)
        self.assertIn("Say what the retry should do differently", [str(m) for m in response.context["messages"]])
        self.assertFalse(retry.retries.exists())

    def test_uncertain_values_block_until_accepted_or_corrected(self):
        record = self.run.records.get(key="classification")
        record.uncertain = [{"field": "top_1", "reason": "Table 1 and the text disagree"}]
        record.save()
        self.client.post(self.url, {"action": "approve", "note": "Looked fine"})
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "review")  # approval alone does not clear it
        self.client.post(self.url, {"action": "accept", "record": record.pk, "field": "top_1",
                                    "note": "Table 1 is the main result"})
        record.refresh_from_db()
        self.assertEqual((record.uncertain, record.data["top_1"], record.overrides[0]["note"]),
                         ([], 83.1, "Table 1 is the main result"))
        self.client.post(self.url, {"action": "approve", "note": "Checked"})
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "imported", self.run.error)

    def test_reject_and_notes_are_required(self):
        self.client.post(self.url, {"action": "reject", "note": ""})
        self.run.refresh_from_db()
        self.assertNotEqual(self.run.status, "rejected")
        self.client.post(self.url, {"action": "reject", "note": "Not a backbone paper"})
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "rejected")
        self.assertFalse(ExtractedRecord.objects.filter(run=self.run).exclude(status="rejected").exists())

    def test_imported_runs_cannot_be_corrected_here(self):
        self.assertEqual(apply_run(self.run, publish=True), [])
        record = self.run.records.get(key="classification")
        self.client.post(self.url, {"action": "correct", "record": record.pk, "field": "top_1", "value": "80",
                                    "note": "test"})
        self.assertEqual(ClassificationResult.objects.get(pretrained_backbone__name="FixtureNet-T-IN1k").top_1, 83.1)
        self.assertContains(self.client.get(self.url + "?show=all"), "stored as stats.classificationresult")

import json
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

from django.core.management import call_command
from django.test import TestCase

from ingestion import agent
from ingestion.models import IngestionRun, PaperVersion
from ingestion.pipeline import candidates, discover, shortlist


def decision(arxiv_id, qualifies):
    return {"arxiv_id": arxiv_id, "qualifies": qualifies, "architecture_or_pretraining_contribution": qualifies,
            "imagenet_1k_results": qualifies, "downstream_results": False, "reason": "test"}


def claude_result(decisions, **extra):
    """What `claude -p --output-format json --json-schema ...` prints."""
    return Mock(stdout=json.dumps({"type": "result", "is_error": False, "total_cost_usd": 0.02, "session_id": "s",
                                   "usage": {"input_tokens": 3000, "output_tokens": 300},
                                   "modelUsage": {"claude-opus-5-5": {"outputTokens": 300}},
                                   "structured_output": {"decisions": decisions}, **extra}), stderr="")


class ScreeningTests(TestCase):
    def setUp(self):
        self.papers = [PaperVersion.objects.create(arxiv_id=f"2609.1000{i}", revision="2026-09-01", title=f"Paper {i}",
                                                   abstract="Abstract") for i in range(3)]

    def test_batches_are_screened_without_tools_or_api_keys(self):
        result = claude_result([decision("2609.10000", True), decision("2609.10001", False)])
        with patch("ingestion.agent.subprocess.run", return_value=result) as run, \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "PATH": "/usr/bin"}):
            runs = agent.screen(self.papers)
        # (The later subprocess call is `git rev-parse` for the code version.)
        claude_call = next(call for call in run.call_args_list if call.args[0][0] == "claude")
        command, kwargs = claude_call.args[0], claude_call.kwargs
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertIn("--json-schema", command)
        self.assertNotIn("ANTHROPIC_API_KEY", kwargs["env"])
        self.assertEqual([r.status for r in runs], ["shortlisted", "rejected", "failed"])
        self.assertEqual((runs[0].provider, runs[0].model), ("claude-code", "claude-opus-5-5"))
        # The batch's usage is split across its three papers.
        self.assertEqual((runs[0].input_tokens, runs[0].output_tokens), (1000, 100))
        self.assertEqual(runs[0].calls[0]["stage"], "abstract")
        self.assertEqual([r.pk for r in shortlist(5)], [runs[0].pk])
        # The paper without a decision can be retried; the screened ones are done.
        self.assertEqual(candidates(5), [self.papers[2]])

    def test_a_failed_session_marks_the_batch_failed(self):
        with patch("ingestion.agent.subprocess.run", return_value=Mock(stdout="not json", stderr="boom")):
            runs = agent.screen(self.papers[:2])
        self.assertEqual({r.status for r in runs}, {"failed"})
        self.assertIn("no JSON", runs[0].error)

    def test_batch_size(self):
        with patch("ingestion.agent.screen_batch", return_value=({}, {"stage": "abstract"})) as batch:
            agent.screen(self.papers, batch=2)
        self.assertEqual([len(call.args[0]) for call in batch.call_args_list], [2, 1])

    def test_three_failures_stop_automatic_retries(self):
        for _ in range(3):
            IngestionRun.objects.create(paper=self.papers[0], status="failed")
        self.assertNotIn(self.papers[0], candidates(5))

    def test_ingest_screens_then_reads_up_to_the_limit(self):
        screened = [IngestionRun.objects.create(paper=paper, status="shortlisted") for paper in self.papers]
        read = Mock(side_effect=lambda run: run)
        with patch("ingestion.management.commands.ingest_papers.screen", return_value=[]) as screen, \
                patch("ingestion.agent.AgentRunner.read", read):
            call_command("ingest_papers", "--skip-discovery", "--screen-limit", "4", "--limit", "2", stdout=StringIO())
        self.assertEqual(screen.call_args.kwargs["batch"], 20)
        self.assertEqual([call.args[0].pk for call in read.call_args_list], [screened[0].pk, screened[1].pk])


class DiscoveryTests(TestCase):
    def test_discovery_is_repeatable_and_syncs_existing_ids(self):
        item = {"arxiv_id": "2609.54321", "title": "New", "abstract": "Abstract",
                "created": "2026-09-01T00:00:00+00:00", "updated": "2026-09-02T00:00:00+00:00"}
        joint = dict(item, arxiv_id="2609.54322")
        client = Mock()
        client.papers.return_value = [item]
        client.tag_search.return_value = [joint]
        client.sync.return_value = []
        with patch("ingestion.pipeline.current_paper_ids", return_value={"2501.12345"}):
            created, missing = discover(client, "backbones", "working")
            again, _ = discover(client, "backbones", "working")
        self.assertEqual((created, again), (2, 0))
        client.sync.assert_called_with("backbones", "working", ["2501.12345"])
        self.assertEqual(client.tag_search.call_args.args[0], "working")
        self.assertEqual(PaperVersion.objects.get(arxiv_id="2609.54322").metadata["discovery"]["action"], "tag_search")

    def test_download_pins_the_arxiv_version(self):
        from ingestion.sources import fetch_pdf
        metadata = Mock(content=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2609.12345v2</id></entry></feed>')
        pdf = Mock()
        pdf.iter_content.return_value = [b"%PDF-fixture-bytes"]
        response = MagicMock()
        response.__enter__.return_value = pdf
        session = Mock()
        session.get.side_effect = [metadata, response]
        with patch("ingestion.sources.pdf_pages", return_value=["Source page text"]):
            version, data, pages = fetch_pdf("2609.12345", session)
        self.assertEqual((version, data, pages), (2, b"%PDF-fixture-bytes", ["Source page text"]))
        self.assertEqual(session.get.call_args.args[0], "https://arxiv.org/pdf/2609.12345v2")

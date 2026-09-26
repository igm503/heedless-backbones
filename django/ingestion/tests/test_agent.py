import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, TestCase

from ingestion import agent
from ingestion.evidence import check_citation
from ingestion.models import ExtractedRecord, ImportChange, IngestionRun, PaperVersion, WebSource
from ingestion.tests.test_importer import TEXT, citation, decision
from ingestion.web_sources import check_allowed, fetch, html_text, official_links
from stats.models import BackboneFamily, ClassificationResult

SWIN = "Swin-T on ImageNet-1k at 256 resolution: 83.1 top-1 with 4.5 GFLOPs."
README = "| Model | IN-22K top-1 |\n| FixtureNet-T | 85.2 |\nFixtureNet-T 22k checkpoint: 85.2 top-1"


def extraction(extra_records=()):
    def record(key, kind, **data):
        return {"key": key, "kind": kind, "fields": [{"name": k, "value": v} for k, v in data.items()],
                "evidence": [{"field": f, "citation": {**citation(), "url": None}, "derivation": None} for f in data],
                "uncertain": [], "inferred": [], "note": ""}
    records = [
        record("family", "family", name="FixtureNet", model_type="Convolution", hierarchical=True, pretrain_method="Supervised"),
        record("backbone", "backbone", name="FixtureNet-T", family="$family", m_parameters=28),
        record("pretrain", "pretrained_backbone", name="FixtureNet-T-IN1k", family="$family", backbone="$backbone",
               pretrain_dataset="ImageNet-1k", pretrain_method="Supervised", pretrain_resolution=224, pretrain_epochs=300),
        record("classification", "classification", pretrained_backbone="$pretrain", dataset="ImageNet-1k",
               resolution=224, top_1=83.1, top_5=95.4, gflops=4.5),
        *extra_records,
    ]
    return {"decision": {**decision(), "evidence": [{**citation(), "url": None}]}, "records": records}


class WorkFolder:
    def __init__(self, output, sources=()):
        self.path = Path(tempfile.mkdtemp())
        (self.path / "sources").mkdir()
        agent.write_json(self.path / "pages.json", [TEXT])
        agent.write_json(self.path / "links.json", ["https://github.com/fixture/fixturenet"])
        agent.write_json(self.path / "metadata.json", {"arxiv_id": "2609.12345", "updated": "2026-09-01T00:00:00Z",
                                                       "created": "2026-09-01T00:00:00Z", "title": "FixtureNet",
                                                       "abstract": "Abstract", "version": 1})
        (self.path / "paper.pdf").write_bytes(b"%PDF-fixture")
        for number, item in enumerate(sources, 1):
            agent.write_json(self.path / "sources" / f"{number:02d}.json", item)
        agent.write_json(self.path / agent.FINAL, output)


class AgentTests(TestCase):
    fixtures = [str(Path(__file__).resolve().parents[3] / "db.json")]

    def test_validation_is_rolled_back_and_reports_problems(self):
        folder = WorkFolder(extraction())
        counts = (IngestionRun.objects.count(), ExtractedRecord.objects.count(), BackboneFamily.objects.count())
        report = agent.validate(folder.path)
        self.assertTrue(report["publishable"], report)
        self.assertEqual((IngestionRun.objects.count(), ExtractedRecord.objects.count(), BackboneFamily.objects.count()), counts)
        output = extraction()
        output["records"][3]["fields"][3]["value"] = 99.9  # top_1 not in its quotation
        report = agent.validate(WorkFolder(output).path)
        self.assertFalse(report["publishable"])
        self.assertIn("not supported", report["problems"][0]["problem"])

    def test_web_sources_can_supply_values(self):
        source = {"url": "https://github.com/fixture/fixturenet", "fetched_from": "https://api.github.com/repos/fixture/fixturenet/readme",
                  "fetched_at": "2026-09-26T00:00:00+00:00", "sha256": "0" * 64, "text": README}
        web = {"page": None, "url": source["url"], "location": "README model table", "quote": "| FixtureNet-T | 85.2 |"}
        output = extraction()
        output["records"][3]["fields"][3]["value"] = 85.2
        output["records"][3]["evidence"][3]["citation"] = web
        self.assertTrue(agent.validate(WorkFolder(output, [source]).path)["publishable"])
        # Without the saved source, the same citation fails.
        self.assertIn("was not fetched", agent.validate(WorkFolder(output).path)["problems"][0]["problem"])
        run = agent.submit(WorkFolder(output, [source]).path, publish=True)
        self.assertEqual(run.status, "imported", run.error)
        self.assertEqual(WebSource.objects.get(run=run).text, README)
        self.assertEqual(ClassificationResult.objects.get(pretrained_backbone__name="FixtureNet-T-IN1k").top_1, 85.2)

    def test_repository_only_model_links_to_the_repository(self):
        source = {"url": "https://github.com/fixture/fixturenet", "fetched_from": "https://api.github.com/repos/fixture/fixturenet/readme",
                  "fetched_at": "2026-09-26T00:00:00+00:00", "sha256": "0" * 64, "text": README}
        output = extraction()
        output["records"][3]["fields"].append({"name": "paper", "value": source["url"]})
        run = agent.submit(WorkFolder(output, [source]).path, publish=True)
        self.assertEqual(run.status, "imported", run.error)
        self.assertEqual(ClassificationResult.objects.get(pretrained_backbone__name="FixtureNet-T-IN1k").paper, source["url"])
        # Any other link is refused.
        output["records"][3]["fields"][-1]["value"] = "https://example.com/elsewhere"
        self.assertIn("source fetched for it", agent.validate(WorkFolder(output, [source]).path)["import"])

    def test_submit_imports_through_the_ledger(self):
        run = agent.submit(WorkFolder(extraction()).path, agent={"model": "claude", "cost_usd": 1.5}, publish=True)
        self.assertEqual((run.status, run.provider), ("imported", "claude-code"))
        self.assertEqual(run.calls[0]["stage"], "agent")
        self.assertEqual(ImportChange.objects.filter(run=run).count(), 4)

    def test_gap_filling_result_for_an_existing_model(self):
        # A result the paper reports for a stored model, with no stored result at these settings.
        baseline = {"key": "swin", "kind": "classification", "uncertain": [], "note": "",
                    "inferred": [{"field": "pretrained_backbone", "source": "Table 1, Swin-T (ImageNet-1k) row"}],
                    "fields": [{"name": n, "value": v} for n, v in
                               [("pretrained_backbone", "Swin-T-IN1k"), ("dataset", "ImageNet-1k"), ("resolution", 256),
                                ("top_1", 83.1), ("gflops", 4.5)]],
                    "evidence": [{"field": f, "citation": {**citation(SWIN), "url": None}, "derivation": None}
                                 for f in ("dataset", "resolution", "top_1", "gflops")]}
        folder = WorkFolder(extraction([baseline]))
        agent.write_json(folder.path / "pages.json", [TEXT + SWIN])
        report = agent.validate(folder.path)
        self.assertTrue(report["publishable"], report)

    def test_orchestrator_confines_claude_code(self):
        events = [json.dumps({"type": "assistant"}) + "\n",
                  json.dumps({"type": "result", "is_error": False, "result": "Done", "total_cost_usd": 2.5,
                              "num_turns": 12, "usage": {"input_tokens": 10, "output_tokens": 20}}) + "\n"]
        process = MagicMock(stdout=iter(events))
        folder = WorkFolder(extraction())
        (folder.path / "prompt.md").write_text("Extract.")
        with patch("ingestion.agent.subprocess.Popen", return_value=process) as popen, \
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "ANTHROPIC_AUTH_TOKEN": "t"}):
            result = agent.run_agent(folder.path, model="claude-opus-5-5")
        command, kwargs = popen.call_args.args[0], popen.call_args.kwargs
        self.assertEqual((result["cost_usd"], result["turns"], result["is_error"]), (2.5, 12, False))
        self.assertEqual(kwargs["cwd"], folder.path)
        # Sessions use the claude.ai login, never an API key.
        self.assertFalse({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(kwargs["env"]))
        self.assertIn("dontAsk", command)
        self.assertIn("Bash(./hb validate *)", command)
        self.assertIn("WebFetch", command[command.index("--disallowedTools"):])
        settings = json.loads((folder.path / ".claude" / "settings.json").read_text())
        self.assertIn("Write(./**)", settings["permissions"]["allow"])

    def test_agent_runner_reads_a_shortlisted_run(self):
        paper = PaperVersion.objects.create(arxiv_id="2609.12345", revision="2026-09-01T00:00:00Z", title="FixtureNet",
                                            metadata={"created": "2026-09-01T00:00:00Z"})
        run = IngestionRun.objects.create(paper=paper, status="shortlisted", provider="anthropic", decision=decision())
        folder = WorkFolder(extraction())
        with patch("ingestion.agent.prepare", return_value=folder.path), \
                patch("ingestion.agent.run_agent", return_value={"model": "claude-opus-5-5", "cost_usd": 1.0}), \
                patch.object(agent.AgentRunner, "folder_for", return_value=folder.path):
            result = agent.AgentRunner(publish=True).read(run)
        self.assertEqual(result.status, "imported", result.error)
        self.assertEqual(result.provider, "claude-code")
        self.assertTrue(BackboneFamily.objects.filter(name="FixtureNet").exists())

    def test_lookup_is_read_only_and_hides_eval_families(self):
        out = StringIO()
        call_command("lookup", "model", "Swin-T-IN1k", stdout=out)
        self.assertIn("classification", out.getvalue())
        work = Path(tempfile.mkdtemp())
        agent.write_json(work / "hidden.json", ["Swin"])
        with self.assertRaises(CommandError):
            call_command("lookup", "family", "Swin", work=str(work), stdout=StringIO())
        out = StringIO()
        call_command("lookup", "names", work=str(work), stdout=out)
        self.assertNotIn('"Swin"', out.getvalue())


class WebSourceTests(SimpleTestCase):
    def test_only_sources_linked_from_the_paper(self):
        pages = ["Code: https://github.com/fixture/fix-\nturenet and project page https://fixture.github.io/net", "", "refs"]
        repos, hosts = official_links(pages, metadata={"comment": "Code at github.com/fixture/other"})
        self.assertTrue({("fixture", "fixturenet"), ("fixture", "fix-turenet"), ("fixture", "other")} <= repos)
        check_allowed("https://github.com/fixture/fixturenet/releases", repos, hosts)
        check_allowed("https://raw.githubusercontent.com/fixture/fixturenet/main/MODEL_ZOO.md", repos, hosts)
        check_allowed("https://fixture.github.io/net/results.html", repos, hosts)
        for url in ("https://github.com/someone/else", "https://example.com/results", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                check_allowed(url, repos, hosts)

    def test_github_pages_are_read_as_text(self):
        session = Mock()
        response = MagicMock()
        response.__enter__.return_value = Mock(iter_content=Mock(return_value=[b"# FixtureNet\n| T | 85.2 |"]),
                                               headers={"content-type": "text/plain"}, raise_for_status=Mock())
        session.get.return_value = response
        source, text, digest = fetch("https://github.com/fixture/fixturenet", session)
        self.assertEqual(source, "https://api.github.com/repos/fixture/fixturenet/readme")
        self.assertIn("85.2", text)
        source, _, _ = fetch("https://github.com/fixture/fixturenet/blob/main/docs/zoo.md", session)
        self.assertEqual(source, "https://raw.githubusercontent.com/fixture/fixturenet/main/docs/zoo.md")
        self.assertEqual(html_text("<html><style>x</style><table><tr><td>T</td><td>85.2</td></tr></table></html>"),
                         "| T | 85.2")

    def test_url_citations_are_checked_against_the_saved_text(self):
        web = {"url": "https://github.com/fixture/fixturenet", "page": None, "location": "README", "quote": "| FixtureNet-T | 85.2 |"}
        check_citation(web, [], {web["url"]: README})
        with self.assertRaises(ValueError):
            check_citation({**web, "quote": "| FixtureNet-T | 99.9 |"}, [], {web["url"]: README})
        with self.assertRaises(ValueError):
            check_citation(web, [], {})

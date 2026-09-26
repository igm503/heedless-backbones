import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.utils import timezone

from ingestion import publication
from ingestion.importer import apply_run
from ingestion.models import IngestionRun
from ingestion.tests import test_importer
from stats.models import BackboneFamily

README = """# Heedless Backbones

## Models

| Model | Paper | Added |
|---|---|---|
| ConvNeXt | [arXiv 2201.03545](https://arxiv.org/abs/2201.03545) | 2024-09-03 |

## Updates

- 11-8-2025: added CoCAViT
"""

ABOUT = """<div class="drawer">Latest Updates</div>
      <div class="drawerContents show" style="background-color: #f8f9fa; margin-bottom: 2rem;">
        <!-- November 8, 2025 -->
        <div style="margin-bottom: 1rem;">
          <p style="margin-bottom: 0.3rem; font-weight: 600;">November 8, 2025</p>
          <ul class="list-disc" style="padding-left: 1.5rem; margin-top: 0; margin-bottom: 0.5rem;">
            <li style="margin-bottom: 0.3rem;">Added <a href="{% url "family" "CoCAViT" %}">CoCAViT</a></li>
          </ul>
        </div>
      </div>
"""


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


class FakeGitHub:
    """Stands in for the gh CLI: remembers pull requests by branch."""

    def __init__(self):
        self.pulls, self.titles, self.edits = {}, {}, []

    def __call__(self, repo, *args):
        if args[:2] == ("pr", "list"):
            return json.dumps([{"number": number, "headRefName": branch, "url": f"https://github.test/pull/{number}"}
                               for branch, number in self.pulls.items()])
        if args[:2] == ("pr", "create"):
            branch = args[args.index("--head") + 1]
            self.pulls[branch] = len(self.pulls) + 1
            self.titles[branch] = args[args.index("--title") + 1]
            return f"https://github.test/pull/{self.pulls[branch]}"
        if args[:2] == ("pr", "edit"):
            self.edits.append(args)
            return ""
        raise AssertionError(args)


class PublicationTests(TestCase):
    fixtures = test_importer.ImportTests.fixtures
    add = test_importer.ImportTests.add

    def setUp(self):
        test_importer.ImportTests.setUp(self)
        root = Path(tempfile.mkdtemp())
        self.origin, seed, self.clone = root / "origin.git", root / "seed", root / "records"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        subprocess.run(["git", "clone", "-q", str(self.origin), str(seed)], check=True, capture_output=True)
        git(seed, "config", "user.email", "test@example.com")
        git(seed, "config", "user.name", "Test")
        (seed / "family_data").mkdir()
        (seed / "family_data" / "ConvNeXt.yml").write_text("name: ConvNeXt\n")
        (seed / "README.md").write_text(README)
        (seed / publication.ABOUT).parent.mkdir(parents=True)
        (seed / publication.ABOUT).write_text(ABOUT)
        (seed / "db.json").write_text("[]\n")
        git(seed, "add", ".")
        git(seed, "commit", "-q", "-m", "seed")
        git(seed, "push", "-q", "origin", "main")
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.clone)], check=True, capture_output=True)
        git(self.clone, "config", "user.email", "agent@example.com")
        git(self.clone, "config", "user.name", "Agent")
        self.github = FakeGitHub()
        patches = [override_settings(RECORDS_REPO=str(self.clone), MEDIA_ROOT=tempfile.mkdtemp()),
                   patch.object(publication.RecordsRepo, "gh", lambda repo, *args: self.github(repo, *args))]
        for item in patches:
            item.enable() if hasattr(item, "enable") else item.start()
            self.addCleanup(item.disable if hasattr(item, "disable") else item.stop)

    def published_run(self):
        self.assertEqual(apply_run(self.run, publish=True), [])
        return self.run

    def branch_file(self, branch, path):
        return git(self.origin, "show", f"{branch}:{path}")

    def test_publish_records_the_family_on_its_own_branch_and_pull_request(self):
        outcomes = publication.record([self.published_run()])
        self.assertEqual(outcomes[0]["branch"], "auto.fixturenet")
        self.assertEqual(outcomes[0]["url"], "https://github.test/pull/1")
        self.assertEqual(self.github.titles["auto.fixturenet"],
                         "Adds FixtureNet (Convolution, hierarchical), arXiv 2609.12345.")
        self.assertEqual(git(self.origin, "log", "-1", "--format=%s", "auto.fixturenet"),
                         self.github.titles["auto.fixturenet"])
        self.assertIn("name: FixtureNet", self.branch_file("auto.fixturenet", "family_data/FixtureNet.yml"))
        dump = json.loads(self.branch_file("auto.fixturenet", "db.json"))
        names = {o["fields"]["name"] for o in dump if o["model"] == "stats.backbonefamily"}
        self.assertEqual(names, {"ConvNeXt", "FixtureNet"})  # the families in the branch
        self.assertTrue(all(o["fields"].get("source_record") is None for o in dump if "source_record" in o["fields"]))
        readme = self.branch_file("auto.fixturenet", "README.md")
        self.assertIn("| FixtureNet | [arXiv 2609.12345](https://arxiv.org/abs/2609.12345) |", readme)
        self.assertIn(": added FixtureNet", readme)
        about = self.branch_file("auto.fixturenet", str(publication.ABOUT))
        today = timezone.now().date()
        self.assertEqual(about.count('Added <a href="{% url "family" "FixtureNet" %}">FixtureNet</a>'), 1)
        self.assertLess(about.index(f"{today:%B} {today.day}, {today.year}"), about.index("November 8, 2025"))
        self.run.refresh_from_db()
        self.assertEqual(self.run.calls[-1]["url"], "https://github.test/pull/1")
        # Recording again changes nothing and opens no second pull request.
        self.assertTrue(publication.record([self.run])[0]["unchanged"])
        self.assertEqual(len(self.github.pulls), 1)
        # A branch with an older title is rewritten and its pull request retitled.
        git(self.clone, "checkout", "-q", "auto.fixturenet")
        git(self.clone, "commit", "-q", "--amend", "-m", "Add FixtureNet")
        git(self.clone, "push", "-q", "--force", "origin", "auto.fixturenet")
        self.assertNotIn("unchanged", publication.record([self.run])[0])
        self.assertEqual(git(self.origin, "log", "-1", "--format=%s", "auto.fixturenet"), self.github.titles["auto.fixturenet"])
        self.assertIn("--title", self.github.edits[-1])

    def test_open_branches_stay_mergeable_after_one_is_merged(self):
        publication.record([self.published_run()])
        # A second family from a second run.
        second = test_importer.PaperVersion.objects.create(arxiv_id="2609.54321", revision="r", title="Other",
                                                           pages=[test_importer.TEXT], metadata={"created": "2026-09-02T00:00:00Z"})
        self.run = IngestionRun.objects.create(paper=second, provider="claude-code", decision=test_importer.decision())
        self.add("family", "family", name="OtherNet", model_type="Convolution", hierarchical=True, pretrain_method="Supervised")
        self.add("backbone", "backbone", name="OtherNet-T", family="$family", m_parameters=28)
        self.add("pretrain", "pretrained_backbone", name="OtherNet-T-IN1k", family="$family", backbone="$backbone",
                 pretrain_dataset="ImageNet-1k", pretrain_method="Supervised", pretrain_resolution=224, pretrain_epochs=300)
        self.add("classification", "classification", pretrained_backbone="$pretrain", dataset="ImageNet-1k",
                 resolution=224, top_1=83.1, gflops=4.5)
        publication.record([self.published_run()])
        # Merge the first pull request on "GitHub".
        git(self.origin, "update-ref", "refs/heads/main", git(self.origin, "rev-parse", "auto.fixturenet"))
        del self.github.pulls["auto.fixturenet"]
        publication.refresh()
        base = git(self.origin, "rev-parse", "main")
        # The other branch now sits directly on the new main (a clean merge) and lists both families.
        self.assertEqual(git(self.origin, "merge-base", "main", "auto.othernet"), base)
        readme = self.branch_file("auto.othernet", "README.md")
        self.assertIn("| FixtureNet |", readme)
        self.assertIn("| OtherNet |", readme)
        about = self.branch_file("auto.othernet", str(publication.ABOUT))
        today = timezone.now().date()
        self.assertEqual(about.count(f">{today:%B} {today.day}, {today.year}</p>"), 1)  # one group for the day
        self.assertIn('"OtherNet" %}">OtherNet</a>', about)
        self.assertIn('"FixtureNet" %}">FixtureNet</a>', about)

    def test_publish_imports_then_records_in_the_background(self):
        with patch("ingestion.publication.subprocess.Popen") as popen:
            self.assertEqual(publication.publish(self.run, "reviewer"), [])
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "imported")
        command = popen.call_args.args[0]
        self.assertEqual(command[2:4], ["record_publication", str(self.run.pk)])

    def test_approval_and_the_agent_publish_through_the_same_function(self):
        from ingestion import review
        with patch("ingestion.publication.publish", return_value=[]) as publish:
            review.approve(self.run, "reviewer", "Checked")
        publish.assert_called_once_with(self.run, "reviewer", allow_updates=False)
        self.run.status = "ready"
        self.run.save()
        with patch("ingestion.publication.record") as record:
            published, blocked = publication.publish_ready()
        self.assertEqual((published, blocked), ([self.run], []))
        record.assert_called_once_with([self.run], refresh_others=True)

    def test_dump_keeps_unattached_measurements_of_the_families(self):
        from stats.models import FPSMeasurement
        FPSMeasurement.objects.create(backbone_name="Swin-T", resolution=224, fps=1.0, gpu="V100")
        FPSMeasurement.objects.create(backbone_name="Unrelated-X", resolution=224, fps=2.0, gpu="V100")
        dump = json.loads(publication.dump_database({"Swin"}))
        names = {o["fields"]["backbone_name"] for o in dump if o["model"] == "stats.fpsmeasurement"}
        self.assertIn("Swin-T", names)
        self.assertNotIn("Unrelated-X", names)

    def test_readme_rows_are_added_once(self):
        self.published_run()
        once = publication.update_readme(README, {"ConvNeXt", "FixtureNet"})
        self.assertEqual(once.count("| FixtureNet |"), 1)
        self.assertEqual(publication.update_readme(once, {"ConvNeXt", "FixtureNet"}), once)

    def test_unconfigured_record_keeping_is_noted_on_the_run(self):
        with override_settings(RECORDS_REPO=None):
            publication.record([self.published_run()])
        self.run.refresh_from_db()
        self.assertIn("not configured", self.run.calls[-1]["error"])
        self.assertTrue(BackboneFamily.objects.filter(name="FixtureNet").exists())  # the publish itself stands

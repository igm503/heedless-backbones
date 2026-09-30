import json
import subprocess
import tempfile
from pathlib import Path
from datetime import date
from unittest.mock import Mock, patch

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
        self.pulls, self.titles, self.bodies, self.edits, self.labels = {}, {}, {}, [], set()

    def __call__(self, repo, *args):
        if args[:2] == ("pr", "list"):
            return json.dumps([{"number": number, "headRefName": branch, "url": f"https://github.test/pull/{number}"}
                               for branch, number in self.pulls.items()])
        if args[:2] == ("pr", "create"):
            branch = args[args.index("--head") + 1]
            self.pulls[branch] = len(self.pulls) + 1
            self.titles[branch] = args[args.index("--title") + 1]
            self.bodies[branch] = args[args.index("--body") + 1]
            assert args[args.index("--label") + 1] in self.labels
            return f"https://github.test/pull/{self.pulls[branch]}"
        if args[:2] == ("label", "create"):
            self.labels.add(args[2])
            return ""
        if args[:2] == ("pr", "edit"):
            self.edits.append(args)
            branch = next(branch for branch, number in self.pulls.items() if str(number) == args[2])
            self.titles[branch] = args[args.index("--title") + 1]
            self.bodies[branch] = args[args.index("--body") + 1]
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
        self.seed = seed
        git(self.origin, "config", "receive.denyNonFastforwards", "true")
        git(seed, "config", "user.email", "test@example.com")
        git(seed, "config", "user.name", "Test")
        git(seed, "config", "commit.gpgsign", "false")
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
        git(self.clone, "config", "commit.gpgsign", "false")
        self.github = FakeGitHub()
        patches = [override_settings(RECORDS_REPO=str(self.clone), MEDIA_ROOT=tempfile.mkdtemp()),
                   patch.object(publication.RecordsRepo, "gh", lambda repo, *args: self.github(repo, *args))]
        for item in patches:
            item.enable() if hasattr(item, "enable") else item.start()
            self.addCleanup(item.disable if hasattr(item, "disable") else item.stop)

    @property
    def batch(self):
        return next(branch for branch in self.github.pulls if branch.startswith(publication.BATCH_PREFIX))

    def published_run(self):
        self.assertEqual(apply_run(self.run, publish=True), [])
        return self.run

    def branch_file(self, branch, path):
        return git(self.origin, "show", f"{branch}:{path}")

    def test_publish_records_the_family_in_one_batch_and_repeat_is_idempotent(self):
        outcomes = publication.record([self.published_run()])
        self.assertEqual(outcomes[0]["branch"], self.batch)
        self.assertEqual(outcomes[0]["url"], "https://github.test/pull/1")
        self.assertEqual(self.github.titles[self.batch],
                         "Add FixtureNet")
        self.assertEqual(git(self.origin, "log", "-1", "--format=%s", self.batch),
                         self.github.titles[self.batch])
        self.assertIn("name: FixtureNet", self.branch_file(self.batch, "family_data/FixtureNet.yml"))
        dump = json.loads(self.branch_file(self.batch, "db.json"))
        names = {o["fields"]["name"] for o in dump if o["model"] == "stats.backbonefamily"}
        self.assertEqual(names, {"ConvNeXt", "FixtureNet"})  # the families in the branch
        self.assertTrue(all(o["fields"].get("source_record") is None for o in dump if "source_record" in o["fields"]))
        readme = self.branch_file(self.batch, "README.md")
        self.assertIn("| FixtureNet | 1 | [arXiv 2609.12345](https://arxiv.org/abs/2609.12345) |", readme)
        self.assertIn(": added FixtureNet", readme)
        about = self.branch_file(self.batch, str(publication.ABOUT))
        today = timezone.now().date()
        self.assertEqual(about.count('Added <a href="{% url "family" "FixtureNet" %}">FixtureNet</a>'), 1)
        self.assertLess(about.index(f"{today:%B} {today.day}, {today.year}"), about.index("November 8, 2025"))
        self.run.refresh_from_db()
        self.assertEqual(self.run.calls[-1]["url"], "https://github.test/pull/1")
        # Recording again changes nothing and opens no second pull request.
        self.assertTrue(publication.record([self.run])[0]["unchanged"])
        self.assertEqual(len(self.github.pulls), 1)
        previous = git(self.origin, "rev-parse", self.batch)
        self.github.titles[self.batch] = "Old title"
        self.assertTrue(publication.record([self.run])[0]["unchanged"])
        self.assertEqual(git(self.origin, "rev-parse", self.batch), previous)
        self.assertEqual(self.github.titles[self.batch], "Add FixtureNet")

    def test_records_are_pushed_and_proposed_as_the_github_app(self):
        bot = ("heedless-backbones-agent[bot]", "42+heedless-backbones-agent[bot]@users.noreply.github.com")
        app = Mock(token=Mock(return_value="installation-token"), identity=Mock(return_value=bot))
        seen = []
        real_run = subprocess.run
        def run(command, *args, **kwargs):
            seen.append((command, kwargs.get("env") or {}))
            return real_run(command, *args, **kwargs)
        with patch("ingestion.publication.github_app", return_value=app), \
                patch("ingestion.publication.subprocess.run", side_effect=run):
            publication.record([self.published_run()])
        author = git(self.origin, "log", "-1", "--format=%an <%ae>", self.batch)
        self.assertEqual(author, f"{bot[0]} <{bot[1]}>")
        self.assertTrue(all(env.get("GH_TOKEN") == "installation-token" for command, env in seen if command[0] == "git"))
        body = self.github.bodies[self.batch]
        self.assertIn(publication.FOOTER, body)
        self.assertEqual(self.github.labels, {publication.LABEL})

    def test_app_token_is_signed_by_the_app_key(self):
        import jwt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        path = Path(tempfile.mkdtemp()) / "app.pem"
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        responses = {"/repos/igm503/heedless-backbones/installation": {"id": 7},
                     "/app/installations/7/access_tokens": {"token": "ghs_x"},
                     "/app": {"slug": "heedless-backbones-agent"},
                     "/users/heedless-backbones-agent[bot]": {"id": 42}}
        calls = []
        def request(method, url, headers, timeout):
            path_ = url.removeprefix(publication.GITHUB_API)
            calls.append((method, path_, headers.get("Authorization")))
            return Mock(json=Mock(return_value=responses[path_]), raise_for_status=Mock())
        app = publication.GitHubApp(123, path, "igm503/heedless-backbones", session=Mock(request=request))
        self.assertEqual(app.token(), "ghs_x")
        self.assertEqual(app.token(), "ghs_x")  # cached for the hour
        self.assertEqual(app.identity(), ("heedless-backbones-agent[bot]",
                                          "42+heedless-backbones-agent[bot]@users.noreply.github.com"))
        claims = jwt.decode(calls[0][2].removeprefix("Bearer "), key.public_key(), algorithms=["RS256"])
        self.assertEqual(claims["iss"], "123")
        self.assertEqual([c[:2] for c in calls][:2], [("GET", "/repos/igm503/heedless-backbones/installation"),
                                                     ("POST", "/app/installations/7/access_tokens")])
        self.assertIsNone(calls[-1][2])  # the bot's public user record needs no auth

    def test_second_family_appends_to_the_same_pr_without_rewriting_history(self):
        publication.record([self.published_run()])
        branch = self.batch
        first = git(self.origin, "rev-parse", branch)
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
        outcome = publication.record([self.published_run()])[0]
        self.assertEqual(outcome["branch"], branch)
        self.assertEqual(len(self.github.pulls), 1)
        self.assertEqual(git(self.origin, "merge-base", first, branch), first)
        self.assertEqual(self.github.titles[branch], "Add FixtureNet and OtherNet")
        self.assertEqual(git(self.origin, "log", "-1", "--format=%s", branch), "Add OtherNet")
        dump = json.loads(self.branch_file(branch, "db.json"))
        self.assertEqual({o["fields"]["name"] for o in dump if o["model"] == "stats.backbonefamily"},
                         {"ConvNeXt", "FixtureNet", "OtherNet"})
        about = self.branch_file(branch, str(publication.ABOUT))
        today = timezone.now().date()
        self.assertEqual(about.count(f">{today:%B} {today.day}, {today.year}</p>"), 1)
        self.assertIn('"OtherNet" %}">OtherNet</a>', about)
        self.assertIn('"FixtureNet" %}">FixtureNet</a>', about)
        self.assertIn("2609.54321", self.github.bodies[branch])
        self.assertIn("2609.12345", self.github.bodies[branch])

    def test_mixed_batch_counts_each_family_once_relative_to_main(self):
        publication.record([self.published_run()])
        branch = self.batch
        family = BackboneFamily.objects.get(name="FixtureNet")
        family.github = "https://github.com/example/fixture"
        family.save()
        outcome = publication.record([self.run], families=["ConvNeXt", "FixtureNet", "FixtureNet"])[0]
        self.assertNotIn("error", outcome)
        self.assertEqual(self.github.titles[branch], "Add 1 backbone family; update 1")
        self.assertEqual(self.github.bodies[branch].count("| Add | FixtureNet |"), 1)
        self.assertEqual(self.github.bodies[branch].count("| Update | ConvNeXt |"), 1)
        self.assertEqual(outcome["families"], ["ConvNeXt", "FixtureNet"])
        self.assertTrue(publication.refresh()[0]["unchanged"])

    def test_main_advances_with_shared_file_conflicts_using_a_normal_merge(self):
        publication.record([self.published_run()])
        branch = self.batch
        previous = git(self.origin, "rev-parse", branch)
        (self.seed / "db.json").write_text('[{"changed": true}]\n')
        (self.seed / "README.md").write_text(README + "\nAn unrelated documentation improvement.\n")
        (self.seed / "unrelated.txt").write_text("keep me\n")
        git(self.seed, "add", ".")
        git(self.seed, "commit", "-qm", "Change main")
        git(self.seed, "push", "-q", "origin", "main")
        outcome = publication.refresh()[0]
        self.assertNotIn("error", outcome)
        self.assertEqual(git(self.origin, "merge-base", previous, branch), previous)
        self.assertEqual(git(self.origin, "merge-base", "main", branch), git(self.origin, "rev-parse", "main"))
        self.assertEqual(self.branch_file(branch, "unrelated.txt"), "keep me")
        self.assertIn("An unrelated documentation improvement.", self.branch_file(branch, "README.md"))
        self.assertIn("| FixtureNet |", self.branch_file(branch, "README.md"))
        self.assertEqual(self.github.titles[branch], "Add FixtureNet")
        self.assertTrue(publication.refresh()[0]["unchanged"])

    def test_unmanaged_conflict_is_reported_and_merge_is_aborted(self):
        publication.record([self.published_run()])
        branch = self.batch
        git(self.clone, "checkout", "-q", branch)
        (self.clone / "custom.txt").write_text("PR edit\n")
        git(self.clone, "add", "custom.txt")
        git(self.clone, "commit", "-qm", "Custom edit")
        git(self.clone, "push", "-q", "origin", branch)
        previous = git(self.origin, "rev-parse", branch)
        (self.seed / "custom.txt").write_text("main edit\n")
        git(self.seed, "add", "custom.txt")
        git(self.seed, "commit", "-qm", "Conflicting edit")
        git(self.seed, "push", "-q", "origin", "main")
        self.assertIn("error", publication.refresh()[0])
        self.assertEqual(git(self.origin, "rev-parse", branch), previous)
        self.assertEqual(git(self.clone, "status", "--porcelain"), "")

    def test_squash_merge_finishes_batch_and_next_publication_gets_fresh_branch(self):
        publication.record([self.published_run()])
        old = self.batch
        git(self.seed, "fetch", "-q", "origin")
        git(self.seed, "merge", "--squash", f"origin/{old}")
        git(self.seed, "commit", "-qm", "Squash batch")
        git(self.seed, "push", "-q", "origin", "main")
        del self.github.pulls[old]
        self.assertIn("skipped", publication.refresh()[0])
        self.assertIn("skipped", publication.record([self.run])[0])
        self.assertEqual(self.github.pulls, {})
        family = BackboneFamily.objects.get(name="FixtureNet")
        family.github = "https://github.com/example/fixturenet"
        family.save()
        outcome = publication.record([self.run])[0]
        self.assertNotIn("error", outcome)
        self.assertNotEqual(self.batch, old)
        self.assertEqual(self.github.titles[self.batch], "Update FixtureNet")
        self.assertEqual(git(self.origin, "merge-base", "main", self.batch), git(self.origin, "rev-parse", "main"))

    def test_legacy_consolidation_is_explicit_and_leaves_old_prs_for_review(self):
        self.published_run()
        self.github.pulls["auto.fixturenet"] = 17
        self.assertIn("skipped", publication.refresh()[0])
        outcome = publication.refresh(include_legacy=True)[0]
        self.assertNotIn("error", outcome)
        self.assertIn("auto.fixturenet", self.github.pulls)
        self.assertIn("FixtureNet", outcome["families"])
        self.assertEqual(self.github.titles[self.batch], "Add FixtureNet")

    def test_missing_pending_family_does_not_disappear_silently(self):
        publication.record([self.published_run()])
        previous = git(self.origin, "rev-parse", self.batch)
        BackboneFamily.objects.filter(name="FixtureNet").delete()
        self.assertIn("missing from the database", publication.refresh()[0]["error"])
        self.assertEqual(git(self.origin, "rev-parse", self.batch), previous)

    def test_batch_titles(self):
        for added, updated, expected in [
            (["Swin"], [], "Add Swin"),
            (["Swin", "ConvNeXt"], [], "Add ConvNeXt and Swin"),
            (["Swin", "ConvNeXt", "LocalViT"], [], "Add ConvNeXt, LocalViT, and Swin"),
            (["A", "B", "C", "D"], [], "Add 4 backbone families"),
            (["Swin", "Swin"], ["ConvNeXt"], "Add 1 backbone family; update 1"),
            ([], ["Swin"], "Update Swin"),
            ([], ["A", "B", "C", "D"], "Update 4 backbone families"),
        ]:
            with self.subTest(expected=expected):
                self.assertEqual(publication.batch_title(added, updated), expected)

    def test_about_sorts_old_and_new_dates_and_preserves_handwritten_updates(self):
        family = BackboneFamily.objects.get(name="Swin")
        source = ABOUT.replace("          </ul>", "            <li>Improved search performance.</li>\n          </ul>")
        text = publication.update_about(source, [(date(2025, 1, 1), family)])
        self.assertLess(text.index("November 8, 2025"), text.index("January 1, 2025"))
        self.assertEqual(publication.update_about(text, [(date(2025, 1, 1), family)]), text)
        # Repair an already out-of-order section, including updates unrelated to families.
        early = text.index("        <!-- November")
        late = text.index("        <!-- January")
        end = text.index("\n      </div>", late) + 1
        shuffled = text[:early] + text[late:end] + text[early:late] + text[end:]
        self.assertEqual(publication.update_about(shuffled, []), text)
        self.assertIn('"CoCAViT" %}', text)
        self.assertIn("<li>Improved search performance.</li>", text)

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
        record.assert_called_once_with([self.run])

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

    def test_readme_orders_existing_rows_and_combines_additions_by_date(self):
        text = README.replace(
            "## Updates",
            "| Zebra | [paper](https://example.com/z) | 2026-09-29 |\n"
            "| Alpha | [paper](https://example.com/a) | 2026-09-29 |\n\n## Updates",
        ).replace("\n\n| Zebra", "\n| Zebra")
        text += ("- 9-28-2026: added Middle\n"
                 "- 9-29-2026: added Zebra\n"
                 "- 9-28-2026: improved plot defaults; bug fixes\n"
                 "- 9-29-2026: added Alpha, Zebra\n"
                 "\n## Other notes\n\n- 1-1-2030: Leave this section alone\n")
        # No new family is required to repair historical ordering.
        result = publication.update_readme(text, {"ConvNeXt"})
        self.assertLess(result.index("| Alpha |"), result.index("| Zebra |"))
        self.assertLess(result.index("| Zebra |"), result.index("| ConvNeXt |"))
        self.assertEqual(result.count("- 9-29-2026:"), 1)
        self.assertIn("- 9-29-2026: added Alpha, Zebra", result)
        self.assertLess(result.index("- 9-29-2026:"), result.index("- 9-28-2026:"))
        self.assertLess(result.index("- 9-28-2026:"), result.index("- 11-8-2025:"))
        self.assertIn("- 9-28-2026: improved plot defaults; bug fixes", result)
        self.assertTrue(result.endswith("## Other notes\n\n- 1-1-2030: Leave this section alone\n"))
        self.assertEqual(publication.update_readme(result, {"ConvNeXt"}), result)

    def test_readme_places_late_arriving_older_additions_by_date(self):
        self.published_run()
        with patch.object(publication, "added_on", return_value=date(2025, 1, 1)):
            result = publication.update_readme(README, {"ConvNeXt", "FixtureNet"})
        self.assertLess(result.index("| FixtureNet |"), result.index("| ConvNeXt |"))
        self.assertLess(result.index("- 11-8-2025:"), result.index("- 1-1-2025:"))

    def test_model_counts_use_backbone_variants_and_refresh_existing_rows(self):
        self.published_run()
        records = json.loads(publication.dump_database({"ConvNeXt", "FixtureNet"}))
        family_id = next(obj["pk"] for obj in records if obj["model"] == "stats.backbonefamily"
                         and obj["fields"]["name"] == "FixtureNet")
        backbone = next(obj for obj in records if obj["model"] == "stats.backbone"
                        and obj["fields"]["family"] == family_id)
        checkpoint = next(obj for obj in records if obj["model"] == "stats.pretrainedbackbone"
                          and obj["fields"]["family"] == family_id)
        records.append({**checkpoint, "pk": 999999})  # another checkpoint of the same variant
        once = publication.update_readme(README, {"ConvNeXt", "FixtureNet"}, records=records)
        self.assertIn("| FixtureNet | 1 |", once)
        self.assertIn(publication.TABLE_HEADER, once)
        dates = publication.listed_dates(once)
        records.append({**backbone, "pk": 999999, "fields": {**backbone["fields"], "name": "FixtureNet-L"}})
        updated = publication.update_readme(once, {"ConvNeXt", "FixtureNet"}, records=records)
        self.assertIn("| FixtureNet | 2 |", updated)
        self.assertEqual(publication.listed_dates(updated), dates)
        self.assertEqual(updated.count(": added FixtureNet"), 1)
        self.assertEqual(publication.update_readme(updated, {"ConvNeXt", "FixtureNet"}, records=records), updated)

    def test_historical_paper_row_counts_distinct_variants_for_that_paper(self):
        records = json.loads(publication.dump_database({"FAN"}))
        text = "\n".join([publication.LEGACY_TABLE_HEADER, "|---|---|---|",
                           "| FAN | [paper](https://arxiv.org/abs/2204.12451) | 2025-04-14 |",
                           "| FAN STL | [paper](https://arxiv.org/pdf/2401.03844) | 2025-04-22 |",
                           "| Unknown | [paper](https://example.com/unknown) | 2025-04-23 |"])
        result = publication.update_model_counts(text, records)
        self.assertIn("| FAN | 8 |", result)
        self.assertIn("| FAN STL | 4 |", result)
        self.assertIn("| Unknown | — |", result)
        self.assertEqual(publication.update_model_counts(result, records), result)

    def test_fallback_added_date_is_preserved_while_batch_stays_open(self):
        self.published_run()
        with patch.object(publication, "added_on", return_value=None), \
                patch.object(publication.timezone, "now", return_value=timezone.datetime(2026, 9, 25, tzinfo=timezone.get_current_timezone())):
            publication.record([self.run])
        branch = self.batch
        previous = git(self.origin, "rev-parse", branch)
        with patch.object(publication, "added_on", return_value=None), \
                patch.object(publication.timezone, "now", return_value=timezone.datetime(2026, 9, 30, tzinfo=timezone.get_current_timezone())):
            outcome = publication.refresh()[0]
        self.assertTrue(outcome["unchanged"])
        self.assertEqual(git(self.origin, "rev-parse", branch), previous)
        self.assertIn("2026-09-25", self.branch_file(branch, "README.md"))
        self.assertIn("September 25, 2026", self.branch_file(branch, str(publication.ABOUT)))

    def test_unconfigured_record_keeping_is_noted_on_the_run(self):
        with override_settings(RECORDS_REPO=None):
            publication.record([self.published_run()])
        self.run.refresh_from_db()
        self.assertIn("not configured", self.run.calls[-1]["error"])
        self.assertTrue(BackboneFamily.objects.filter(name="FixtureNet").exists())  # the publish itself stands

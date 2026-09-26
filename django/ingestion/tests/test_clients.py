from unittest.mock import Mock

from django.test import SimpleTestCase

from ingestion.sources import Troller, arxiv_id


def selection():
    return {"qualifies": False, "architecture_or_pretraining_contribution": False,
            "imagenet_1k_results": False, "downstream_results": False,
            "reason": "Unrelated paper", "evidence": []}


class SourceClientTests(SimpleTestCase):
    def test_id_parsing(self):
        for source, expected in [
            ("https://arxiv.org/pdf/2201.03545v2", "2201.03545"),
            ("https://arxiv.org/html/2507.00698v3", "2507.00698"),
            ("https://arxiv.org/abs/cs/9901001", "cs/9901001"),
            ("https://doi.org/10.48550/arXiv.2311.17132", "2311.17132"),
            ("https://doi.org/10.1109/CVPR.2022.01167", None),
            ("https://example.com/2201.03545", None),
            ("https://github.com/test/repo", None),
        ]:
            self.assertEqual(arxiv_id(source), expected)

    def test_pagination_consumes_all_pages(self):
        client = Troller("https://example.com", "test")
        client.request = Mock(side_effect=[{"papers": [{"arxiv_id": "1"}], "next_cursor": 100},
                                           {"papers": [{"arxiv_id": "2"}], "next_cursor": None}])
        self.assertEqual(len(list(client.papers("tag", tag="backbones"))), 2)
        self.assertNotIn("cursor", client.request.call_args_list[0].kwargs)
        self.assertEqual(client.request.call_args.kwargs["cursor"], 100)

    def test_joint_tag_search_passes_opaque_cursors(self):
        client = Troller("https://example.com", "test")
        client.request = Mock(side_effect=[{"papers": [{"arxiv_id": "1"}], "next_cursor": "MSwy"},
                                           {"papers": [{"arxiv_id": "2"}], "next_cursor": None}])
        self.assertEqual(len(list(client.tag_search("working", "2026-09-01T00:00:00+00:00"))), 2)
        self.assertEqual(client.request.call_args.args, ("search",))
        self.assertEqual(client.request.call_args.kwargs, {"type": "tag", "tag": "working",
                                                          "since": "2026-09-01T00:00:00+00:00", "cursor": "MSwy"})

    def test_search_stops_before_the_cursor_is_too_long_for_a_url(self):
        client = Troller("https://example.com", "test")
        client.request = Mock(side_effect=[{"papers": [{"arxiv_id": "1"}], "next_cursor": "x" * 5000}])
        self.assertEqual(len(list(client.tag_search("working", "2026-09-01T00:00:00+00:00"))), 1)

    def test_pagination_loop_is_reported(self):
        client = Troller("https://example.com", "test")
        client.request = Mock(return_value={"papers": [], "next_cursor": "same"})
        with self.assertRaisesRegex(ValueError, "did not advance"):
            list(client.papers("tag", tag="backbones"))

    def test_login_and_mutations_send_csrf_token(self):
        session = Mock()
        session.cookies.get.return_value = "csrf-token"
        session.get.return_value.json.return_value = {"ok": True, "tags": []}
        session.post.return_value.json.return_value = {"ok": True}
        client = Troller("https://example.com", "test-account", session)
        client.login()
        self.assertEqual(session.post.call_args.kwargs["data"]["email"], "test-account")
        client.request("copy_tag", write=True, source="backbones", target="working")
        self.assertEqual(session.post.call_args.kwargs["headers"]["X-CSRFToken"], "csrf-token")

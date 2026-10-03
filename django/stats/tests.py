import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.db import connection
from django.test import SimpleTestCase, TestCase
from django.test.utils import CaptureQueriesContext

from .form import PlotForm, get_default_request, get_most_common_dataset
from .models import (
    BackboneFamily,
    ClassificationResult,
    Dataset,
    FPSMeasurement,
    InstanceResult,
    PretrainedBackbone,
    SemanticSegmentationResult,
    Task,
    TaskType,
)
from .plot import get_plot, get_plot_data
from .request import PlotRequest
from .tables import (
    get_dataset_classification_table,
    get_dataset_downstream_table,
    get_family_classification_table,
    get_family_downstream_table,
    get_github,
    get_head_downstream_table,
    get_paper,
    get_plot_table,
)


class BenchmarkRenderingTests(TestCase):
    fixtures = [str(Path(__file__).resolve().parents[2] / "db.json")]

    def plot_request(self, **changes):
        form = PlotForm(dict(get_default_request(), **changes))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertTrue(form.is_ready())
        return PlotRequest(form.cleaned_data)

    def render_plot(self, request, expected_queries):
        # Query counts stay fixed even though the fixture contains hundreds of results.
        with self.assertNumQueries(expected_queries):
            backbones = list(get_plot_data(request))
        with self.assertNumQueries(0), patch("stats.plot.plot", side_effect=lambda fig, **kw: fig):
            figure = get_plot(backbones, request)
            table = get_plot_table(backbones, request)
        self.assertTrue(figure.data)
        self.assertTrue(table["rows"])
        self.assertEqual(sum(len(trace.x) for trace in figure.data), len(table["rows"]))
        return backbones, figure, table

    def test_all_task_plots_render_without_per_row_queries(self):
        for task, dataset, metric in [
            (4, 1, "top_1"),
            (2, 4, "mAP"),
            (3, 4, "mAP"),
            (5, 20, "ms_m_iou"),
        ]:
            with self.subTest(task=task):
                self.render_plot(
                    self.plot_request(y_task=task, y_dataset=dataset, y_metric=metric), 4
                )

    def test_multi_task_comparison_and_both_legends(self):
        request = self.plot_request(
            x_axis="results", x_task=2, x_dataset=4, x_metric="mAP",
            legend_attribute="instance.head.name",
            **{"legend_attribute_(second)": "classification.resolution"},
        )
        self.assertEqual(request.query_type, PlotRequest.MULTI)
        backbones, figure, table = self.render_plot(request, 5)
        expected_pairs = sum(
            sum(x.mAP is not None and y.top_1 is not None
                for x in pb.x_results for y in pb.y_results)
            for pb in backbones
        )
        self.assertEqual(len(table["rows"]), expected_pairs)
        self.assertTrue(all(" / " in trace.name for trace in figure.data))

    def test_pretraining_and_resolution_filters(self):
        request = self.plot_request(
            _pretrain_dataset=2, _pretrain_method="Supervised", y_resolution=384
        )
        backbones, _, _ = self.render_plot(request, 4)
        for pb in backbones:
            self.assertEqual(pb.pretrain_dataset_id, 2)
            self.assertEqual(pb.pretrain_method, "Supervised")
            self.assertTrue(all(result.resolution == 384 for result in pb.filtered_results))

    def test_family_plot_preserves_point_and_row_order(self):
        request = self.plot_request()
        with self.assertNumQueries(4):
            backbones = list(get_plot_data(request, family_name="Swin"))
        with self.assertNumQueries(0), patch("stats.plot.plot", side_effect=lambda fig, **kw: fig):
            figure = get_plot(backbones, request)
            table = get_plot_table(backbones, request, page="family")
        expected = [81.3, 83.2, 83.5, 84.5, 85.2, 86.4, 87.3, 86.3, 80.9, 83.2]
        self.assertEqual(list(figure.data[0].y), expected)
        self.assertEqual([row["Top-1"] for row in table["rows"]], expected)

    def test_spiking_families_hidden_unless_requested(self):
        BackboneFamily.objects.filter(name="Swin").update(spiking=True)
        spiking = set(BackboneFamily.objects.filter(spiking=True).values_list("name", flat=True))
        hidden = list(get_plot_data(self.plot_request()))
        self.assertTrue(hidden)
        self.assertFalse(any(pb.family.spiking for pb in hidden))

        request = self.plot_request(_show_spiking="on")
        shown = list(get_plot_data(request))
        self.assertEqual(len(shown), len(hidden) + PretrainedBackbone.objects.filter(
            family__spiking=True, classificationresult__dataset=1).distinct().count())
        with patch("stats.plot.plot", side_effect=lambda fig, **kw: fig):
            figure = get_plot(shown, request)
        for trace in figure.data:
            family = re.sub(r"<[^>]+>", "", trace.name)
            expected = "diamond" if family in spiking else "circle"
            self.assertEqual(set(trace.marker.symbol), {expected}, family)

        # A spiking family's own page always shows its models.
        self.assertTrue(list(get_plot_data(self.plot_request(), family_name="Swin")))

    def test_metadata_axes_and_publication_dates(self):
        self.render_plot(self.plot_request(x_axis="pub_date"), 4)
        request = self.plot_request(y_axis="m_parameters", x_axis="pub_date")
        self.assertEqual(request.query_type, PlotRequest.NONE)
        self.render_plot(request, 6)

    def test_classification_throughput(self):
        request = self.plot_request(x_axis="fps", x_gpu="V100", x_precision="AMP")
        backbones, _, _ = self.render_plot(request, 4)
        measurements = {
            (backbone.pk, fps.resolution): fps.fps
            for pb in backbones
            for backbone in [pb.backbone]
            for fps in backbone.fps_measurements.filter(gpu="V100", precision="AMP")
        }
        for pb in backbones:
            for result in pb.filtered_results:
                self.assertEqual(result.fps, measurements[pb.backbone_id, result.resolution])

    def test_downstream_throughput_filters_gpu_and_precision(self):
        for model, task, dataset, metric in [
            (InstanceResult, 2, 4, "mAP"),
            (SemanticSegmentationResult, 5, 20, "ms_m_iou"),
        ]:
            with self.subTest(model=model):
                result = model.objects.filter(dataset_id=dataset).first()
                for gpu, precision, fps in [("H100", "BF16", 321), ("H100", "FP32", 123)]:
                    measurement = FPSMeasurement.objects.create(
                        backbone_name=result.pretrained_backbone.backbone.name,
                        resolution=512, fps=fps, gpu=gpu, precision=precision,
                    )
                    result.fps_measurements.add(measurement)
                request = self.plot_request(
                    y_task=task, y_dataset=dataset, y_metric=metric,
                    x_axis="fps", x_gpu="H100", x_precision="BF16",
                )
                backbones, _, _ = self.render_plot(request, 4)
                results = [r for pb in backbones for r in pb.filtered_results]
                self.assertEqual([(r.pk, r.fps) for r in results], [(result.pk, 321)])

    def test_classification_tables_use_four_queries(self):
        for build, name in [
            (get_family_classification_table, "Swin"),
            (get_dataset_classification_table, "ImageNet-1k"),
        ]:
            with self.subTest(table=build.__name__), self.assertNumQueries(4):
                table = build(name)
                self.assertTrue(table["rows"])
                self.assertEqual(len(table["rows"]), len(table["links"]))
        swin = get_family_classification_table("Swin")
        row = next(row for row in swin["rows"] if row["model"] == "Swin-T")
        self.assertEqual(row["IN-1k"], "81.3/95.5")
        self.assertEqual(row["pretrain"], "IN-1k : Sup. : 300")

    def test_downstream_tables_use_four_queries(self):
        for task, dataset, head in [
            (TaskType.DETECTION, "COCO (val)", "Mask R-CNN"),
            (TaskType.INSTANCE_SEG, "COCO (val)", "Mask R-CNN"),
            (TaskType.SEMANTIC_SEG, "ADE20K (val)", "UPerNet"),
        ]:
            for build, name in [
                (get_family_downstream_table, "Swin"),
                (get_head_downstream_table, head),
                (get_dataset_downstream_table, dataset),
            ]:
                with self.subTest(task=task, table=build.__name__), self.assertNumQueries(4):
                    tables = build(name, task)
                    if isinstance(tables, dict):
                        tables = [tables]
                    self.assertTrue(tables)
                    for table in tables:
                        self.assertTrue(table["rows"])
                        self.assertEqual(len(table["rows"]), len(table["links"]))

    def test_default_dataset_preserves_first_seen_ties(self):
        task = Task.objects.get(name=TaskType.CLASSIFICATION.value)
        first_dataset = Dataset.objects.get(pk=1)
        second_dataset = Dataset.objects.get(pk=7)
        result = ClassificationResult.objects.first()
        results = []
        for dataset in [second_dataset, first_dataset]:
            results.append(ClassificationResult(
                pretrained_backbone_id=result.pretrained_backbone_id,
                dataset=dataset, resolution=224, top_1=80,
            ))
        ClassificationResult.objects.bulk_create(results)
        queryset = ClassificationResult.objects.filter(pk__in=[r.pk for r in results]).order_by("pk")
        with self.assertNumQueries(1):
            self.assertEqual(get_most_common_dataset(queryset, task), second_dataset.pk)

    def test_empty_tables(self):
        self.assertIsNone(get_dataset_classification_table("missing"))
        self.assertEqual(get_family_downstream_table("missing", TaskType.DETECTION), [])

    def test_page_query_budgets_and_real_plotly_output(self):
        for path, budget in [
            ("/", 16),
            ("/families/Swin/", 35),
            ("/heads/Mask R-CNN/", 30),
            ("/datasets/ImageNet-1k/", 25),
            ("/datasets/COCO (val)/", 30),
            ("/datasets/ADE20K (val)/", 30),
        ]:
            with self.subTest(path=path), CaptureQueriesContext(connection) as queries:
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertContains(response, "Plotly.newPlot")
                self.assertLessEqual(len(queries), budget)

    def test_plot_updates_return_only_the_plot_section(self):
        bundle = "plotly.js v"  # the banner of the inlined plotly.js library
        for path in ["/", "/families/Swin/", "/heads/Mask R-CNN/", "/datasets/COCO (val)/"]:
            with self.subTest(path=path):
                page = self.client.get(path)
                self.assertContains(page, bundle)
                with CaptureQueriesContext(connection) as page_queries:
                    self.client.get(path)
                with CaptureQueriesContext(connection) as update_queries:
                    update = self.client.get(path, headers={"X-Plot-Update": "1"})
                self.assertEqual(update["Cache-Control"], "no-store")
                for part in ['id="plot-area"', 'id="plot-options"', 'id="plot-table"', "Plotly.newPlot"]:
                    self.assertContains(update, part)
                self.assertNotContains(update, bundle)
                self.assertNotContains(update, "<html")
                self.assertLessEqual(len(update_queries), len(page_queries))

    def test_incomplete_plot_update_returns_only_the_options(self):
        # A task chosen without its dataset: the page keeps its current plot.
        update = self.client.get(
            "/",
            {"y_axis": "results", "y_task": 4, "x_axis": "gflops", "legend_attribute": "family.name"},
            headers={"X-Plot-Update": "1"},
        )
        self.assertContains(update, 'id="plot-options"')
        self.assertContains(update, 'name="y_dataset"')
        self.assertNotContains(update, 'id="plot-area"')
        self.assertNotContains(update, 'id="plot-table"')

    def test_paper_and_github_fallback_precedence(self):
        family = SimpleNamespace(paper="family-paper", github="family-code")
        backbone = SimpleNamespace(paper="backbone-paper", github="backbone-code", family=family)
        pb = SimpleNamespace(paper="pretrain-paper", github="pretrain-code", backbone=backbone)
        result = SimpleNamespace(paper="result-paper")
        for obj, field, expected in [
            (result, "paper", "result-paper"),
            (pb, "paper", "pretrain-paper"),
            (pb, "github", "pretrain-code"),
            (backbone, "paper", "backbone-paper"),
            (backbone, "github", "backbone-code"),
        ]:
            self.assertEqual(get_paper(result, pb), expected)
            setattr(obj, field, "")
        self.assertEqual(get_paper(result, pb), "family-paper")
        self.assertEqual(get_github(pb), "family-code")


class MarkerColorTests(SimpleTestCase):
    def test_wrapped_hue_is_valid_css(self):
        from unittest.mock import patch
        import plotly.graph_objs as go
        from stats.plot import get_marker_configs
        with patch("stats.plot.random.randint", return_value=340):
            markers = get_marker_configs(list(range(54)))
        for marker in markers.values():
            go.Scatter(marker=marker)

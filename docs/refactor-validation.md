# Rendering refactor validation

The refactor shares the related-record queries used by plots and tables, combines
duplicated downstream table and throughput handling, and simplifies form defaults
and task list construction. It changes no models, migrations, templates, URLs, or
dependencies.

## Database work

Measurements used the bundled `db.json`, Django 5.1, Plotly 5.23.0, and an isolated
in-memory SQLite database. Counts include form construction and template rendering.

| Page | Queries before | Queries after |
| --- | ---: | ---: |
| All Backbones, default plot | 757 | 15 |
| Swin family, default plot | 152 | 32 |
| Mask R-CNN head | 1,508 | 29 |
| UPerNet head | 987 | 24 |
| ImageNet-1k dataset | 1,333 | 21 |
| COCO (val) dataset | 2,381 | 27 |
| ADE20K (val) dataset | 1,093 | 27 |

Single-task plot data and detail tables now load in four queries, independent of
the number of displayed results. Rendering their rows, links, and hover text
requires no additional database queries. Related backbone families and pretraining
datasets use batch prefetches to retain the original parent query's joins and row
order. Default dataset selection reads only dataset IDs instead of constructing
complete result and dataset objects.

These are local query counts, not production latency measurements. PostgreSQL and
browser timings were not measured.

## Behavior checks

Compared 378 requests against the original code, including family, head, and
dataset detail pages, list pages, all supported task choices, throughput,
pretraining/resolution filters, publication dates, multiple legend attributes,
and cross-task plots. For comparison, the Plotly HTML wrapper was replaced by the
figure's JSON, with marker randomness fixed. All 342 successful responses matched
exactly, including figure data, hover text, layout, form options, table order, and
links. The other 36 requests had the same existing empty-result errors in both
versions.

The 13 automated regression tests cover constant query counts, classification,
detection, instance and semantic segmentation, multi-task result pairs, throughput
filtering, source-link fallback, default-dataset tie breaking, and point/row order.
Page smoke tests also exercise the real Plotly HTML renderer.

```sh
python django/manage.py test stats --settings=heedless-backbones.test_settings
```

The test suite and Django system checks pass. `git diff --check` passes.

## Existing issues outside this refactor

`makemigrations --check --dry-run` reports pending field-choice changes for
`BackboneFamily.pretrain_method`, `PretrainedBackbone.pretrain_method`, and
`FPSMeasurement.gpu`. The original code reports the same changes. No migration
was generated as part of this refactor.

Some detail requests with no results raise `ValueError` while selecting a default
dataset. These existing errors were preserved to keep this change focused on
behavior-preserving simplification.

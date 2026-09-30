# Paper ingestion

Papers are screened and read by local Claude Code sessions on your claude.ai login (API
keys are removed from their environment, so nothing is billed per token). Nothing runs
until `ingest_papers` is run, by hand or by the optional launchd job on a Mac
([Scheduled runs](#scheduled-runs-on-a-mac)).

## Quick start

1. Log Claude Code in to your claude.ai account (`claude`), and install `requirements.txt`.
2. On a Mac, copy `deploy/local-agent/env.example` to `~/.config/heedless-agent/env`, fill
   it in, test with `deploy/local-agent/run.sh`, then schedule it with
   `deploy/local-agent/install.sh` (every three hours while logged in).
3. Each run discovers papers, screens abstracts, and has Claude Code read up to five
   shortlisted papers (retries first); the server then publishes the clean extractions and
   appends to one aggregate pull request with family YAML, `db.json` and README/About updates.
4. Review everything else in the admin: *Ingestion runs → Review*
   (`/admin/ingestion/ingestionrun/review/`). See [Review page](#review-page).
5. Change the rules by editing the [data entry guide](data-entry-guide.md).

## Setup and manual runs

Install `requirements.txt` and [Claude Code](https://code.claude.com) (logged in with
`claude`), configure `.env` using `example.env`, and run `python django/manage.py
migrate`. `INGESTION_MODEL` and `INGESTION_SCREEN_MODEL` optionally choose the Claude
Code models; by default Claude Code's configured model is used.

Discovery uses Arxiv Troller's JSON API (`/api/ingestion/`, maintained in the
arxiv-troller repository; see [`integrations/arxiv_troller`](../integrations/arxiv_troller/README.md)).
Set `ARXIV_TROLLER_ACCOUNT` to the account string; this installation uses
`igm503@gmail.com`. On every run with discovery it merges `backbones` into
`heedless-backbones` and adds existing arXiv papers found in the benchmark database;
the original tag is not modified. It then queues papers from:

- `tag`: every paper in the working tag;
- `tag_search`: Troller's joint similarity search over all tagged papers (`search`
  with `type=tag`, up to 400 results), within `--lookback-days`;
- `similar`: the 20 nearest papers to each tagged paper, within `--lookback-days`.

Keyword search is not used: it returns many out-of-scope papers, and in-scope papers
are expected to be similar to tagged ones.

The joint tag search returns at most 400 papers; its cursor lists every paper returned so
far, so the client stops just before the request would exceed the server's URL length
limit (about 390 papers). Missing Troller papers are reported with their arXiv IDs. Non-arXiv source links
cannot be represented in an arXiv tag.

From `django/`:

```sh
# Discovery and tag synchronization only; no model calls.
python manage.py ingest_papers --discover-only

# Screen up to 25 abstracts, then read up to 3 shortlisted papers in full and
# validate; saves proposals, not benchmark records.
python manage.py ingest_papers --screen-limit 25 --limit 3

# Only work through the existing shortlist.
python manage.py ingest_papers --skip-discovery --screen-limit 0 --limit 5

# Publish validated additions, including saved ready runs, up to the paper limit.
python manage.py ingest_papers --publish --limit 3

# Reprocess one saved PaperVersion, or include papers already in the database.
python manage.py ingest_papers --skip-discovery --paper-id 123
python manage.py ingest_papers --include-existing
```

Screening and full reads are separate steps. Abstracts are screened in batches of 20
(`--screen-batch`) by a tool-less `claude -p` call with a JSON schema; a paper whose
abstract passes becomes a `shortlisted` run and waits for a full read by the agent,
which is recorded in the same run.
Abstract screening is deliberately inclusive. Full-paper inspection requires an
innovative architecture or pretraining contribution and ImageNet-1k classification
results for the proposed models. Detection and segmentation results are extracted when
present but are not required. Panoptic results are excluded. Results for other models are
added only to fill gaps (see the guide).

## Review and provenance

Five tables preserve the chain from paper to database:

| Table | Records |
| --- | --- |
| `PaperVersion` | Discovery metadata, version-pinned PDF, SHA-256 and extracted pages |
| `IngestionRun` | Engine/model, code and prompt version, screening decision, agent transcript path, turns, time, API-equivalent cost and auth source, errors |
| `ExtractedRecord` | Proposed model fields, quotations (PDF page or saved web source), derivation inputs, inferred fields, notes, uncertainties, reviewer corrections |
| `WebSource` | The repository, release or project pages the agent fetched, with hash and text |
| `ImportChange` | Target object, before/after snapshots, originating extraction or YAML file, run, actor and timestamp |

Each automatically imported object links to its extraction through `source_record`. The Django
admin exposes those links, run details, proposals, private source PDFs, and the
change ledger. Editing an imported object in the stats admin also records before
and after values. Direct SQL or arbitrary shell edits are outside this audit path.

Imports are atomic per paper. Re-running the same extraction is idempotent. Matching
experiments are checked against existing settings; conflicting values and uncertain
matches caused by incomplete metadata stop for review. Existing primary source
links are preserved. A corroborating paper receives its own ledger entry.

Unknown optional metadata remains null; unknown required metadata goes to review. Epoch counts support fractions. Arithmetic
is independently recomputed, including `iterations * effective_batch_size /
dataset_size`. Inputs quoted in the PDF can validate automatically. An input based
on an external source or convention needs a source description, explicit assumption,
and human review. Quotations must occur on the cited PDF page; numeric values must
be supported by their quotations. These checks cannot prove that the model chose
the right table cell or interpreted a result correctly; review the stored evidence.

The extraction follows the [data entry guide](data-entry-guide.md), which the agent's
prompt includes verbatim (its hash is part of `prompt_version`). The model applies judgment
where the guide runs out and explains each call in the record's `note`; judgment does
not send a paper to review. The benchmark schema is strict: required fields
(parameters, pretraining dataset, epochs and resolution, downstream training dataset
and epochs, crop size, ...) must be known. The model lists a field in `uncertain` only
when the paper does not give it, contradicts itself or cannot be read. An uncertain
field, a missing required value, a failed check, a new category, dataset or head, or
a disagreement with a stored value sends the whole paper to review; nothing from it
is added.

Citations: every quotation given must occur on its page, and a value may cite several
places. Names built by the naming conventions and `$key` links between records need no
citation. The guide's dataset sizes (ADE20K 20,000, COCO 117,000, Cityscapes 3,000)
may be used uncited as `source: "convention: <dataset>"`, and a named schedule (1x, 2x,
3x, 6x) supports its epochs.

A new dataset or head is approved with `review_ingestion <run> --approve`. A
disagreement with a stored value is reported as a proposed correction, with both
values; `review_ingestion <run> --approve --allow-updates` applies it.

## Publishing and records

Every publish goes through one function, `publication.publish()`, on the server: your
approval on the review page, `review_ingestion --publish`, the admin action, and the
agent's clean runs (the scheduled job validates on your Mac, then runs `publish_ready` on
the server over SSH). After a successful import it records the change in git, in the
background: changes accumulate in one `auto.records-<batch>` branch and pull request with

- `family_data/<family>.yml` for every pending family, regenerated from the database;
- one `db.json`, exported from the database for the families in the branch (links to
  extraction records are blanked, so it loads without the ingestion tables);
- the README's model table and Updates list, plus the About page's Latest Updates.

Later publications append ordinary commits to the same branch. Titles describe the whole
batch relative to `main`: `Add Swin`, `Add ConvNeXt and Swin`, `Add 8 backbone families`,
`Add 5 backbone families; update 3`, or `Update 4 backbone families`. Up to three families
are named in an additions-only or updates-only title. The description lists every family,
its paper, and whether it is added or updated. A family is counted once. Commit titles
identify the changes in that publication. The run's review page links the shared PR (or
shows why recording failed; database publication still stands).

Review and merge the batch when ready; squash merges are supported. The PR records data
that has already been published to the database, so merging it is not the publication
approval step. After the batch merges, the next publication starts a fresh branch from
current `main`.
A refresh with no pending batch does nothing. Main changes are merged into an open batch
with a normal merge commit, and shared output is regenerated; there are no force-pushes.
Conflicts outside the generated files are reported for manual resolution. The family YAML,
`db.json`, README and About output in this branch are machine-managed; make data corrections
through the importer rather than editing their generated branch copies.

The README model table and dated Updates entries are sorted newest first, with family
names alphabetized within each date and repeated model-addition entries combined.
The README’s **Models** column counts distinct backbone variants in the batch’s `db.json`,
not pretrained checkpoints. Counts are refreshed for existing rows as well as new ones.
Historical paper-specific rows, such as FAN STL, count distinct variants with checkpoints
from that paper; an unmatched row shows an em dash rather than an invented count.
About date groups are also sorted newest first, including existing groups, while preserving
handwritten entries. Dates reflect first publication to the database, not PR merge order.
Families without an ingestion creation record retain the existing fallback to today's date
when first listed; that date is retained while the batch remains open.

Use *Refresh records PR* on the review list (`python manage.py refresh_auto_prs`) to update the
current batch from the database. The sync timer also refreshes it when `main` advances.

**Sync timer.** `deploy/server/sync-main.sh`, run every 30 seconds by
`heedless-sync.timer`, checks GitHub's `main` (one `git ls-remote`). When it has moved, it
deploys it to the site (fast-forward pull, `pip install` if requirements changed, `migrate`,
`collectstatic`, graceful gunicorn reload) and then refreshes the aggregate records PR. So
**anything merged to `main` goes live within about 30 seconds.** It never runs
`makemigrations` or loads `db.json` (that would overwrite rows published since the dump).
Install: `sudo cp deploy/server/heedless-sync.{service,timer} /etc/systemd/system/ && sudo
systemctl daemon-reload && sudo systemctl enable --now heedless-sync.timer`; logs with
`journalctl -u heedless-sync`. If the site's checkout has local commits, the pull refuses and
each tick logs the failure until that is resolved.

Server setup: `RECORDS_REPO` points at a clone used only for this (never the site's
checkout). Pull requests are opened as a GitHub App, so they show as
`heedless-backbones-agent[bot]`, carry the `automated` label and end with a line saying the
pipeline opened them. Create the app (Settings → Developer settings → GitHub Apps) with no
webhook, *Contents* and *Pull requests* read and write, installed only on this repository;
put its private key on the server readable only by `django`, and set `GITHUB_APP_ID` and
`GITHUB_APP_KEY` (the key's path) in the deploy settings. Each push or pull request action
uses a one-hour installation token, and commits are authored by the bot. Without the app,
git and `gh` use whatever login the `django` user has. Without `RECORDS_REPO`, publishing
works and the run notes that nothing was recorded.

## Manual review

```sh
# Runs waiting for review, and a draft of one in the family_data format.
python manage.py review_ingestion --list
python manage.py review_ingestion 42 --export   # family_data/review/<family>-run42.yml
```

The draft starts with a `review` block listing each uncertain or failing field, the
reason, and the cited PDF page and quotation. A paper that adds to an existing family
produces the whole family, so matching records are recognised on import. Correct the
values, delete the `review` block, then import the file:

```sh
python manage.py add_yaml family_data/review/FixtureNet-run42.yml --run 42 --actor ian
```

This validates the file with the same model checks, imports it atomically, marks run
42 as imported, records every change with the file and run in `ImportChange`, and
records the family like any other publish (in the aggregate pull request; without
`RECORDS_REPO` it writes `family_data/FixtureNet.yml` locally instead). New records without a paper link
get the run's arXiv version. `add_yaml` without `--run` still imports a hand-written
family file. Existing records are never changed without `--allow-updates`.

## Categories

New token mixers (`model_type`) and pretraining methods need your approval; neither
the model nor `review_ingestion --approve` can add one. Proposals block their paper
until approved:

```sh
python manage.py approve_category --pending
python manage.py approve_category model_type 'Novel mixer' --actor ian --note 'Distinct operator (Sec. 3)'
```

Approved categories are stored in `stats.Category` and appear in validation, forms and
the admin, together with the built-in enums.

## Re-validating a run

```sh
# Validate a saved run again, without model calls or publishing.
python manage.py review_ingestion 42

# Accept a documented derivation assumption after inspecting the evidence.
python manage.py review_ingestion 42 --approve --actor ian \
  --note 'Checked the cited training split and effective batch size' --publish

# Explicitly permit a reviewed correction to existing data.
python manage.py review_ingestion 42 --approve --allow-updates --actor ian \
  --note 'Correcting the transcription against Table 3' --publish
```

Approval does not bypass missing quotes, invalid values, failed calculations,
uncertain fields or new categories.
Raw proposals are read-only in the admin; rerun a paper to obtain a fresh extraction,
or correct an existing benchmark through the audited stats admin. Runs needing
review are not retried automatically. Network/extraction failures are retried on
later manual runs up to three failed runs; `--paper-id` explicitly overrides that.

## The agent

A local Claude Code session reads each shortlisted paper. For each paper the orchestrator (`ingestion/agent.py`) prepares a
working folder under `INGESTION_STORAGE/agent/` with the PDF, the page text citations
are checked against (`pages/page-NNN.txt`), arXiv metadata, the first pages' links and
the prompt, and runs `claude -p` confined to it: reads anywhere, writes only inside the
folder, and only these commands through `./hb`:

- `./hb lookup names|family|model|search`: read-only database lookups (to reuse names
  and find gaps in other models' results);
- `./hb fetch <url>`: save the paper's own repository (README, files, releases) or
  project page, if linked on its first two pages or in its arXiv abstract/comments.
  Snapshots are stored with the run (`WebSource`), and URL citations are checked
  against them;
- `./hb validate`: every importer check on `extraction.json`, inside a transaction that
  is always rolled back.

Web search, web fetch and subagents are disabled. The agent cannot write to the
database: the orchestrator validates its output again and submits it through the
importer, with the transcript, turns, API-equivalent cost and auth source in the run's
`calls` (`auth: "none"` means the claude.ai login was used).

```sh
python manage.py agent_extract 2207.03620                  # one paper; validate only
python manage.py agent_extract 2207.03620 --submit --publish
python manage.py ingest_papers --limit 5 --publish
```

### Scheduled runs on a Mac

Scheduled runs use their own checkout: `install.sh` creates a git worktree of this repository
at `origin/main` (`~/.local/share/heedless-agent/repo`, or `AGENT_REPO`), and each run first
moves it to the latest `main` and re-runs the updated script, installing requirements when
they change. So merged changes (the guide included) reach the agent at its next run, and
your own checkout, its branch and uncommitted work are never touched. Running `run.sh` from
your own checkout uses that checkout as it is.

`deploy/local-agent/run.sh` opens an SSH tunnel to the server's PostgreSQL, runs
`ingest_papers` (validation only), copies new PDFs to the server's storage for the review
page every minute during the run (the server needs `rsync`), and then runs `publish_ready` on the server over SSH (with `PUBLISH=1`). Configure `~/.config/heedless-agent/env` from `env.example`, then
`deploy/local-agent/install.sh` installs a launchd job that runs every three hours
while you are logged in (log: `~/Library/Logs/heedless-agent.log`). A run missed while the
Mac was asleep happens once on wake; during a run the Mac is kept from idle sleep. If a run
is cut off anyway (lid closed, shutdown), nothing is half-imported: the paper's run is
marked failed at the next start and retried, up to three times. Uninstall with
`launchctl bootout gui/$(id -u)/com.heedlessbackbones.agent`.

## Review page

In the Django admin, *Ingestion runs → Review* (`/admin/ingestion/ingestionrun/review/`)
lists runs waiting for review and recently imported runs to audit. A run's page shows
each flagged value (uncertain, inferred, derived, web-sourced, corrected, or with a
problem), or every value, next to a crop of the PDF around its quotation (or the saved
web page excerpt, highlighted), with the agent's notes and summary. Approve and
publish, approve including corrections to stored data, reject, or correct a single
value; each needs a note and is recorded in `ImportChange`. A corrected value's source
is the reviewer. Imported records are corrected through the stats admin.

The page follows the `family_data` layout (family → backbones → pretrained models →
results, with throughput under its owner), with sections for proposals (each listing the
results that use it) and for results added to existing models. Filter to uncertain
fields only, flagged fields, or everything. *Remove* takes a section out of the import
together with everything under it (a pretrained model's results, or the results using a
proposed head); *Restore* brings it back. Each removal is logged, and the run is
re-validated at once.

*Reject and retry with this note* (or `review_ingestion <run> --reject-retry --actor
<you> --note "..."`) rejects the run and queues a new full read of the paper, read first at
the next ingestion run. The agent is given the feedback of every earlier review in the
chain (your notes, the sections you removed and why, the values you corrected) and the
rejected records for reference.

## Eval mode

`evaluate_ingestion` has fresh Claude Code sessions read the papers behind the 20
earliest-added families (`--families`, or `--paper <arXiv ID>`) and scores the output
against the database. It never writes to the database: every statement other than a
read raises, the sessions' validator skips the import check, and `./hb lookup` hides
the paper's own stored families. Each paper is read in full.

```sh
python manage.py evaluate_ingestion --paper 2207.03620
python manage.py evaluate_ingestion                          # all 17 papers
python manage.py evaluate_ingestion --rescore evals/<run>    # re-score saved output; no model calls
```

Results go to `evals/<timestamp>-claude-code/` (`EVAL_DIR`, not committed): `config.json`,
`summary.md`/`summary.json`, and per paper the agent's working folder (prompt, page
text, sources, extraction, validation report, transcript), `agent.json` (turns, time,
tokens, API-equivalent cost, auth source), `records.json`, the stored families as
`reference.yml`, and `score.json`.

Results are matched by backbone, pretraining dataset and method, plus dataset and
resolution (classification), head, dataset and task (detection/instance), or head and
dataset (semantic). Recall counts only stored results whose source is the paper being read; precision
also credits matches with results stored from other papers. Results the paper reports
for other models are scored separately as baselines (already stored, or new gap-fills).
Scores are result recall and precision, accuracy of metric values (within 0.05) and of
all scored fields, uncertain fields and citation failures;
`score.json` lists missed and extra items and every wrong value. The reference is the
database itself, so a mismatch can also be an error in the stored record.

## Limits and operation

By default a run screens up to 25 abstracts (`--screen-limit`, `INGESTION_MAX_SCREENS`)
and reads up to three papers in full (`--limit`, `INGESTION_MAX_PAPERS`), each within
`--agent-timeout` seconds (default 3600). Usage counts against your Claude plan's
limits; the recorded cost is Claude Code's API-equivalent estimate.

Local file and PostgreSQL advisory locks prevent overlapping jobs. Paper downloads
are pinned to an arXiv version, capped at 30 MB/200 pages/350,000 text characters,
and kept in private `INGESTION_STORAGE`. Page text is rebuilt from word positions
(`ingestion/pdf_text.py`, pdfplumber): table cells stay separated, superscripts are
written with a caret (`224^2`), and two-column pages are read column by column. The
model reads this text and citations are checked against it. PDFs without extractable
text require separate inspection. Back up that directory together with PostgreSQL. Never expose
it as public media.

Tests use isolated SQLite and mocked Claude Code sessions:

```sh
python django/manage.py test stats ingestion --settings=heedless-backbones.test_settings
```

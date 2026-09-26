"""Discovery and the queues of papers to screen and to read in full."""
import hashlib
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from django.db import OperationalError, connection
from django.db.models import Count, F, Q, Subquery, OuterRef
from django.utils import timezone

from .importer import REGISTRY
from .models import IngestionRun, PaperVersion
from .sources import arxiv_id



def current_paper_ids():
    ids = set()
    for model in REGISTRY.values():
        for field in ("paper", "source"):
            if any(f.name == field for f in model._meta.fields):
                ids.update(filter(None, (arxiv_id(value) for value in model.objects.values_list(field, flat=True))))
    return ids


def discover(client, source_tag, target_tag, lookback_days=14):
    existing = current_paper_ids()
    missing = client.sync(source_tag, target_tag, sorted(existing))
    since = (timezone.now() - timedelta(days=lookback_days)).isoformat()
    sources = [("tag", {"tag": target_tag}, client.papers("tag", tag=target_tag)),
               ("tag_search", {"tag": target_tag, "since": since}, client.tag_search(target_tag, since)),
               ("similar", {"tag": target_tag, "since": since}, client.papers("similar", tag=target_tag, since=since))]
    created = 0
    for action, params, found in sources:
        for candidate in found:
            identifier = arxiv_id(candidate["arxiv_id"])
            if not identifier:
                raise ValueError("Arxiv Troller returned an invalid arXiv ID")
            revision = candidate.get("updated") or candidate["created"]
            metadata = dict(candidate, discovery={"action": action, **params})
            paper, new = PaperVersion.objects.get_or_create(
                arxiv_id=identifier, revision=revision,
                defaults={"title": candidate["title"], "abstract": candidate["abstract"],
                          "metadata": metadata, "existing_in_database": identifier in existing},
            )
            created += new
            if identifier in existing and not paper.existing_in_database:
                paper.existing_in_database = True
                paper.save(update_fields=["existing_in_database"])
    return created, missing


# pg advisory lock held by the ingest_papers process for its whole run.
RUN_LOCK = 73410290


def reconnect(attempts=10, wait=30):
    """Replace a database connection that died during a long agent session (the SSH tunnel
    dropped, say), retrying while the tunnel comes back, and take the run lock again."""
    if connection.connection is None or connection.is_usable():
        return
    for attempt in range(attempts):
        connection.close()
        try:
            connection.ensure_connection()
            break
        except OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)
    if connection.vendor == "postgresql":
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", [RUN_LOCK])
            if not cursor.fetchone()[0]:
                raise RuntimeError("Another ingestion job took the run lock while the connection was down")


def shortlist(limit):
    """Runs whose abstract passed screening, oldest first, for a full read."""
    return list(IngestionRun.objects.filter(status=IngestionRun.Status.SHORTLISTED)
                .select_related("paper").order_by(F("retry_of").desc(nulls_last=True), "pk")[:limit])


def candidates(limit, include_existing=False):
    """Papers to screen: never screened, or failed fewer than three times."""
    latest = IngestionRun.objects.filter(paper_id=OuterRef("pk")).order_by("-pk")
    papers = PaperVersion.objects.annotate(
        last_status=Subquery(latest.values("status")[:1]),
        failures=Count("runs", filter=Q(runs__status=IngestionRun.Status.FAILED)),
    ).filter(Q(last_status__isnull=True) | Q(last_status=IngestionRun.Status.FAILED, failures__lt=3))
    if not include_existing:
        papers = papers.filter(existing_in_database=False)
    # Alternate recent discoveries and the oldest backlog to avoid starving either.
    newest = list(papers.order_by("-revision", "pk")[:limit])
    oldest = list(papers.order_by("discovered_at", "pk")[:limit])
    selected = {}
    for pair in zip(newest, oldest):
        for paper in pair:
            selected.setdefault(paper.pk, paper)
    return list(selected.values())[:limit]


def code_version():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ("ingestion", "stats"):
        for path in sorted((root / "django" / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        revision = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = os.getenv("INGESTION_CODE_VERSION", "unknown")
    return f"{revision[:40]}+sha256:{digest.hexdigest()[:48]}"


def parse_records(output):
    """Records from an extraction response, as ExtractedRecord field values."""
    records, seen = [], set()
    for item in output["records"]:
        key = item["key"]
        if key in seen or not 1 <= len(key) <= 150:
            raise ValueError("Extraction contains invalid or repeated record keys")
        seen.add(key)
        data = {field["name"]: field["value"] for field in item["fields"]}
        if len(data) != len(item["fields"]):
            raise ValueError(f"Repeated fields in record {key}")
        records.append({"key": key, "kind": item["kind"], "data": data,
                        "evidence": item["evidence"], "uncertain": item.get("uncertain", []),
                        "inferred": item.get("inferred", []), "note": item.get("note", "")})
    return records

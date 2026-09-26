from django.db import models


class PaperVersion(models.Model):
    arxiv_id = models.CharField(max_length=50)
    revision = models.CharField(max_length=100, help_text="Discovery source's updated timestamp")
    version = models.PositiveIntegerField(null=True, blank=True)
    title = models.TextField()
    abstract = models.TextField(blank=True)
    metadata = models.JSONField(default=dict)
    discovered_at = models.DateTimeField(auto_now_add=True)
    existing_in_database = models.BooleanField(default=False)
    pdf = models.FileField(upload_to="papers/%Y/%m/", blank=True)
    sha256 = models.CharField(max_length=64, blank=True)
    pages = models.JSONField(default=list, help_text="Extracted text, in PDF page order")

    class Meta:
        constraints = [models.UniqueConstraint(fields=["arxiv_id", "revision"], name="unique_paper_revision")]

    def __str__(self):
        return f"{self.arxiv_id} v{self.version or '?'}: {self.title[:80]}"

    @property
    def url(self):
        suffix = f"v{self.version}" if self.version else ""
        return f"https://arxiv.org/abs/{self.arxiv_id}{suffix}"


class IngestionRun(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running"
        SHORTLISTED = "shortlisted", "Shortlisted: abstract passed, awaiting full read"
        REJECTED = "rejected"
        REVIEW = "review", "Needs review"
        READY = "ready"
        IMPORTED = "imported"
        FAILED = "failed"

    paper = models.ForeignKey(PaperVersion, on_delete=models.PROTECT, related_name="runs")
    status = models.CharField(max_length=20, choices=Status, default=Status.RUNNING, db_index=True)
    provider = models.CharField(max_length=20)
    model = models.CharField(max_length=100)
    prompt_version = models.CharField(max_length=50)
    code_version = models.CharField(max_length=100)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    decision = models.JSONField(default=dict)
    calls = models.JSONField(default=list, help_text="Requests, responses and usage; never API keys")
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    estimated_cost = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)
    error = models.TextField(blank=True)
    retry_of = models.ForeignKey("self", on_delete=models.PROTECT, null=True, blank=True, related_name="retries",
                                 help_text="The rejected run this one retries")
    feedback = models.TextField(blank=True, help_text="The reviewer's note that asked for this retry")

    def __str__(self):
        return f"{self.paper.arxiv_id}: {self.status} (run {self.pk})"


class WebSource(models.Model):
    """A page from the paper's official repository, release page or project page, as fetched
    for an extraction. Web citations are checked against this text, not the live page."""
    run = models.ForeignKey(IngestionRun, on_delete=models.PROTECT, related_name="sources")
    url = models.URLField(max_length=500)
    fetched_from = models.URLField(max_length=500, help_text="The URL actually downloaded (e.g. the raw README)")
    fetched_at = models.DateTimeField()
    sha256 = models.CharField(max_length=64)
    text = models.TextField()

    class Meta:
        constraints = [models.UniqueConstraint(fields=["run", "url"], name="unique_run_source")]

    def __str__(self):
        return f"{self.url} (run {self.run_id})"


class ExtractedRecord(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending"
        APPROVED = "approved"
        REJECTED = "rejected"
        IMPORTED = "imported"

    run = models.ForeignKey(IngestionRun, on_delete=models.PROTECT, related_name="records")
    key = models.CharField(max_length=150)
    kind = models.CharField(max_length=40)
    data = models.JSONField(default=dict)
    evidence = models.JSONField(default=list)
    uncertain = models.JSONField(default=list, help_text="Fields the model could not establish, with reasons")
    inferred = models.JSONField(default=list, help_text="Reference/classification fields implied, not quoted, with their source")
    note = models.TextField(blank=True, help_text="The model's judgment calls for this record")
    overrides = models.JSONField(default=list, help_text="Values corrected by a reviewer: field, old and new value, who, why")
    issues = models.JSONField(default=list)
    status = models.CharField(max_length=20, choices=Status, default=Status.PENDING, db_index=True)
    reviewed_by = models.CharField(max_length=150, blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    review_note = models.TextField(blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["run", "key"], name="unique_extraction_key")]

    def __str__(self):
        return f"{self.kind}: {self.key} (run {self.run_id})"


class ImportChange(models.Model):
    # Automatic imports cite the extraction; manual YAML imports cite the file (and run, if any).
    record = models.ForeignKey(ExtractedRecord, on_delete=models.PROTECT, related_name="changes", null=True, blank=True)
    run = models.ForeignKey(IngestionRun, on_delete=models.PROTECT, related_name="changes", null=True, blank=True)
    origin = models.CharField(max_length=300, blank=True, help_text="YAML file for a manual import")
    model = models.CharField(max_length=100)
    object_id = models.PositiveBigIntegerField()
    before = models.JSONField(null=True, blank=True)
    after = models.JSONField()
    actor = models.CharField(max_length=150)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["model", "object_id"], name="import_target_idx")]

    def __str__(self):
        source = f"extraction {self.record_id}" if self.record_id else self.origin or "manual edit"
        return f"{self.model} #{self.object_id} from {source}"

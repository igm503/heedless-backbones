import math
import re
import unicodedata

# Conventions from docs/data-entry-guide.md: round dataset sizes for iteration-to-epoch
# conversions, and the batch size of standard downstream setups.
CONVENTIONS = {"ade20k": 20000, "coco": 117000, "cityscapes": 3000}
STANDARD_BATCH_SIZE = 16
# Named detection schedules and their epochs.
SCHEDULES = {"1": 12, "2": 24, "3": 36, "6": 72}
# Constructed by naming conventions or links between records; not quoted from the paper.
# instance_type follows from the quoted metric (box or mask AP).
UNCITED_FIELDS = {"name", "backbone_name", "tasks", "instance_type"}
# Classifications applying the guide: a quotation or a note naming the field.
JUDGMENT_FIELDS = {"model_type", "hierarchical", "pretrain_method"}


def normalized(text):
    """Compare text as read: ligatures unfolded (ﬁ -> fi), hyphens between word parts and
    line-break hyphenation removed, whitespace collapsed, case ignored."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*(?=\w)", "", text)
    text = re.sub(r"(?<=\w)-\s*$", "", text)  # A quote ending at a line-break hyphen ("we em-").
    return " ".join(text.split()).casefold()


def check_citation(citation, pages, sources=None):
    """A quotation from a PDF page or from a fetched web source ({url: text})."""
    quote = citation.get("quote", "")
    if citation.get("url"):
        text = (sources or {}).get(citation["url"])
        if text is None:
            raise ValueError(f"Citation source {citation['url']} was not fetched for this extraction")
        if len(quote.strip()) < 6 or normalized(quote) not in normalized(text):
            raise ValueError(f"Citation quotation does not occur in {citation['url']}")
    else:
        page = citation.get("page", 0)
        if not isinstance(page, int) or not 1 <= page <= len(pages):
            raise ValueError("Citation has an invalid PDF page")
        if len(quote.strip()) < 6 or normalized(quote) not in normalized(pages[page - 1]):
            raise ValueError(f"Citation quotation does not occur on PDF page {page}")
    if not citation.get("location", "").strip():
        raise ValueError("Citation needs a table/row/column or section location")


SUFFIXES = {"k": 1e3, "m": 1e6}


def contains_number(quote, value):
    """Whether the quote states the value; "160K" supports both 160 and 160,000, "28M" both
    28 and 28,000,000 (parameters are recorded in millions)."""
    for number, suffix in re.findall(r"(?<![\w.])(-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:\s?([kKmM])(?![a-zA-Z]))?", quote):
        number = float(number.replace(",", ""))
        candidates = [number] + ([number * SUFFIXES[suffix.lower()]] if suffix else [])
        if any(math.isclose(candidate, value, rel_tol=1e-6, abs_tol=1e-6) for candidate in candidates):
            return True
    return False


def is_convention(item):
    """An uncited input from the guide's conventions: a dataset size ("convention: ADE20K")
    or the standard downstream batch size ("convention: standard batch size")."""
    source = (item.get("source") or "").casefold()
    if not source.startswith("convention:"):
        return False
    if item["name"] == "effective_batch_size":
        return item["value"] == STANDARD_BATCH_SIZE
    if item["name"] != "dataset_size":
        return False
    dataset = re.sub(r"[^a-z0-9]", "", source.split(":", 1)[1])
    return any(dataset.startswith(key) and item["value"] == size for key, size in CONVENTIONS.items())


def names_schedule(quote, value):
    """A quote naming a detection schedule ("1x", "3× schedule") supports its epochs."""
    return any(SCHEDULES.get(number) == value for number in re.findall(r"(?<![\w.])(\d)\s*[x×]", quote))


def derive(derivation, pages, reviewed=False, sources=None):
    inputs = {}
    for item in derivation["inputs"]:
        if item["name"] in inputs or not math.isfinite(item["value"]):
            raise ValueError("Invalid or repeated derivation input")
        if item.get("citation"):
            check_citation(item["citation"], pages, sources)
            if not contains_number(item["citation"]["quote"], item["value"]):
                raise ValueError(f"Derivation input {item['name']} is not supported by its quotation")
        elif is_convention(item):
            pass
        elif not item.get("source") or not derivation["assumptions"]:
            raise ValueError("An uncited derivation input needs a source description and explicit assumption")
        elif not reviewed:
            raise ValueError("Convention-based derivation inputs need review")
        inputs[item["name"]] = item["value"]
    method = derivation["method"]
    if method == "iterations_to_epochs":
        if set(inputs) != {"iterations", "effective_batch_size", "dataset_size"} or min(inputs.values()) <= 0:
            raise ValueError("Epoch derivation needs positive iterations, effective_batch_size and dataset_size")
        return inputs["iterations"] * inputs["effective_batch_size"] / inputs["dataset_size"]
    if set(inputs) != {"a", "b"}:
        raise ValueError("Arithmetic derivations require inputs a and b")
    if method == "multiply":
        return inputs["a"] * inputs["b"]
    if method == "divide" and inputs["b"] != 0:
        return inputs["a"] / inputs["b"]
    raise ValueError("Invalid derivation method or division by zero")


def reference_fields(kind):
    """Fields naming an existing entity (head, dataset, task): a note may stand in for a quote."""
    from django.db import models
    from .importer import REGISTRY
    model = REGISTRY.get(kind)
    references = {field.name for field in model._meta.fields if isinstance(field, models.ForeignKey)} if model else set()
    return references | JUDGMENT_FIELDS


# Proposal metadata for new datasets, heads and categories: always reviewed by a person.
PROPOSAL_FIELDS = {"eval", "website", "paper", "github", "scope", "value"}


def validate_evidence(record, pages, sources=None):
    references = reference_fields(record.kind)
    inferred = {item.get("field") for item in (getattr(record, "inferred", None) or [])}
    # A reviewer's correction is its own source (the review page records who and why).
    overridden = {item.get("field") for item in (getattr(record, "overrides", None) or [])}
    proposal = record.kind in {"category", "dataset", "head"}
    evidence = {}
    for item in record.evidence:
        check_citation(item["citation"], pages, sources)  # Every quotation given must be real, even if not required.
        evidence.setdefault(item["field"], []).append(item)  # A value may cite several places.
    for field, value in record.data.items():
        if value is None or value == "" or field in {"paper", "source"} | UNCITED_FIELDS:
            continue
        if proposal and field in PROPOSAL_FIELDS or field in overridden:
            continue
        if isinstance(value, str) and value.startswith("$"):
            continue  # A reference to another record in the same extraction.
        items = evidence.get(field)
        if not items and field in references:
            if field not in inferred:
                raise ValueError(f"{field} has no quotation and is not listed in inferred with its source")
            continue
        if not items:
            raise ValueError(f"Missing evidence for {field}")
        derivations = [item["derivation"] for item in items if item.get("derivation")]
        if derivations:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Derived field {field} must be numeric")
            for derivation in derivations:
                expected = derive(derivation, pages, reviewed=bool(record.reviewed_by), sources=sources)
                if not math.isclose(expected, value, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(f"Incorrect derivation for {field}: expected {expected}")
                if derivation["assumptions"] and not record.reviewed_by:
                    if not all(item.get("citation") or is_convention(item) for item in derivation["inputs"]):
                        raise ValueError(f"Derivation assumptions for {field} need review")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            quotes = [item["citation"]["quote"] for item in items]
            supported = any(contains_number(quote, value) for quote in quotes) or (
                field.endswith("train_epochs") and any(names_schedule(quote, value) for quote in quotes))
            if not math.isfinite(value) or not supported:
                raise ValueError(f"Value of {field} is not supported by its quotation")

import hashlib
import json
from pathlib import Path

from django.db import models

GUIDE_PATH = Path(__file__).resolve().parents[2] / "docs" / "data-entry-guide.md"


def guide():
    """The model-facing part of docs/data-entry-guide.md."""
    text = GUIDE_PATH.read_text()
    return text.split("<!-- prompt:start -->", 1)[1].split("<!-- prompt:end -->", 1)[0].strip()


VERSION = "backbones-v3+guide-" + hashlib.sha256(guide().encode()).hexdigest()[:8]
CRITERIA = """Select papers that introduce an innovative general-purpose vision backbone
architecture OR pretraining regime, and report ImageNet-1k classification results for
the proposed models. Object detection, instance segmentation and semantic segmentation
results are extracted when present but are not required (downstream_results records
whether there are any). Ignore panoptic segmentation. Results must concern the proposed
method, not just comparison baselines. Novelty needs a concrete contribution, not the
authors' claim alone. Be inclusive at abstract screening: if ImageNet-1k evaluation or
novelty is unclear, inspect the full paper. Reject at screening only when clearly out
of scope.
"""


def screening(papers):
    """Abstract screening for a batch of {"arxiv_id", "title", "abstract"}."""
    return CRITERIA + """
Screen each paper below from its title and abstract, and return one decision per paper
(use its arxiv_id). qualifies is true when the paper should be read in full. Be inclusive:
the full read decides. Give a one-sentence reason.

""" + json.dumps(papers, ensure_ascii=False)


def field_spec(registry):
    fields = {}
    for kind, model in registry.items():
        fields[kind] = {
            field.name: (
                f"reference to {field.related_model._meta.model_name}: use $key for a record in this response, otherwise an existing exact name"
                if isinstance(field, models.ForeignKey) else
                f"{field.get_internal_type()}; {'nullable' if field.null or field.blank else 'required'}"
                + (f"; one of {[value for value, _ in field.choices]}" if field.choices else "")
            )
            for field in model._meta.fields if not field.primary_key and field.name != "source_record"
        }
    fields["category"] = {"scope": "model_type or pretrain_method", "value": "new category name"}
    fields["dataset"]["tasks"] = "array of exact task names"
    fields["head"]["tasks"] = "array of exact task names"
    fields["fps"]["owner"] = "$key of backbone, instance, or semantic record"
    return fields


RULES = """
Output format:
- Return flat records with unique keys. References to other records in this response
  use $key; references to existing entities use their exact name from the vocabulary.
- Each fields entry has a name and a scalar value (or an array for tasks). Use null for
  unknown optional fields. Omit id and audit fields.
- Every non-null factual value needs an evidence entry: a verbatim quote from the PDF
  text, its 1-based PDF page (url null), and the table/row/column or section. A number's
  quote must contain that number (quote the table row, not only its caption). A value
  may have more than one evidence entry (e.g. a table cell and the setup section).
  Record names you construct by the naming conventions, and $key references, need no
  evidence. Every field naming an existing head, dataset or task (head, dataset,
  train_dataset, pretrain_dataset, fine_tune_dataset, instance_type, ...) must either be
  quoted where the paper states it or be listed in inferred with its source; the same
  applies to classifications you make by the guide (model_type, hierarchical,
  pretrain_method). E.g. {"field": "head", "source": "follows the ConvNeXt UPerNet setup,
  Sec. 4.2"}. Paper URLs and the family pub_date may be omitted: the importer attaches
  this exact arXiv version and its date.
- Put each judgment call in the record's note: what you decided and why (e.g. "dense
  FLOPs used; paper headlines 5.0G sparsity-aware"). Leave note empty when nothing
  needed judgment.
- List a field in uncertain only when the sources do not give a required value, give
  contradictory values, or cannot be read reliably. A judgment call is a note, not an
  uncertainty. Any uncertain field sends the paper to human review.

Derived values:
- Derive values when the inputs are supported. For iterations_to_epochs use inputs
  named iterations, effective_batch_size and dataset_size; for multiply/divide, a and b.
  The importer recomputes the result. Use the global effective batch size, including
  accumulation. Do not round a fractional result.
- Cite each input. An input from the guide's conventions is uncited: set citation=null
  and source="convention: <name>" (e.g. "convention: ADE20K" for its dataset size,
  "convention: standard batch size" for 16), and mention it in the note. Any other
  uncited input needs its source and an explicit assumption, and sends the paper to
  review. Do not invent supporting quotations.
- A named schedule (1x, 2x, 3x, 6x) is cited by its quote; its epochs follow the guide.

New entities:
- A new model_type or pretrain_method needs a category record with scope and value.
- A new dataset or head needs its own record and tasks. New categories, datasets and
  heads are proposals the maintainer must approve; propose one only when nothing in
  the vocabulary describes the same entity.

GFLOPs: classification is backbone-only; downstream is backbone+head. Never substitute
one for the other. ImageNet-C top_1 means mCE and ImageNet-C-bar top_1 means CE. Keep
box and mask AP distinct (instance_type 'Object Detection' or 'Instance Segmentation').
"""


def guide_block():
    return """
Follow the maintainer's data entry guide below. It records the conventions already used
in the database; where it does not cover a case, use your judgment in its spirit.

=== DATA ENTRY GUIDE ===
""" + guide() + """
=== END OF GUIDE ===
"""

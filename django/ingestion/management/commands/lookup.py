import json
from pathlib import Path

import yaml
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q

from ingestion.evaluate import hidden_vocabulary, no_database_writes
from ingestion.family_yaml import family_to_dict, pick, CLASSIFICATION_FIELDS, INSTANCE_FIELDS, SEMANTIC_FIELDS
from stats.models import (
    Backbone, BackboneFamily, ClassificationResult, Dataset, DownstreamHead, InstanceResult, PretrainedBackbone,
    SemanticSegmentationResult,
)


class Command(BaseCommand):
    help = "Read-only database lookups for the extraction agent (never writes)."

    def add_arguments(self, parser):
        parser.add_argument("what", choices=["names", "family", "model", "search"])
        parser.add_argument("query", nargs="?", default="")
        parser.add_argument("--work", help="agent working folder (families hidden for evals)")

    def handle(self, *args, **options):
        hidden = set()
        if options["work"] and (Path(options["work"]) / "hidden.json").exists():
            hidden = set(json.loads((Path(options["work"]) / "hidden.json").read_text()))
        with no_database_writes():
            families = BackboneFamily.objects.exclude(name__in=hidden)
            what, query = options["what"], options["query"].strip()
            if what == "names":
                self.stdout.write(json.dumps(hidden_vocabulary(list(BackboneFamily.objects.filter(name__in=hidden))),
                                             indent=1, ensure_ascii=False))
            elif what == "family":
                family = families.filter(name=query).first()
                if family is None:
                    raise CommandError(f"No family named {query!r}; try ./hb lookup search")
                self.stdout.write(yaml.dump(family_to_dict(family), sort_keys=False, allow_unicode=True))
            elif what == "model":
                self.model(query, families)
            else:
                self.search(query, families)

    def model(self, name, families):
        pretrained = list(PretrainedBackbone.objects.filter(family__in=families).filter(
            Q(name=name) | Q(backbone__name=name)).select_related("backbone", "family", "pretrain_dataset"))
        if not pretrained:
            raise CommandError(f"No pretrained backbone or backbone named {name!r}; try ./hb lookup search")
        out = []
        for item in pretrained:
            entry = {"pretrained_backbone": item.name, "backbone": item.backbone.name, "family": item.family.name,
                     "pretrain": {"dataset": item.pretrain_dataset.name, "method": item.pretrain_method,
                                  "epochs": item.pretrain_epochs, "resolution": item.pretrain_resolution}}
            for key, model, fields in [("classification", ClassificationResult, CLASSIFICATION_FIELDS),
                                       ("instance", InstanceResult, INSTANCE_FIELDS),
                                       ("semantic", SemanticSegmentationResult, SEMANTIC_FIELDS)]:
                entry[key] = [pick(result, fields + ["paper"], None)
                              for result in model.objects.filter(pretrained_backbone=item).order_by("pk")]
            out.append(entry)
        self.stdout.write(yaml.dump(out, sort_keys=False, allow_unicode=True))

    def search(self, text, families):
        if len(text) < 2:
            raise CommandError("Search needs at least two characters")
        found = {
            "families": list(families.filter(name__icontains=text).values_list("name", flat=True)),
            "backbones": list(Backbone.objects.filter(family__in=families, name__icontains=text).values_list("name", flat=True)),
            "pretrained_backbones": list(PretrainedBackbone.objects.filter(family__in=families, name__icontains=text)
                                         .values_list("name", flat=True)),
            "datasets": list(Dataset.objects.filter(name__icontains=text).values_list("name", flat=True)),
            "heads": list(DownstreamHead.objects.filter(name__icontains=text).values_list("name", flat=True)),
        }
        self.stdout.write(json.dumps(found, indent=1, ensure_ascii=False))

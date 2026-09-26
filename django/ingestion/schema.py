"""JSON schemas for screening decisions and the agent's extraction.json; model fields are
described in the prompt."""


def object_schema(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items):
    return {"type": "array", "items": items}


STRING = {"type": "string"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
VALUE = {"anyOf": [STRING, NUMBER, BOOLEAN, {"type": "null"}, array(STRING)]}
# A citation is to a PDF page, or (agent extractions) to a fetched web source's url.
CITATION = object_schema(page={"type": ["integer", "null"], "minimum": 1}, url={"type": ["string", "null"]},
                         location=STRING, quote=STRING)
INPUT = object_schema(name=STRING, value=NUMBER,
                      citation={"anyOf": [CITATION, {"type": "null"}]},
                      source={"type": ["string", "null"]})
DERIVATION = {"anyOf": [
    {"type": "null"},
    object_schema(method={"type": "string", "enum": ["iterations_to_epochs", "multiply", "divide"]},
                  inputs=array(INPUT), assumptions=array(STRING)),
]}
EVIDENCE = object_schema(field=STRING, citation=CITATION, derivation=DERIVATION)
DECISION = object_schema(
    qualifies=BOOLEAN, architecture_or_pretraining_contribution=BOOLEAN,
    imagenet_1k_results=BOOLEAN, downstream_results=BOOLEAN,
    reason=STRING, evidence=array(CITATION),
)
KINDS = ["category", "dataset", "head", "family", "backbone", "pretrained_backbone", "classification", "instance", "semantic", "fps"]
RECORD = object_schema(
    key=STRING, kind={"type": "string", "enum": KINDS},
    fields=array(object_schema(name=STRING, value=VALUE)), evidence=array(EVIDENCE),
    uncertain=array(object_schema(field=STRING, reason=STRING)),
    inferred=array(object_schema(field=STRING, source=STRING)),
    note=STRING,
)
EXTRACTION = object_schema(decision=DECISION, records=array(RECORD))
SCREENING = object_schema(decisions=array(object_schema(
    arxiv_id=STRING, qualifies=BOOLEAN, architecture_or_pretraining_contribution=BOOLEAN,
    imagenet_1k_results=BOOLEAN, downstream_results=BOOLEAN, reason=STRING,
)))

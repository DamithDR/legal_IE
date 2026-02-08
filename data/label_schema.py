"""Label schema for InLegalNER dataset — 14 entity types in BIO format."""

ENTITY_TYPES = [
    "COURT",
    "PETITIONER",
    "RESPONDENT",
    "JUDGE",
    "LAWYER",
    "DATE",
    "ORG",
    "GPE",
    "STATUTE",
    "PROVISION",
    "PRECEDENT",
    "CASE_NUMBER",
    "WITNESS",
    "OTHER_PERSON",
]

# BIO tag list — will be validated/overridden from dataset features on load
LABEL_LIST = ["O"]
for etype in ENTITY_TYPES:
    LABEL_LIST.append(f"B-{etype}")
    LABEL_LIST.append(f"I-{etype}")

label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}


def update_schema_from_dataset(feature_names: list[str]):
    """Override the label list with the authoritative ordering from the dataset."""
    global LABEL_LIST, label2id, id2label
    LABEL_LIST = list(feature_names)
    label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
    id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}

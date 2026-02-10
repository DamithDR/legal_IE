"""Label schema for NER datasets — supports InLegalNER (14), ICDAC (27), and IE4Wills (21 types)."""

# --- InLegalNER schema ---
INLEGALNER_ENTITY_TYPES = [
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

# --- ICDAC schema ---
ICDAC_ENTITY_TYPES = [
    "APPL",
    "APT",
    "AUTH",
    "BCH",
    "CD",
    "CIVL",
    "CRNL",
    "CRTOF",
    "DC",
    "DOC",
    "DOJUD",
    "HC",
    "JD",
    "JDG",
    "JUR-ADVSY",
    "JUR-APLT",
    "JUR-OGNL",
    "ORD",
    "ORG",
    "PC",
    "PETR",
    "PLC",
    "PT",
    "PTN",
    "RES",
    "SC",
    "STE",
]

# --- IE4Wills schema ---
IE4WILLS_ENTITY_TYPES = [
    "ASSET",
    "BENEFICIARY",
    "BOND",
    "CODICIL",
    "CONDITION",
    "COUNTY",
    "DATE",
    "DEBT",
    "DUTY",
    "EXECUTOR",
    "EXECUTOR1",
    "EXECUTOR2",
    "EXPENSE",
    "RIGHT",
    "STATE",
    "TAX",
    "TESTATOR",
    "TIME",
    "TRIGGER",
    "WILL",
    "WITNESS",
]

_SCHEMAS = {
    "inlegalner": INLEGALNER_ENTITY_TYPES,
    "icdac": ICDAC_ENTITY_TYPES,
    "ie4wills": IE4WILLS_ENTITY_TYPES,
}


def _build_label_list(entity_types):
    """Build BIO label list from entity types."""
    label_list = ["O"]
    for etype in entity_types:
        label_list.append(f"B-{etype}")
        label_list.append(f"I-{etype}")
    return label_list


def get_schema(dataset_name):
    """
    Return (LABEL_LIST, label2id, id2label, ENTITY_TYPES) for a given dataset.
    """
    entity_types = _SCHEMAS[dataset_name]
    label_list = _build_label_list(entity_types)
    l2id = {label: idx for idx, label in enumerate(label_list)}
    i2l = {idx: label for idx, label in enumerate(label_list)}
    return label_list, l2id, i2l, entity_types


# --- Active (module-level) schema — defaults to InLegalNER ---
ENTITY_TYPES = list(INLEGALNER_ENTITY_TYPES)
LABEL_LIST = _build_label_list(ENTITY_TYPES)
label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}


def set_active_schema(dataset_name):
    """Set the active module-level schema for a given dataset."""
    global ENTITY_TYPES, LABEL_LIST, label2id, id2label
    LABEL_LIST, label2id, id2label, ENTITY_TYPES = get_schema(dataset_name)


def update_schema_from_dataset(feature_names: list[str]):
    """Override the label list with the authoritative ordering from the dataset."""
    global LABEL_LIST, label2id, id2label
    LABEL_LIST = list(feature_names)
    label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
    id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}

"""Relation type schema for the RE task (IE4Wills dataset)."""

# The 4 core relation types from the original paper (Kwak et al., EMNLP 2023)
CORE_RELATION_TYPES = [
    "TESTATOR-BENEFICIARY",
    "TESTATOR-ASSET",
    "BENEFICIARY-ASSET",
    "TESTATOR-WILL",
]

# For BERT classification: 4 relation types + NO_RELATION
RE_LABEL_LIST = ["NO_RELATION"] + CORE_RELATION_TYPES

re_label2id = {label: idx for idx, label in enumerate(RE_LABEL_LIST)}
re_id2label = {idx: label for idx, label in enumerate(RE_LABEL_LIST)}

NUM_RELATION_LABELS = len(RE_LABEL_LIST)

# Entity types involved in the 4 core relations
RELEVANT_ENTITY_TYPES = {"TESTATOR", "BENEFICIARY", "ASSET", "WILL"}

# Expected (from_type, to_type) for each relation
RELATION_SIGNATURES = {
    "TESTATOR-BENEFICIARY": ("TESTATOR", "BENEFICIARY"),
    "TESTATOR-ASSET": ("TESTATOR", "ASSET"),
    "BENEFICIARY-ASSET": ("BENEFICIARY", "ASSET"),
    "TESTATOR-WILL": ("TESTATOR", "WILL"),
}

# Reverse lookup: (type1, type2) -> list of possible relation types
ENTITY_PAIR_TO_RELATIONS = {}
for rel, (t1, t2) in RELATION_SIGNATURES.items():
    ENTITY_PAIR_TO_RELATIONS.setdefault((t1, t2), []).append(rel)

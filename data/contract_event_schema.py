"""Label schema for Contractual Event Detection — BIO tagging of event triggers."""

# --- Contractual Events schema (US court decisions) ---
CONTRACT_EVENT_TYPES = [
    "Purchase-contract",
    "Other-contract",
    "Execution",
    "Agreement",
    "Rescission",
    "Negotiation",
    "Expiration",
]


def _build_label_list(event_types):
    """Build BIO label list from event types."""
    label_list = ["O"]
    for etype in event_types:
        label_list.append(f"B-{etype}")
        label_list.append(f"I-{etype}")
    return label_list


def get_contract_event_schema(dataset_name="contractual_events"):
    """
    Return (LABEL_LIST, label2id, id2label, EVENT_TYPES) for the contractual events dataset.
    """
    label_list = _build_label_list(CONTRACT_EVENT_TYPES)
    l2id = {label: idx for idx, label in enumerate(label_list)}
    i2l = {idx: label for idx, label in enumerate(label_list)}
    return label_list, l2id, i2l, CONTRACT_EVENT_TYPES


# --- Active (module-level) schema ---
EVENT_TYPES = list(CONTRACT_EVENT_TYPES)
LABEL_LIST = _build_label_list(EVENT_TYPES)
label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}


def set_active_contract_event_schema(dataset_name="contractual_events"):
    """Set the active module-level contract event schema."""
    global EVENT_TYPES, LABEL_LIST, label2id, id2label
    LABEL_LIST, label2id, id2label, EVENT_TYPES = get_contract_event_schema(dataset_name)

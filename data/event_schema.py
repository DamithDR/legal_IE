"""Label schema for Event Detection datasets — BIO tagging of event triggers."""

# --- Events Matter schema (ECHR court decisions) ---
EVENTS_MATTER_EVENT_TYPES = [
    "PROCEDURE",
    "CIRCUMSTANCE",
]

_SCHEMAS = {
    "events_matter": EVENTS_MATTER_EVENT_TYPES,
}


def _build_label_list(event_types):
    """Build BIO label list from event types."""
    label_list = ["O"]
    for etype in event_types:
        label_list.append(f"B-{etype}")
        label_list.append(f"I-{etype}")
    return label_list


def get_event_schema(dataset_name):
    """
    Return (LABEL_LIST, label2id, id2label, EVENT_TYPES) for a given dataset.
    """
    event_types = _SCHEMAS[dataset_name]
    label_list = _build_label_list(event_types)
    l2id = {label: idx for idx, label in enumerate(label_list)}
    i2l = {idx: label for idx, label in enumerate(label_list)}
    return label_list, l2id, i2l, event_types


# --- Active (module-level) schema — defaults to Events Matter ---
EVENT_TYPES = list(EVENTS_MATTER_EVENT_TYPES)
LABEL_LIST = _build_label_list(EVENT_TYPES)
label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
id2label = {idx: label for idx, label in enumerate(LABEL_LIST)}


def set_active_event_schema(dataset_name):
    """Set the active module-level event schema for a given dataset."""
    global EVENT_TYPES, LABEL_LIST, label2id, id2label
    LABEL_LIST, label2id, id2label, EVENT_TYPES = get_event_schema(dataset_name)

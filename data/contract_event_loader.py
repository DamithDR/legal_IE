"""Dataset loading for Contractual Events in Court Decisions (BRAT format).

Parses BRAT .ann + .txt files, merges annotations from two annotators via
majority voting (intersection), segments into sentences, and converts to
BIO-tagged sequences for event trigger detection.
"""

import re
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

import config
from data import contract_event_schema


# ===========================================================================
# BRAT parsing
# ===========================================================================

def _parse_brat_ann(ann_path: Path) -> list[dict]:
    """
    Parse a BRAT .ann file and extract event trigger spans with their types.

    In BRAT format:
        T1  Purchase-contract 3076 3080  sold       (text-bound annotation)
        E1  Purchase-contract:T1 Seller:T4 ...      (event annotation)

    We extract triggers by finding T-annotations that are referenced as the
    main trigger in E-annotations. The event type comes from the E-annotation.

    Returns:
        List of dicts with 'start', 'end', 'event_type', 'text'.
    """
    content = ann_path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    # Parse T-annotations (text-bound)
    t_annotations = {}  # T_id -> {type, start, end, text}
    # Parse E-annotations (events)
    e_annotations = []  # list of {event_type, trigger_id, args}

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")

        if line.startswith("T") and len(parts) >= 3:
            t_id = parts[0]
            type_and_offsets = parts[1].split()
            if len(type_and_offsets) >= 3:
                t_type = type_and_offsets[0]
                try:
                    start = int(type_and_offsets[1])
                    end = int(type_and_offsets[2])
                except ValueError:
                    continue
                t_annotations[t_id] = {
                    "type": t_type,
                    "start": start,
                    "end": end,
                    "text": parts[2] if len(parts) > 2 else "",
                }

        elif line.startswith("E") and len(parts) >= 2:
            args = parts[1].split()
            if args:
                # First arg is EventType:TriggerID
                event_ref = args[0]
                if ":" in event_ref:
                    event_type, trigger_id = event_ref.split(":", 1)
                    e_annotations.append({
                        "event_type": event_type,
                        "trigger_id": trigger_id,
                    })

    # Extract triggers: T-annotations referenced as event triggers
    triggers = []
    for e in e_annotations:
        trigger_id = e["trigger_id"]
        if trigger_id in t_annotations:
            t = t_annotations[trigger_id]
            event_type = e["event_type"]
            if event_type in contract_event_schema.CONTRACT_EVENT_TYPES:
                triggers.append({
                    "start": t["start"],
                    "end": t["end"],
                    "event_type": event_type,
                    "text": t["text"],
                })

    return triggers


def _merge_annotations_intersection(
    triggers_a: list[dict],
    triggers_b: list[dict],
    tolerance: int = 2,
) -> list[dict]:
    """
    Merge two annotators' triggers using intersection (majority voting).

    A trigger is kept only if both annotators agree — matched by overlapping
    character spans (within tolerance) and same event type.

    Args:
        triggers_a: Triggers from annotator-0.
        triggers_b: Triggers from annotator-5.
        tolerance: Max char offset difference for span matching.

    Returns:
        List of agreed-upon triggers (using annotator-0's span offsets).
    """
    merged = []
    used_b = set()

    for ta in triggers_a:
        for j, tb in enumerate(triggers_b):
            if j in used_b:
                continue
            if ta["event_type"] != tb["event_type"]:
                continue
            if (abs(ta["start"] - tb["start"]) <= tolerance and
                    abs(ta["end"] - tb["end"]) <= tolerance):
                merged.append(ta)
                used_b.add(j)
                break

    return merged


def _load_document_triggers(data_dir: Path) -> dict[str, list[dict]]:
    """
    Load event triggers for all documents using annotator-5 (most annotations).

    Returns:
        Dict mapping file_id to list of trigger dicts.
    """
    ann_dir = data_dir / "annotator-5"

    ann_files = {f.stem: f for f in ann_dir.glob("*.ann")} if ann_dir.exists() else {}

    doc_triggers = {}
    total_triggers = 0

    for file_id in sorted(ann_files.keys()):
        triggers = _parse_brat_ann(ann_files[file_id])
        doc_triggers[file_id] = triggers
        total_triggers += len(triggers)

    print(f"  Loaded annotations from annotator-5: "
          f"{len(doc_triggers)} documents, {total_triggers} total triggers")

    return doc_triggers


# ===========================================================================
# Sentence segmentation and BIO conversion
# ===========================================================================

def _split_into_sentences(text: str, offset: int) -> list[dict]:
    """
    Split a paragraph into sentences using punctuation boundaries.

    Returns:
        List of dicts with 'text', 'start', 'end'.
    """
    pattern = r'(?<=[.!?])\s+(?=[A-Z"\(])'
    parts = re.split(pattern, text)

    sentences = []
    current_pos = 0
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            current_pos += len(part)
            continue

        idx = text.find(part, current_pos)
        if idx == -1:
            idx = current_pos

        sentences.append({
            "text": part_stripped,
            "start": offset + idx,
            "end": offset + idx + len(part_stripped),
        })
        current_pos = idx + len(part)

    return sentences


def _text_to_sentences(text: str) -> list[dict]:
    """
    Split document text into sentence-level segments.

    Returns:
        List of dicts with 'text', 'start', 'end'.
    """
    segments = []
    lines = text.split("\n")
    current_pos = 0

    for line in lines:
        stripped = line.strip()
        if stripped:
            line_start = text.find(line, current_pos)
            if line_start == -1:
                line_start = current_pos

            sents = _split_into_sentences(stripped, line_start)
            segments.extend(sents)

        current_pos += len(line) + 1  # +1 for newline

    # Filter out very short segments (< 3 tokens)
    filtered = [s for s in segments if len(s["text"].split()) >= 3]
    return filtered


def _word_tokenize_with_offsets(text: str) -> list[dict]:
    """Tokenize text into words with character offsets."""
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def _sentences_to_bio(sentences: list[dict], triggers: list[dict]) -> list[dict]:
    """
    Convert sentences to BIO-tagged token sequences.

    Args:
        sentences: List of dicts with 'text', 'start', 'end'.
        triggers: List of dicts with 'start', 'end', 'event_type'.

    Returns:
        List of dicts with 'tokens' (list[str]) and 'event_tags' (list[int]).
    """
    examples = []
    for sent in sentences:
        sent_start = sent["start"]
        token_infos = _word_tokenize_with_offsets(sent["text"])

        if not token_infos:
            continue

        tokens = [t["token"] for t in token_infos]
        tags = ["O"] * len(tokens)

        for trigger in triggers:
            trig_start = trigger["start"]
            trig_end = trigger["end"]
            etype = trigger["event_type"]

            # Check if trigger overlaps with sentence
            if trig_end <= sent_start or trig_start >= sent["end"]:
                continue

            first = True
            for i, tok in enumerate(token_infos):
                tok_abs_start = sent_start + tok["start"]
                tok_abs_end = sent_start + tok["end"]

                if tok_abs_end > trig_start and tok_abs_start < trig_end:
                    if tags[i] == "O":
                        if first:
                            tags[i] = f"B-{etype}"
                            first = False
                        else:
                            tags[i] = f"I-{etype}"

        tag_ids = [contract_event_schema.label2id.get(tag, 0) for tag in tags]
        examples.append({"tokens": tokens, "event_tags": tag_ids})

    return examples


# ===========================================================================
# Dataset loading
# ===========================================================================

def load_contract_event_dataset(dataset_name="contractual_events"):
    """
    Load the Contractual Events dataset from BRAT files.

    Parses annotations, merges via intersection, segments into sentences,
    converts to BIO tags, and splits 80/10/10 at document level.

    Returns:
        Tuple of (DatasetDict{train, validation, test}, tag_column_name).
    """
    contract_event_schema.set_active_contract_event_schema(dataset_name)

    data_dir = config.CONTRACTUAL_EVENTS_DIR
    text_dir = data_dir / "text"
    print(f"Loading Contractual Events dataset from {data_dir}...")

    # Load and merge triggers
    doc_triggers = _load_document_triggers(data_dir)

    # Process each document into sentence-level examples
    doc_examples = {}  # file_id -> list of examples
    for file_id, triggers in doc_triggers.items():
        txt_path = text_dir / f"{file_id}.txt"
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        sentences = _text_to_sentences(text)
        examples = _sentences_to_bio(sentences, triggers)
        if examples:
            doc_examples[file_id] = examples

    print(f"  Processed {len(doc_examples)} documents, "
          f"{sum(len(e) for e in doc_examples.values())} sentences")

    # Split at document level: 80/10/10
    doc_ids = sorted(doc_examples.keys())
    random.seed(config.CONTRACTUAL_EVENTS_SPLIT_SEED)
    random.shuffle(doc_ids)

    n = len(doc_ids)
    train_end = int(n * 0.8)
    dev_end = train_end + int(n * 0.1)

    train_ids = doc_ids[:train_end]
    dev_ids = doc_ids[train_end:dev_end]
    test_ids = doc_ids[dev_end:]

    def _collect(ids):
        records = {"tokens": [], "event_tags": []}
        for fid in ids:
            for ex in doc_examples.get(fid, []):
                records["tokens"].append(ex["tokens"])
                records["event_tags"].append(ex["event_tags"])
        return records

    dataset = DatasetDict({
        "train": Dataset.from_dict(_collect(train_ids)),
        "validation": Dataset.from_dict(_collect(dev_ids)),
        "test": Dataset.from_dict(_collect(test_ids)),
    })

    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    tag_column = "event_tags"
    print(f"Using {len(contract_event_schema.LABEL_LIST)} labels: "
          f"{contract_event_schema.LABEL_LIST}")

    return dataset, tag_column


# ===========================================================================
# LLM sample utilities
# ===========================================================================

def get_contract_event_llm_samples(dataset, split="test", n=None, seed=42,
                                    tag_column="event_tags"):
    """Get samples for LLM evaluation."""
    split_data = dataset[split]

    if n is not None and n < len(split_data):
        indices = list(range(len(split_data)))
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:n]
        split_data = split_data.select(indices)

    samples = []
    for example in split_data:
        tokens = example["tokens"]
        tag_ids = example[tag_column]
        tags = [contract_event_schema.id2label[t] for t in tag_ids]
        samples.append({"tokens": tokens, "tags": tags})

    return samples


def get_contract_event_few_shot_examples(dataset, n=5, seed=42,
                                          tag_column="event_tags"):
    """
    Select diverse few-shot examples from training set using greedy set-cover
    to maximize event type coverage. Prefers shorter examples.
    """
    train_data = dataset["train"]

    example_event_types = []
    for example in train_data:
        tag_ids = example[tag_column]
        tags = [contract_event_schema.id2label[t] for t in tag_ids]
        etypes = set()
        for tag in tags:
            if tag.startswith("B-"):
                etypes.add(tag[2:])
        example_event_types.append(etypes)

    selected_indices = []
    covered_types = set()

    for _ in range(n):
        best_idx = None
        best_new_types = -1
        best_length = float("inf")

        for idx, etypes in enumerate(example_event_types):
            if idx in selected_indices:
                continue
            new_types = len(etypes - covered_types)
            length = len(train_data[idx]["tokens"])

            if new_types > best_new_types or (
                new_types == best_new_types and length < best_length
            ):
                if length <= 80 or best_idx is None:
                    best_idx = idx
                    best_new_types = new_types
                    best_length = length

        if best_idx is not None:
            selected_indices.append(best_idx)
            covered_types.update(example_event_types[best_idx])

    examples = []
    for idx in selected_indices:
        example = train_data[idx]
        tokens = example["tokens"]
        tag_ids = example[tag_column]
        tags = [contract_event_schema.id2label[t] for t in tag_ids]
        examples.append({"tokens": tokens, "tags": tags})

    print(f"Selected {len(examples)} few-shot examples covering "
          f"{len(covered_types)}/{len(contract_event_schema.EVENT_TYPES)} event types")
    return examples


def print_contract_event_dataset_stats(dataset, tag_column="event_tags"):
    """Print event type distribution across splits."""
    for split_name, split_data in dataset.items():
        counter = Counter()
        total_tokens = 0
        for example in split_data:
            tag_ids = example[tag_column]
            tags = [contract_event_schema.id2label[t] for t in tag_ids]
            total_tokens += len(tags)
            for tag in tags:
                if tag.startswith("B-"):
                    counter[tag[2:]] += 1

        total_events = sum(counter.values())
        print(f"\n{split_name} split - {len(split_data)} examples, "
              f"{total_tokens} tokens, {total_events} event triggers")
        print(f"  Event type counts:")
        for etype, count in counter.most_common():
            print(f"    {etype}: {count}")

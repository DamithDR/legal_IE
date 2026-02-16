"""Dataset loading and preprocessing for Event Detection (Events Matter corpus).

Parses GATE XML files containing ECHR court decisions with event annotations.
Extracts Event_what trigger spans, links them to parent Event types, and
converts to sentence-level BIO-tagged sequences.
"""

import re
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

import config
from data import event_schema


# ===========================================================================
# GATE XML parsing
# ===========================================================================

def _parse_gate_xml(filepath: Path):
    """
    Parse a GATE XML file and extract raw text with character-offset mapping.

    The <TextWithNodes> element contains the document text interspersed with
    <Node id="N"/> tags that define character offsets. We strip the Node tags
    to get clean text and build a mapping from Node IDs to character positions
    in the clean text.

    Returns:
        Tuple of (clean_text, node_id_to_char_offset dict).
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    text_with_nodes = root.find("TextWithNodes")
    if text_with_nodes is None:
        return "", {}

    # Build the clean text and node-to-offset mapping
    node_to_offset = {}
    parts = []
    current_pos = 0

    # The text content is interleaved: text, <Node id="X"/>, text, <Node id="Y"/>, ...
    # We need to handle the initial text, then iterate through child elements
    if text_with_nodes.text:
        parts.append(text_with_nodes.text)
        current_pos += len(text_with_nodes.text)

    for child in text_with_nodes:
        # Record the node's position in the clean text
        node_id = child.get("id")
        if node_id is not None:
            node_to_offset[int(node_id)] = current_pos

        # The tail text after this <Node/> tag
        if child.tail:
            parts.append(child.tail)
            current_pos += len(child.tail)

    clean_text = "".join(parts)
    return clean_text, node_to_offset


def _extract_events(root):
    """
    Extract event annotations from the consensus annotation set.

    Groups Event, Event_what, Event_who, Event_when by their shared 'id' field.
    Returns Event_what trigger spans with their parent Event type.

    Returns:
        List of dicts with 'start_node', 'end_node', 'event_type' for each trigger.
    """
    consensus_set = None
    for ann_set in root.findall("AnnotationSet"):
        if ann_set.get("Name") == "consensus":
            consensus_set = ann_set
            break

    if consensus_set is None:
        return []

    # Group annotations by id
    events_by_id = {}  # id -> Event annotation (has 'type')
    triggers_by_id = defaultdict(list)  # id -> list of Event_what annotations

    for ann in consensus_set.findall("Annotation"):
        ann_type = ann.get("Type")
        start_node = int(ann.get("StartNode"))
        end_node = int(ann.get("EndNode"))

        # Extract features
        features = {}
        for feat in ann.findall("Feature"):
            name_el = feat.find("Name")
            value_el = feat.find("Value")
            if name_el is not None and value_el is not None:
                features[name_el.text] = value_el.text

        ann_id = features.get("id")
        if ann_id is None:
            continue

        if ann_type == "Event":
            event_type = features.get("type", "").upper()
            if event_type in event_schema.EVENT_TYPES:
                events_by_id[ann_id] = {
                    "start_node": start_node,
                    "end_node": end_node,
                    "event_type": event_type,
                }
        elif ann_type == "Event_what":
            triggers_by_id[ann_id].append({
                "start_node": start_node,
                "end_node": end_node,
            })

    # Link triggers to parent events
    trigger_spans = []
    for event_id, trigger_list in triggers_by_id.items():
        parent = events_by_id.get(event_id)
        if parent is None:
            continue
        for trigger in trigger_list:
            trigger_spans.append({
                "start_node": trigger["start_node"],
                "end_node": trigger["end_node"],
                "event_type": parent["event_type"],
            })

    # Deduplicate by (start_node, end_node, event_type)
    seen = set()
    deduped = []
    for t in trigger_spans:
        key = (t["start_node"], t["end_node"], t["event_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


def _text_to_sentences(text: str) -> list[dict]:
    """
    Split document text into sentence-level segments.

    Uses paragraph structure (numbered paragraphs like "5. ", "6. ") and
    sentence-ending punctuation to segment the text.

    Returns:
        List of dicts with 'text', 'start' (char offset), 'end' (char offset).
    """
    # Split on paragraph boundaries: numbered paragraphs, section headers, double newlines
    # First split on double newlines or numbered paragraph starts
    segments = []

    # Split on newlines to get paragraphs
    lines = text.split("\n")
    current_pos = 0

    for line in lines:
        stripped = line.strip()
        if stripped:
            # Find the actual start position in the original text
            line_start = text.find(line, current_pos)
            if line_start == -1:
                line_start = current_pos

            # Further split long paragraphs into sentences
            sents = _split_into_sentences(stripped, line_start)
            segments.extend(sents)

        current_pos += len(line) + 1  # +1 for the newline

    # Filter out very short segments (< 3 tokens)
    filtered = [s for s in segments if len(s["text"].split()) >= 3]

    return filtered


def _split_into_sentences(text: str, offset: int) -> list[dict]:
    """
    Split a paragraph into sentences using punctuation boundaries.

    Returns:
        List of dicts with 'text', 'start', 'end'.
    """
    # Split on sentence-ending punctuation followed by space and uppercase
    # but be careful not to split on abbreviations like "no." or "v."
    pattern = r'(?<=[.!?])\s+(?=[A-Z"\(])'
    parts = re.split(pattern, text)

    sentences = []
    current_pos = 0
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            current_pos += len(part)
            continue

        # Find actual position in the paragraph
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


def _word_tokenize_with_offsets(text: str) -> list[dict]:
    """
    Tokenize text into words, tracking character offsets.

    Returns:
        List of dicts with 'token', 'start', 'end' keys.
    """
    tokens = []
    for match in re.finditer(r"\S+", text):
        tokens.append({
            "token": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def _sentences_to_bio(sentences: list[dict], trigger_spans: list[dict],
                      node_to_offset: dict) -> list[dict]:
    """
    Convert sentences to BIO-tagged token sequences.

    Args:
        sentences: List of dicts with 'text', 'start', 'end'.
        trigger_spans: List of dicts with 'start_node', 'end_node', 'event_type'.
        node_to_offset: Mapping from GATE node IDs to character offsets.

    Returns:
        List of dicts with 'tokens' (list[str]) and 'event_tags' (list[int]).
    """
    # Convert node-based spans to character offsets
    char_triggers = []
    for t in trigger_spans:
        start_char = node_to_offset.get(t["start_node"])
        end_char = node_to_offset.get(t["end_node"])
        if start_char is not None and end_char is not None:
            char_triggers.append({
                "start": start_char,
                "end": end_char,
                "event_type": t["event_type"],
            })

    examples = []
    for sent in sentences:
        sent_start = sent["start"]
        token_infos = _word_tokenize_with_offsets(sent["text"])

        if not token_infos:
            continue

        tokens = [t["token"] for t in token_infos]
        tags = ["O"] * len(tokens)

        # Find triggers that overlap with this sentence
        for trigger in char_triggers:
            trig_start = trigger["start"]
            trig_end = trigger["end"]
            etype = trigger["event_type"]

            # Check if trigger overlaps with sentence
            if trig_end <= sent_start or trig_start >= sent["end"]:
                continue

            # Map trigger to token indices within this sentence
            first = True
            for i, tok in enumerate(token_infos):
                # Token absolute positions
                tok_abs_start = sent_start + tok["start"]
                tok_abs_end = sent_start + tok["end"]

                # Check overlap
                if tok_abs_end > trig_start and tok_abs_start < trig_end:
                    if tags[i] == "O":  # Don't overwrite existing tags
                        if first:
                            tags[i] = f"B-{etype}"
                            first = False
                        else:
                            tags[i] = f"I-{etype}"

        tag_ids = [event_schema.label2id.get(tag, 0) for tag in tags]
        examples.append({"tokens": tokens, "event_tags": tag_ids})

    return examples


# ===========================================================================
# Dataset loading
# ===========================================================================

def _load_split(split_dir: Path) -> list[dict]:
    """Load all GATE XML files from a split directory."""
    all_examples = []

    xml_files = sorted(split_dir.glob("*.xml"))
    print(f"  Loading {len(xml_files)} files from {split_dir.name}/")

    for filepath in xml_files:
        if filepath.stat().st_size == 0:
            print(f"    Skipping empty file: {filepath.name}")
            continue

        tree = ET.parse(filepath)
        root = tree.getroot()

        clean_text, node_to_offset = _parse_gate_xml(filepath)
        if not clean_text:
            print(f"    Skipping empty document: {filepath.name}")
            continue

        trigger_spans = _extract_events(root)
        sentences = _text_to_sentences(clean_text)
        examples = _sentences_to_bio(sentences, trigger_spans, node_to_offset)

        all_examples.extend(examples)
        print(f"    {filepath.name}: {len(sentences)} sentences, "
              f"{len(trigger_spans)} triggers, {len(examples)} examples")

    return all_examples


def load_event_dataset(dataset_name="events_matter"):
    """
    Load an event detection dataset.

    Args:
        dataset_name: 'events_matter'.

    Returns:
        Tuple of (DatasetDict, tag_column_name).
    """
    if dataset_name != "events_matter":
        raise ValueError(f"Unknown event dataset: {dataset_name}. Only 'events_matter' is supported.")

    event_schema.set_active_event_schema(dataset_name)

    data_dir = config.EVENTS_MATTER_DIR
    print(f"Loading Events Matter dataset from {data_dir}...")

    splits = {}
    for split_name, dir_name in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        split_dir = data_dir / dir_name
        examples = _load_split(split_dir)

        records = {
            "tokens": [e["tokens"] for e in examples],
            "event_tags": [e["event_tags"] for e in examples],
        }
        splits[split_name] = Dataset.from_dict(records)
        print(f"  {split_name}: {len(splits[split_name])} examples")

    dataset = DatasetDict(splits)
    tag_column = "event_tags"
    print(f"Using {len(event_schema.LABEL_LIST)} labels: {event_schema.LABEL_LIST}")

    return dataset, tag_column


# ===========================================================================
# LLM sample utilities
# ===========================================================================

def get_event_llm_samples(dataset, split="test", n=None, seed=42, tag_column="event_tags"):
    """
    Get samples for LLM evaluation.

    Returns:
        List of dicts with 'tokens' and 'tags' (string BIO tags).
    """
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
        tags = [event_schema.id2label[t] for t in tag_ids]
        samples.append({"tokens": tokens, "tags": tags})

    return samples


def get_event_few_shot_examples(dataset, n=5, seed=42, tag_column="event_tags"):
    """
    Select diverse few-shot examples from training set using greedy set-cover
    to maximize event type coverage.
    """
    train_data = dataset["train"]

    # Pre-compute event types present in each example
    example_event_types = []
    for example in train_data:
        tag_ids = example[tag_column]
        tags = [event_schema.id2label[t] for t in tag_ids]
        etypes = set()
        for tag in tags:
            if tag.startswith("B-"):
                etypes.add(tag[2:])
        example_event_types.append(etypes)

    # Greedy set-cover
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
        tags = [event_schema.id2label[t] for t in tag_ids]
        examples.append({"tokens": tokens, "tags": tags})

    print(f"Selected {len(examples)} few-shot examples covering "
          f"{len(covered_types)}/{len(event_schema.EVENT_TYPES)} event types")
    return examples


def print_event_dataset_stats(dataset, tag_column="event_tags"):
    """Print event type distribution across splits."""
    for split_name, split_data in dataset.items():
        counter = Counter()
        total_tokens = 0
        for example in split_data:
            tag_ids = example[tag_column]
            tags = [event_schema.id2label[t] for t in tag_ids]
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

"""Dataset loading and preprocessing for NER datasets."""

import json
import random
import re
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

import config
from data import label_schema


# ===========================================================================
# InLegalNER loader (span-annotation format)
# ===========================================================================

def _word_tokenize(text: str) -> list[dict]:
    """
    Simple whitespace/punctuation tokenizer that tracks character offsets.

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


def _spans_to_bio(token_infos: list[dict], annotations: list[dict]) -> list[str]:
    """
    Convert character-level span annotations to token-level BIO tags.

    Args:
        token_infos: List of dicts with 'token', 'start', 'end'.
        annotations: List of entity annotations with 'start', 'end', 'labels'.

    Returns:
        List of BIO tag strings, one per token.
    """
    tags = ["O"] * len(token_infos)

    for ann in annotations:
        ent_start = ann["start"]
        ent_end = ann["end"]
        ent_labels = ann["labels"]
        if not ent_labels:
            continue
        ent_type = ent_labels[0]

        if ent_type not in label_schema.ENTITY_TYPES:
            continue

        # Find tokens that overlap with this entity span
        first = True
        for i, tok in enumerate(token_infos):
            # Token overlaps with entity if token doesn't end before entity starts
            # and token doesn't start after entity ends
            if tok["end"] > ent_start and tok["start"] < ent_end:
                if tags[i] == "O":  # Don't overwrite existing tags
                    if first:
                        tags[i] = f"B-{ent_type}"
                        first = False
                    else:
                        tags[i] = f"I-{ent_type}"

    return tags


def _convert_example(example: dict) -> dict:
    """
    Convert a single span-annotated example to token + BIO tag format.

    Returns:
        Dict with 'tokens' (list[str]) and 'ner_tags' (list[int]).
    """
    text = example["data"]["text"]
    token_infos = _word_tokenize(text)
    tokens = [t["token"] for t in token_infos]

    # Extract annotations
    annotations = []
    for ann_set in example["annotations"]:
        for result in ann_set["result"]:
            if result["type"] == "labels":
                annotations.append(result["value"])

    bio_tags = _spans_to_bio(token_infos, annotations)
    tag_ids = [label_schema.label2id.get(tag, 0) for tag in bio_tags]

    return {"tokens": tokens, "ner_tags": tag_ids}


def _load_inlegalner_dataset():
    """
    Load the InLegalNER dataset and convert from span annotations to token+BIO format.

    Returns:
        Tuple of (converted DatasetDict, tag_column_name).
    """
    print("Loading InLegalNER dataset...")
    raw_dataset = load_dataset(config.DATASET_NAME)

    print("Converting span annotations to BIO tags...")
    converted = {}
    for split_name, split_data in raw_dataset.items():
        records = {"tokens": [], "ner_tags": []}
        for example in split_data:
            result = _convert_example(example)
            records["tokens"].append(result["tokens"])
            records["ner_tags"].append(result["ner_tags"])

        converted[split_name] = Dataset.from_dict(records)
        print(f"  {split_name}: {len(converted[split_name])} examples")

    dataset = DatasetDict(converted)

    tag_column = "ner_tags"
    print(f"Using {len(label_schema.LABEL_LIST)} labels: {label_schema.LABEL_LIST}")

    return dataset, tag_column


# ===========================================================================
# ICDAC loader (Excel with BOI Tag column)
# ===========================================================================

# Tag cleaning rules for ICDAC
_ICDAC_TAG_FIXES = {
    "B- DOJUD": "B-DOJUD",
    "CRTOF": "B-CRTOF",
    "I-AAPL": "I-APPL",
    "I-BD": "O",
}


def _clean_icdac_tag(tag):
    """Strip whitespace and apply known fixes to a raw ICDAC tag."""
    if not isinstance(tag, str):
        return None  # NaN — sentence boundary
    tag = tag.strip()
    if tag in _ICDAC_TAG_FIXES:
        tag = _ICDAC_TAG_FIXES[tag]
    return tag


def _load_icdac_dataset():
    """
    Load ICDAC dataset from Excel, clean tags, split into sentences,
    and create train/dev/test splits.

    Returns:
        Tuple of (DatasetDict, tag_column_name).
    """
    print(f"Loading ICDAC dataset from {config.ICDAC_EXCEL_PATH}...")
    df = pd.read_excel(config.ICDAC_EXCEL_PATH, engine="openpyxl")

    # Clean tags
    df["clean_tag"] = df["BOI Tag"].apply(_clean_icdac_tag)

    # Split into sentences on NaN rows (rows where Token is NaN)
    sentences = []
    current_tokens = []
    current_tags = []

    for _, row in df.iterrows():
        token = row.get("Token")
        tag = row["clean_tag"]

        if pd.isna(token) or tag is None:
            # Sentence boundary
            if current_tokens:
                sentences.append({"tokens": current_tokens, "tags": current_tags})
                current_tokens = []
                current_tags = []
        else:
            current_tokens.append(str(token))
            current_tags.append(tag)

    # Don't forget the last sentence
    if current_tokens:
        sentences.append({"tokens": current_tokens, "tags": current_tags})

    print(f"  Parsed {len(sentences)} sentences, {sum(len(s['tokens']) for s in sentences)} tokens")

    # Convert string tags to integer IDs
    records = {"tokens": [], "ner_tags": []}
    skipped_tags = Counter()
    for sent in sentences:
        tag_ids = []
        for tag in sent["tags"]:
            tid = label_schema.label2id.get(tag)
            if tid is None:
                skipped_tags[tag] += 1
                tid = 0  # fallback to O
            tag_ids.append(tid)
        records["tokens"].append(sent["tokens"])
        records["ner_tags"].append(tag_ids)

    if skipped_tags:
        print(f"  Warning: unknown tags mapped to O: {dict(skipped_tags)}")

    # Shuffle and split 80/10/10
    n = len(records["tokens"])
    indices = list(range(n))
    random.seed(config.ICDAC_SPLIT_SEED)
    random.shuffle(indices)

    train_ratio, dev_ratio, _ = config.ICDAC_SPLIT_RATIOS
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    def _select(indices_subset):
        return {
            "tokens": [records["tokens"][i] for i in indices_subset],
            "ner_tags": [records["ner_tags"][i] for i in indices_subset],
        }

    train_indices = indices[:train_end]
    dev_indices = indices[train_end:dev_end]
    test_indices = indices[dev_end:]

    dataset = DatasetDict({
        "train": Dataset.from_dict(_select(train_indices)),
        "validation": Dataset.from_dict(_select(dev_indices)),
        "test": Dataset.from_dict(_select(test_indices)),
    })

    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    tag_column = "ner_tags"
    print(f"Using {len(label_schema.LABEL_LIST)} labels")

    return dataset, tag_column


# ===========================================================================
# IE4Wills loader (Label Studio JSON, local files)
# ===========================================================================

def _load_ie4wills_dataset():
    """
    Load the IE4Wills dataset from local JSON files and convert span annotations
    to token+BIO format.

    Expects train.json, dev.json, test.json in config.IE4WILLS_DATA_DIR.

    Returns:
        Tuple of (DatasetDict{train, validation, test}, tag_column_name).
    """
    data_dir = config.IE4WILLS_DATA_DIR
    print(f"Loading IE4Wills dataset from {data_dir}...")

    split_files = {
        "train": data_dir / "train.json",
        "validation": data_dir / "dev.json",
        "test": data_dir / "test.json",
    }

    converted = {}
    for split_name, filepath in split_files.items():
        with open(filepath, encoding="utf-8") as f:
            raw_examples = json.load(f)

        records = {"tokens": [], "ner_tags": []}
        for example in raw_examples:
            result = _convert_example(example)
            records["tokens"].append(result["tokens"])
            records["ner_tags"].append(result["ner_tags"])

        converted[split_name] = Dataset.from_dict(records)
        print(f"  {split_name}: {len(converted[split_name])} examples")

    dataset = DatasetDict(converted)
    tag_column = "ner_tags"
    print(f"Using {len(label_schema.LABEL_LIST)} labels")

    return dataset, tag_column


# ===========================================================================
# Unified dispatcher
# ===========================================================================

def load_ner_dataset(dataset_name="inlegalner"):
    """
    Load an NER dataset by name.

    Args:
        dataset_name: 'inlegalner' or 'icdac'.

    Returns:
        Tuple of (DatasetDict, tag_column_name).
    """
    if dataset_name == "inlegalner":
        return _load_inlegalner_dataset()
    elif dataset_name == "icdac":
        return _load_icdac_dataset()
    elif dataset_name == "ie4wills":
        return _load_ie4wills_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'inlegalner', 'icdac', or 'ie4wills'.")


def tokenize_and_align_labels(examples, tokenizer, max_length=512, tag_column="ner_tags"):
    """
    Tokenize pre-tokenized inputs and align BIO labels with subword tokens.

    For subword tokens beyond the first piece: assign -100 (ignored by loss).
    For special tokens ([CLS], [SEP], padding): assign -100.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels = []
    for i, labels in enumerate(examples[tag_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def get_llm_samples(dataset, split="test", n=None, seed=42, tag_column="ner_tags"):
    """
    Get samples for LLM evaluation.

    Args:
        dataset: Converted DatasetDict with 'tokens' and 'ner_tags'.
        split: Which split to sample from.
        n: Number of samples (None = full split).
        seed: Random seed for reproducibility.

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
        tags = [label_schema.id2label[t] for t in tag_ids]
        samples.append({"tokens": tokens, "tags": tags})

    return samples


def get_few_shot_examples(dataset, n=5, seed=42, tag_column="ner_tags"):
    """
    Select diverse few-shot examples from training set using greedy set-cover
    to maximize entity type coverage. Prefers shorter examples for prompt efficiency.

    Args:
        dataset: Converted DatasetDict.
        n: Number of examples to select.
        seed: Random seed.

    Returns:
        List of dicts with 'tokens' and 'tags' (string BIO tags).
    """
    train_data = dataset["train"]

    # Pre-compute entity types present in each example
    example_entity_types = []
    for example in train_data:
        tag_ids = example[tag_column]
        tags = [label_schema.id2label[t] for t in tag_ids]
        entity_types = set()
        for tag in tags:
            if tag.startswith("B-"):
                entity_types.add(tag[2:])
        example_entity_types.append(entity_types)

    # Greedy set-cover
    selected_indices = []
    covered_types = set()

    for _ in range(n):
        best_idx = None
        best_new_types = -1
        best_length = float("inf")

        for idx, etypes in enumerate(example_entity_types):
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
            covered_types.update(example_entity_types[best_idx])

    examples = []
    for idx in selected_indices:
        example = train_data[idx]
        tokens = example["tokens"]
        tag_ids = example[tag_column]
        tags = [label_schema.id2label[t] for t in tag_ids]
        examples.append({"tokens": tokens, "tags": tags})

    print(f"Selected {len(examples)} few-shot examples covering {len(covered_types)}/{len(label_schema.ENTITY_TYPES)} entity types")
    return examples


def print_dataset_stats(dataset, tag_column="ner_tags"):
    """Print entity type distribution across splits."""
    for split_name, split_data in dataset.items():
        counter = Counter()
        total_tokens = 0
        for example in split_data:
            tag_ids = example[tag_column]
            tags = [label_schema.id2label[t] for t in tag_ids]
            total_tokens += len(tags)
            for tag in tags:
                if tag.startswith("B-"):
                    counter[tag[2:]] += 1

        total_entities = sum(counter.values())
        print(f"\n{split_name} split — {len(split_data)} examples, {total_tokens} tokens, {total_entities} entities")
        print(f"  Entity counts:")
        for etype, count in counter.most_common():
            print(f"    {etype}: {count}")

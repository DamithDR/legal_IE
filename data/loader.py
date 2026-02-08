"""Dataset loading and preprocessing for InLegalNER (span-annotation format)."""

import random
import re
from collections import Counter

from datasets import load_dataset

import config
from data.label_schema import ENTITY_TYPES, LABEL_LIST, id2label, label2id


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

        if ent_type not in ENTITY_TYPES:
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
    tag_ids = [label2id.get(tag, 0) for tag in bio_tags]

    return {"tokens": tokens, "ner_tags": tag_ids}


def load_ner_dataset():
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

        from datasets import Dataset
        converted[split_name] = Dataset.from_dict(records)
        print(f"  {split_name}: {len(converted[split_name])} examples")

    from datasets import DatasetDict
    dataset = DatasetDict(converted)

    tag_column = "ner_tags"
    print(f"Using {len(LABEL_LIST)} labels: {LABEL_LIST}")

    return dataset, tag_column


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
        tags = [id2label[t] for t in tag_ids]
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
        tags = [id2label[t] for t in tag_ids]
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
        tags = [id2label[t] for t in tag_ids]
        examples.append({"tokens": tokens, "tags": tags})

    print(f"Selected {len(examples)} few-shot examples covering {len(covered_types)}/{len(ENTITY_TYPES)} entity types")
    return examples


def print_dataset_stats(dataset, tag_column="ner_tags"):
    """Print entity type distribution across splits."""
    for split_name, split_data in dataset.items():
        counter = Counter()
        total_tokens = 0
        for example in split_data:
            tag_ids = example[tag_column]
            tags = [id2label[t] for t in tag_ids]
            total_tokens += len(tags)
            for tag in tags:
                if tag.startswith("B-"):
                    counter[tag[2:]] += 1

        total_entities = sum(counter.values())
        print(f"\n{split_name} split — {len(split_data)} examples, {total_tokens} tokens, {total_entities} entities")
        print(f"  Entity counts:")
        for etype, count in counter.most_common():
            print(f"    {etype}: {count}")

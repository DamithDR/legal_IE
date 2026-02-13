"""Dataset loading and preprocessing for Relationship Extraction (IE4Wills)."""

import json
import random
from collections import Counter
from dataclasses import dataclass, asdict

import config
from data.relation_schema import (
    CORE_RELATION_TYPES,
    RELEVANT_ENTITY_TYPES,
    RELATION_SIGNATURES,
    ENTITY_PAIR_TO_RELATIONS,
    re_label2id,
)


@dataclass
class Entity:
    id: str
    text: str
    label: str
    start: int
    end: int


@dataclass
class RelationInstance:
    doc_id: int
    text: str
    entity1: Entity
    entity2: Entity
    relation_type: str


def _extract_entities_from_doc(example: dict) -> dict[str, Entity]:
    """Build id -> Entity mapping from a Label Studio JSON document."""
    entity_map = {}
    for ann_set in example["annotations"]:
        for result in ann_set["result"]:
            if result["type"] == "labels":
                value = result["value"]
                labels = value.get("labels", [])
                if not labels:
                    continue
                entity_map[result["id"]] = Entity(
                    id=result["id"],
                    text=value["text"],
                    label=labels[0],
                    start=value["start"],
                    end=value["end"],
                )
    return entity_map


def _extract_relations_from_doc(
    example: dict, entity_map: dict[str, Entity],
) -> list[tuple[Entity, Entity, str]]:
    """Extract (entity1, entity2, relation_type) triples for core relation types."""
    relations = []
    for ann_set in example["annotations"]:
        for result in ann_set["result"]:
            if result["type"] != "relation":
                continue
            labels = result.get("labels", [])
            if not labels:
                continue
            rel_type = labels[0]
            if rel_type not in CORE_RELATION_TYPES:
                continue

            from_id = result["from_id"]
            to_id = result["to_id"]
            if from_id not in entity_map or to_id not in entity_map:
                continue

            e1 = entity_map[from_id]
            e2 = entity_map[to_id]

            # Validate entity types match expected signature
            expected = RELATION_SIGNATURES.get(rel_type)
            if expected and (e1.label, e2.label) != expected:
                # Try swapped order
                if (e2.label, e1.label) == expected:
                    e1, e2 = e2, e1
                else:
                    continue

            relations.append((e1, e2, rel_type))
    return relations


def _generate_candidate_pairs(
    doc_id: int,
    text: str,
    entities: list[Entity],
    gold_relations: set[tuple[str, str, str]],
    negative_ratio: float,
    rng: random.Random,
) -> list[RelationInstance]:
    """
    Generate negative (NO_RELATION) instances from entity pairs that have
    valid type signatures but no gold relation.
    """
    negatives = []
    relevant = [e for e in entities if e.label in RELEVANT_ENTITY_TYPES]

    for i, e1 in enumerate(relevant):
        for j, e2 in enumerate(relevant):
            if i == j:
                continue
            # Check if this entity type pair could form a valid relation
            if (e1.label, e2.label) not in ENTITY_PAIR_TO_RELATIONS:
                continue
            # Check if there's already a gold relation for this pair
            key = (e1.id, e2.id)
            if any((e1.id, e2.id, rt) in gold_relations for rt in CORE_RELATION_TYPES):
                continue

            negatives.append(RelationInstance(
                doc_id=doc_id,
                text=text,
                entity1=e1,
                entity2=e2,
                relation_type="NO_RELATION",
            ))

    # Sample negatives according to ratio
    if negative_ratio > 0 and negatives:
        # Count positives for this doc from gold_relations
        n_positives = len(gold_relations)
        max_negatives = max(1, int(n_positives * negative_ratio))
        if len(negatives) > max_negatives:
            negatives = rng.sample(negatives, max_negatives)

    return negatives


def load_re_dataset(split: str = "train", negative_ratio: float = 1.0, seed: int = 42) -> list[RelationInstance]:
    """
    Load RE instances for a given split.

    Args:
        split: 'train', 'validation', or 'test'.
        negative_ratio: Ratio of negative to positive examples (0 = positives only).
        seed: Random seed for negative sampling.

    Returns:
        List of RelationInstance objects.
    """
    split_files = {
        "train": config.IE4WILLS_DATA_DIR / "train.json",
        "validation": config.IE4WILLS_DATA_DIR / "dev.json",
        "test": config.IE4WILLS_DATA_DIR / "test.json",
    }
    filepath = split_files[split]
    print(f"Loading RE instances from {filepath} (split={split})...")

    with open(filepath, encoding="utf-8") as f:
        raw_examples = json.load(f)

    rng = random.Random(seed)
    instances = []

    for example in raw_examples:
        doc_id = example["id"]
        text = example["data"]["text"]
        entity_map = _extract_entities_from_doc(example)
        relations = _extract_relations_from_doc(example, entity_map)

        # Build gold relation set for negative sampling
        gold_set = {(e1.id, e2.id, rt) for e1, e2, rt in relations}

        # Positive instances
        for e1, e2, rel_type in relations:
            instances.append(RelationInstance(
                doc_id=doc_id,
                text=text,
                entity1=e1,
                entity2=e2,
                relation_type=rel_type,
            ))

        # Negative instances
        if negative_ratio > 0:
            all_entities = list(entity_map.values())
            negatives = _generate_candidate_pairs(
                doc_id, text, all_entities, gold_set, negative_ratio, rng,
            )
            instances.extend(negatives)

    pos_count = sum(1 for inst in instances if inst.relation_type != "NO_RELATION")
    neg_count = len(instances) - pos_count
    print(f"  {split}: {pos_count} positive, {neg_count} negative, {len(instances)} total instances")
    return instances


def load_re_documents(split: str = "test") -> list[dict]:
    """
    Load documents with gold entities and gold relations for LLM evaluation.

    Returns:
        List of dicts with doc_id, text, entities, gold_relations.
    """
    split_files = {
        "train": config.IE4WILLS_DATA_DIR / "train.json",
        "validation": config.IE4WILLS_DATA_DIR / "dev.json",
        "test": config.IE4WILLS_DATA_DIR / "test.json",
    }
    filepath = split_files[split]
    print(f"Loading RE documents from {filepath} (split={split})...")

    with open(filepath, encoding="utf-8") as f:
        raw_examples = json.load(f)

    documents = []
    for example in raw_examples:
        doc_id = example["id"]
        text = example["data"]["text"]
        entity_map = _extract_entities_from_doc(example)
        relations = _extract_relations_from_doc(example, entity_map)

        # Filter entities to relevant types only
        relevant_entities = [
            e for e in entity_map.values() if e.label in RELEVANT_ENTITY_TYPES
        ]

        # Skip documents with no relevant entities or no core relations
        if not relevant_entities or not relations:
            continue

        gold_relations = [
            {
                "entity1_text": e1.text,
                "entity1_type": e1.label,
                "entity2_text": e2.text,
                "entity2_type": e2.label,
                "relation_type": rel_type,
            }
            for e1, e2, rel_type in relations
        ]

        documents.append({
            "doc_id": doc_id,
            "text": text,
            "entities": [asdict(e) for e in relevant_entities],
            "gold_relations": gold_relations,
        })

    total_rels = sum(len(d["gold_relations"]) for d in documents)
    print(f"  {split}: {len(documents)} documents, {total_rels} gold relations")
    return documents


def get_re_few_shot_examples(n: int = 3, seed: int = 42) -> list[dict]:
    """
    Select diverse few-shot examples from training set using greedy set-cover
    to maximize relation type coverage. Prefers shorter documents.
    """
    documents = load_re_documents(split="train")

    # Pre-compute relation types per document
    doc_rel_types = []
    for doc in documents:
        rel_types = {r["relation_type"] for r in doc["gold_relations"]}
        doc_rel_types.append(rel_types)

    # Greedy set-cover
    selected = []
    covered = set()

    for _ in range(n):
        best_idx = None
        best_new = -1
        best_len = float("inf")

        for idx, rtypes in enumerate(doc_rel_types):
            if idx in selected:
                continue
            new_types = len(rtypes - covered)
            doc_len = len(documents[idx]["text"])

            if new_types > best_new or (new_types == best_new and doc_len < best_len):
                best_idx = idx
                best_new = new_types
                best_len = doc_len

        if best_idx is not None:
            selected.append(best_idx)
            covered.update(doc_rel_types[best_idx])

    examples = [documents[i] for i in selected]
    print(f"Selected {len(examples)} few-shot examples covering {len(covered)}/{len(CORE_RELATION_TYPES)} relation types")
    return examples


def print_re_dataset_stats():
    """Print relation type distribution across splits."""
    for split in ["train", "validation", "test"]:
        split_files = {
            "train": config.IE4WILLS_DATA_DIR / "train.json",
            "validation": config.IE4WILLS_DATA_DIR / "dev.json",
            "test": config.IE4WILLS_DATA_DIR / "test.json",
        }
        with open(split_files[split], encoding="utf-8") as f:
            raw_examples = json.load(f)

        counter = Counter()
        n_docs_with_rels = 0
        for example in raw_examples:
            entity_map = _extract_entities_from_doc(example)
            relations = _extract_relations_from_doc(example, entity_map)
            if relations:
                n_docs_with_rels += 1
            for _, _, rel_type in relations:
                counter[rel_type] += 1

        total = sum(counter.values())
        print(f"\n{split} split — {len(raw_examples)} documents, "
              f"{n_docs_with_rels} with core relations, {total} total relations")
        print(f"  Relation counts:")
        for rel_type in CORE_RELATION_TYPES:
            print(f"    {rel_type}: {counter.get(rel_type, 0)}")

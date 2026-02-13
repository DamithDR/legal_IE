"""Prompt templates for LLM-based Relationship Extraction."""

import json

from data.relation_schema import CORE_RELATION_TYPES


# Semantic descriptions for each relation type
_RELATION_DESCRIPTIONS = {
    "TESTATOR-BENEFICIARY": "The testator designates or references a beneficiary in the will",
    "TESTATOR-ASSET": "The testator owns, possesses, or references an asset",
    "BENEFICIARY-ASSET": "A beneficiary receives or is associated with a specific asset",
    "TESTATOR-WILL": "The testator creates, signs, or references the will document",
}


def build_re_system_prompt() -> str:
    """Build the system prompt for the RE task."""
    rel_lines = []
    for i, rel_type in enumerate(CORE_RELATION_TYPES, 1):
        desc = _RELATION_DESCRIPTIONS[rel_type]
        rel_lines.append(f"{i}. {rel_type}: {desc}")
    rel_section = "\n".join(rel_lines)

    return f"""You are an expert at Relationship Extraction for US legal wills and testamentary documents.
Given a text and a list of annotated entities, identify all relationships between entity pairs.

The relationship types are:
{rel_section}

Return your answer as a JSON array of objects, each with:
- "entity1": the exact text of the first entity from the provided entity list
- "entity1_type": the type of the first entity (TESTATOR, BENEFICIARY, ASSET, or WILL)
- "entity2": the exact text of the second entity from the provided entity list
- "entity2_type": the type of the second entity
- "relation": one of the relationship types listed above

Only include pairs that have one of the above relationships.
If no relationships are found, return an empty array [].
Return ONLY the JSON array, no other text."""


def format_re_entities(entities: list[dict]) -> str:
    """Format entity list for the prompt."""
    # Deduplicate by (text, label) to avoid repeating the same entity mention
    seen = set()
    lines = []
    for e in entities:
        key = (e["text"], e["label"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'- "{e["text"]}" ({e["label"]})')
    return "\n".join(lines)


def format_re_example(doc: dict) -> str:
    """Format a single document as an RE few-shot example."""
    text = doc["text"]
    entities_str = format_re_entities(doc["entities"])
    relations_json = json.dumps(
        [
            {
                "entity1": r["entity1_text"],
                "entity1_type": r["entity1_type"],
                "entity2": r["entity2_text"],
                "entity2_type": r["entity2_type"],
                "relation": r["relation_type"],
            }
            for r in doc["gold_relations"]
        ],
        indent=2,
    )

    return f'Text: "{text}"\nEntities:\n{entities_str}\nRelations:\n{relations_json}'


def build_re_user_prompt(doc: dict, few_shot_examples: list[dict]) -> str:
    """
    Build the full user prompt with few-shot examples + target document.

    Args:
        doc: Target document dict with 'text' and 'entities'.
        few_shot_examples: List of document dicts with 'text', 'entities', 'gold_relations'.

    Returns:
        Formatted user prompt string.
    """
    # Few-shot section
    example_sections = []
    for example in few_shot_examples:
        example_sections.append(format_re_example(example))
    few_shot_section = "\n---\n".join(example_sections)

    # Target document
    target_text = doc["text"]
    target_entities = format_re_entities(doc["entities"])

    return f"""Here are some examples:
---
{few_shot_section}
---

Now extract relationships from this text:
Text: "{target_text}"
Entities:
{target_entities}
Relations:"""

"""Prompt templates and few-shot example formatting for LLM-based NER."""

from data import label_schema


def build_system_prompt():
    """Build the system prompt dynamically from the active label schema."""
    return f"""You are an expert at Named Entity Recognition for Indian legal court judgments.
Given a text, identify all named entities and classify each into exactly one of these {len(label_schema.ENTITY_TYPES)} types:
{', '.join(label_schema.ENTITY_TYPES)}

Return your answer as a JSON array of objects, each with "entity" (the exact text span from the input) and "type" (one of the entity types listed above).
If no entities are found, return an empty array [].
Return ONLY the JSON array, no other text."""


def format_example_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    """Extract entity spans from BIO-tagged tokens."""
    entities = []
    current_entity = None
    current_type = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # Save previous entity
            if current_entity is not None:
                entities.append({"entity": current_entity, "type": current_type})
            current_entity = token
            current_type = tag[2:]
        elif tag.startswith("I-") and current_entity is not None:
            current_entity += " " + token
        else:
            # O tag — close any open entity
            if current_entity is not None:
                entities.append({"entity": current_entity, "type": current_type})
                current_entity = None
                current_type = None

    # Close last entity if still open
    if current_entity is not None:
        entities.append({"entity": current_entity, "type": current_type})

    return entities


def build_few_shot_section(few_shot_examples: list[dict]) -> str:
    """Build the few-shot examples section of the prompt."""
    sections = []
    for i, example in enumerate(few_shot_examples, 1):
        text = " ".join(example["tokens"])
        entities = format_example_entities(example["tokens"], example["tags"])
        import json
        entities_json = json.dumps(entities, indent=2)
        sections.append(f"Text: \"{text}\"\nEntities:\n{entities_json}")

    return "\n---\n".join(sections)


def build_user_prompt(tokens: list[str], few_shot_examples: list[dict]) -> str:
    """
    Build the full user prompt for a single NER sample.

    Args:
        tokens: List of tokens in the sentence to evaluate.
        few_shot_examples: List of dicts with 'tokens' and 'tags'.

    Returns:
        Formatted user prompt string.
    """
    few_shot_section = build_few_shot_section(few_shot_examples)
    input_text = " ".join(tokens)

    prompt = f"""Here are some examples:
---
{few_shot_section}
---

Now extract entities from this text:
Text: \"{input_text}\"
Entities:"""

    return prompt

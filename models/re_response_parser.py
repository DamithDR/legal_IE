"""Parse LLM responses for Relationship Extraction."""

from models.response_parser import extract_json_from_response
from data.relation_schema import CORE_RELATION_TYPES, RELATION_SIGNATURES


def _match_entity_text(predicted_text: str, entities: list[dict]) -> dict | None:
    """
    Match predicted entity text to a gold entity.

    Tries: exact match, case-insensitive, substring containment.
    Returns the matched entity dict or None.
    """
    if not predicted_text:
        return None

    # Exact match
    for e in entities:
        if e["text"] == predicted_text:
            return e

    # Case-insensitive match
    pred_lower = predicted_text.lower()
    for e in entities:
        if e["text"].lower() == pred_lower:
            return e

    # Predicted text contained in entity text
    for e in entities:
        if pred_lower in e["text"].lower():
            return e

    # Entity text contained in predicted text
    for e in entities:
        if e["text"].lower() in pred_lower:
            return e

    return None


def parse_re_response(response: str, doc_entities: list[dict]) -> list[dict]:
    """
    Parse LLM response into validated relation predictions.

    Args:
        response: Raw LLM response text.
        doc_entities: List of gold entity dicts with 'text', 'label', etc.

    Returns:
        List of validated relation dicts with:
        entity1_text, entity1_type, entity2_text, entity2_type, relation_type
    """
    parsed = extract_json_from_response(response)
    if parsed is None:
        return []

    validated = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        # Extract fields (handle various key naming)
        e1_text = item.get("entity1") or item.get("entity1_text", "")
        e2_text = item.get("entity2") or item.get("entity2_text", "")
        rel_type = item.get("relation") or item.get("relation_type", "")

        # Validate relation type
        if rel_type not in CORE_RELATION_TYPES:
            continue

        # Match entity texts to gold entities
        e1_match = _match_entity_text(e1_text, doc_entities)
        e2_match = _match_entity_text(e2_text, doc_entities)

        if e1_match is None or e2_match is None:
            continue

        # Validate entity types match relation signature
        expected = RELATION_SIGNATURES.get(rel_type)
        if expected:
            if (e1_match["label"], e2_match["label"]) == expected:
                pass  # correct order
            elif (e2_match["label"], e1_match["label"]) == expected:
                # Swap to correct order
                e1_match, e2_match = e2_match, e1_match
            else:
                continue

        validated.append({
            "entity1_text": e1_match["text"],
            "entity1_type": e1_match["label"],
            "entity2_text": e2_match["text"],
            "entity2_type": e2_match["label"],
            "relation_type": rel_type,
        })

    return validated

"""Parse LLM responses (event trigger spans) back to BIO-tagged token sequences for contract events."""

from data import contract_event_schema
from models.response_parser import (
    extract_json_from_response,
    _find_entity_tokens,
)


def _contract_event_spans_to_bio_tags(tokens: list[str], entities: list[dict]) -> list[str]:
    """
    Convert extracted contract event trigger spans to BIO tags aligned with the token list.

    Validates against contract_event_schema.EVENT_TYPES.
    """
    tags = ["O"] * len(tokens)

    if not entities:
        return tags

    token_offsets = []
    current_pos = 0
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        token_offsets.append((start, end))
        current_pos = end + 1  # +1 for space

    reconstructed_text = " ".join(tokens)
    assigned = [False] * len(tokens)

    for ent in entities:
        entity_text = ent.get("entity", "")
        entity_type = ent.get("type", "")

        if not entity_text or not entity_type:
            continue

        # Validate event type
        if entity_type not in contract_event_schema.EVENT_TYPES:
            matched = False
            for et in contract_event_schema.EVENT_TYPES:
                if et.lower() == entity_type.lower():
                    entity_type = et
                    matched = True
                    break
            if not matched:
                continue

        matched_token_indices = _find_entity_tokens(
            entity_text, tokens, token_offsets, reconstructed_text, assigned
        )

        if matched_token_indices:
            for j, token_idx in enumerate(matched_token_indices):
                if not assigned[token_idx]:
                    if j == 0:
                        tags[token_idx] = f"B-{entity_type}"
                    else:
                        tags[token_idx] = f"I-{entity_type}"
                    assigned[token_idx] = True

    return tags


def parse_contract_event_response(response: str, tokens: list[str]) -> list[str]:
    """
    Parse an LLM response into BIO tags for contract event triggers.

    The response should contain JSON with "trigger" and "type" fields.

    Args:
        response: Raw LLM response string.
        tokens: Original token list for alignment.

    Returns:
        List of BIO tag strings. Falls back to all 'O' tags on parse failure.
    """
    entities = extract_json_from_response(response)

    if entities is None:
        return ["O"] * len(tokens)

    valid_entities = []
    for ent in entities:
        if isinstance(ent, dict):
            trigger = ent.get("trigger") or ent.get("entity", "")
            etype = ent.get("type", "")

            if not trigger or not etype:
                continue

            # Normalize event type (case-insensitive matching)
            if etype not in contract_event_schema.EVENT_TYPES:
                matched = False
                for et in contract_event_schema.EVENT_TYPES:
                    if et.lower() == etype.lower():
                        etype = et
                        matched = True
                        break
                if not matched:
                    continue

            valid_entities.append({"entity": trigger, "type": etype})

    return _contract_event_spans_to_bio_tags(tokens, valid_entities)

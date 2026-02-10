"""Parse LLM responses (entity spans) back to BIO-tagged token sequences."""

import json
import re

from data import label_schema


def extract_json_from_response(response: str) -> list | None:
    """
    Extract a JSON array from an LLM response, handling markdown code blocks
    and surrounding text.

    Returns:
        Parsed list of dicts, or None if parsing fails.
    """
    # Try direct parse first
    response = response.strip()
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    if code_block:
        try:
            result = json.loads(code_block.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding a JSON array anywhere in the response
    array_match = re.search(r"\[.*\]", response, re.DOTALL)
    if array_match:
        try:
            result = json.loads(array_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try fixing single quotes -> double quotes
    if array_match:
        fixed = array_match.group(0).replace("'", '"')
        try:
            result = json.loads(fixed)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def spans_to_bio_tags(tokens: list[str], entities: list[dict]) -> list[str]:
    """
    Convert extracted entity spans to BIO tags aligned with the token list.

    Uses character-offset-based matching: reconstruct text from tokens,
    find entity spans in the text, then map back to token indices.

    Args:
        tokens: Original token list.
        entities: List of dicts with 'entity' and 'type' keys.

    Returns:
        List of BIO tag strings, one per token.
    """
    tags = ["O"] * len(tokens)

    if not entities:
        return tags

    # Build character offset map: for each token, its (start, end) in reconstructed text
    text_parts = []
    token_offsets = []
    current_pos = 0
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        token_offsets.append((start, end))
        text_parts.append(token)
        current_pos = end + 1  # +1 for the space

    reconstructed_text = " ".join(tokens)

    # Track which tokens are already assigned
    assigned = [False] * len(tokens)

    for ent in entities:
        entity_text = ent.get("entity", "")
        entity_type = ent.get("type", "")

        if not entity_text or not entity_type:
            continue

        # Validate entity type
        if entity_type not in label_schema.ENTITY_TYPES:
            # Try case-insensitive match
            matched = False
            for et in label_schema.ENTITY_TYPES:
                if et.lower() == entity_type.lower():
                    entity_type = et
                    matched = True
                    break
            if not matched:
                continue

        # Find entity span in reconstructed text
        matched_token_indices = _find_entity_tokens(
            entity_text, tokens, token_offsets, reconstructed_text, assigned
        )

        if matched_token_indices:
            # Assign B/I tags
            for j, token_idx in enumerate(matched_token_indices):
                if not assigned[token_idx]:
                    if j == 0:
                        tags[token_idx] = f"B-{entity_type}"
                    else:
                        tags[token_idx] = f"I-{entity_type}"
                    assigned[token_idx] = True

    return tags


def _find_entity_tokens(
    entity_text: str,
    tokens: list[str],
    token_offsets: list[tuple[int, int]],
    reconstructed_text: str,
    assigned: list[bool],
) -> list[int] | None:
    """
    Find token indices that match an entity span.

    Tries multiple strategies:
    1. Exact substring match in reconstructed text
    2. Case-insensitive match
    3. Token-level sequential matching
    """
    # Strategy 1: Exact substring match
    result = _match_by_substring(entity_text, token_offsets, reconstructed_text, assigned)
    if result:
        return result

    # Strategy 2: Case-insensitive substring match
    result = _match_by_substring(
        entity_text, token_offsets, reconstructed_text, assigned, case_insensitive=True
    )
    if result:
        return result

    # Strategy 3: Token-level sequential match
    result = _match_by_tokens(entity_text, tokens, assigned)
    if result:
        return result

    return None


def _match_by_substring(
    entity_text: str,
    token_offsets: list[tuple[int, int]],
    reconstructed_text: str,
    assigned: list[bool],
    case_insensitive: bool = False,
) -> list[int] | None:
    """Find entity by substring matching in reconstructed text."""
    search_text = reconstructed_text
    search_entity = entity_text

    if case_insensitive:
        search_text = search_text.lower()
        search_entity = search_entity.lower()

    # Find all occurrences and pick the first unassigned one
    start = 0
    while True:
        pos = search_text.find(search_entity, start)
        if pos == -1:
            break

        end_pos = pos + len(entity_text)

        # Map character range to token indices
        matched_indices = []
        for idx, (tok_start, tok_end) in enumerate(token_offsets):
            # Token overlaps with the entity span
            if tok_start < end_pos and tok_end > pos:
                matched_indices.append(idx)

        # Check if any of these tokens are already assigned
        if matched_indices and not any(assigned[i] for i in matched_indices):
            return matched_indices

        start = pos + 1

    return None


def _match_by_tokens(
    entity_text: str,
    tokens: list[str],
    assigned: list[bool],
) -> list[int] | None:
    """Find entity by matching its tokens sequentially against the token list."""
    entity_tokens = entity_text.split()
    if not entity_tokens:
        return None

    for i in range(len(tokens) - len(entity_tokens) + 1):
        if assigned[i]:
            continue

        match = True
        for j, et in enumerate(entity_tokens):
            if i + j >= len(tokens):
                match = False
                break
            # Exact or case-insensitive token match
            if tokens[i + j] != et and tokens[i + j].lower() != et.lower():
                match = False
                break

        if match:
            indices = list(range(i, i + len(entity_tokens)))
            if not any(assigned[idx] for idx in indices):
                return indices

    return None


def parse_llm_response(response: str, tokens: list[str]) -> list[str]:
    """
    Parse an LLM response into BIO tags aligned with the given tokens.

    This is the main entry point for response parsing.

    Args:
        response: Raw LLM response string.
        tokens: Original token list for alignment.

    Returns:
        List of BIO tag strings. Falls back to all 'O' tags on parse failure.
    """
    entities = extract_json_from_response(response)

    if entities is None:
        # Complete parse failure — conservative fallback
        return ["O"] * len(tokens)

    # Validate entity format
    valid_entities = []
    for ent in entities:
        if isinstance(ent, dict) and "entity" in ent and "type" in ent:
            valid_entities.append(ent)

    return spans_to_bio_tags(tokens, valid_entities)

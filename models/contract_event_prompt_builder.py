"""Prompt templates and few-shot example formatting for LLM-based Contract Event Detection."""

import json

from data import contract_event_schema


def build_contract_event_system_prompt(dataset_name="contractual_events"):
    """Build the system prompt for contract event trigger detection."""
    return f"""You are an expert at Event Detection in US court decisions involving contractual disputes.

Given a text, identify all event triggers (the word or phrase that most directly indicates an event occurred) and classify each into exactly one of these {len(contract_event_schema.EVENT_TYPES)} event types:
{', '.join(contract_event_schema.EVENT_TYPES)}

- Purchase-contract: Events indicating a purchase or sale contract (e.g., sold, bought, purchased).
- Other-contract: Events indicating other types of contracts (e.g., leased, rented, licensed).
- Execution: Events where a contract is executed or performed (e.g., signed, executed, delivered).
- Agreement: Events where parties reach an agreement (e.g., agreed, consented, accepted).
- Rescission: Events where a contract is rescinded or cancelled (e.g., rescinded, cancelled, revoked).
- Negotiation: Events related to contract negotiations (e.g., negotiated, proposed, offered).
- Expiration: Events where a contract expires or terminates (e.g., expired, terminated, ended).

Return your answer as a JSON array of objects, each with "trigger" (the exact text span from the input) and "type" (one of the event types listed above).
If no event triggers are found, return an empty array [].
Return ONLY the JSON array, no other text."""


def format_example_events(tokens: list[str], tags: list[str]) -> list[dict]:
    """Extract event trigger spans from BIO-tagged tokens."""
    events = []
    current_trigger = None
    current_type = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_trigger is not None:
                events.append({"trigger": current_trigger, "type": current_type})
            current_trigger = token
            current_type = tag[2:]
        elif tag.startswith("I-") and current_trigger is not None:
            current_trigger += " " + token
        else:
            if current_trigger is not None:
                events.append({"trigger": current_trigger, "type": current_type})
                current_trigger = None
                current_type = None

    if current_trigger is not None:
        events.append({"trigger": current_trigger, "type": current_type})

    return events


def build_few_shot_section(few_shot_examples: list[dict]) -> str:
    """Build the few-shot examples section of the prompt."""
    sections = []
    for i, example in enumerate(few_shot_examples, 1):
        text = " ".join(example["tokens"])
        events = format_example_events(example["tokens"], example["tags"])
        events_json = json.dumps(events, indent=2)
        sections.append(f"Text: \"{text}\"\nEvent triggers:\n{events_json}")

    return "\n---\n".join(sections)


def build_contract_event_user_prompt(tokens: list[str], few_shot_examples: list[dict]) -> str:
    """
    Build the full user prompt for a single contract event detection sample.

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

Now extract event triggers from this text:
Text: \"{input_text}\"
Event triggers:"""

    return prompt

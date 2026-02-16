"""Prompt templates and few-shot example formatting for LLM-based Event Detection."""

import json

from data import event_schema


def build_event_system_prompt(dataset_name="events_matter"):
    """Build the system prompt for event trigger detection."""
    return f"""You are an expert at Event Detection in legal court decisions from the European Court of Human Rights (ECHR).

Given a text, identify all event triggers (the word or phrase that most directly indicates an event occurred) and classify each into exactly one of these {len(event_schema.EVENT_TYPES)} event types:
{', '.join(event_schema.EVENT_TYPES)}

- PROCEDURE: Events related to legal/judicial procedures (e.g., filing applications, lodging appeals, delivering judgments, issuing decisions).
- CIRCUMSTANCE: Events related to factual circumstances of the case (e.g., being born, being arrested, being convicted, sending packages).

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


def build_event_user_prompt(tokens: list[str], few_shot_examples: list[dict]) -> str:
    """
    Build the full user prompt for a single event detection sample.

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

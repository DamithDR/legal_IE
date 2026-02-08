"""Anthropic Claude NER evaluator."""

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from models.llm_base import LLMEvaluator


class ClaudeEvaluator(LLMEvaluator):
    """Evaluator using Anthropic's Claude API."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.content[0].text

"""LLM RE evaluator providers — reuses API call logic from NER evaluators."""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import anthropic

from models.llm_re_base import LLMREEvaluator


class OpenAIREEvaluator(LLMREEvaluator):
    """RE evaluator using OpenAI's ChatGPT API."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return response.choices[0].message.content


class DeepSeekREEvaluator(LLMREEvaluator):
    """RE evaluator using DeepSeek's OpenAI-compatible API."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def call_api(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return response.choices[0].message.content


class ClaudeREEvaluator(LLMREEvaluator):
    """RE evaluator using Anthropic's Claude API."""

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

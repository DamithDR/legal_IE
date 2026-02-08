"""OpenAI (ChatGPT) NER evaluator."""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from models.llm_base import LLMEvaluator


class OpenAIEvaluator(LLMEvaluator):
    """Evaluator using OpenAI's ChatGPT API."""

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

"""Abstract base class for LLM-based NER evaluators."""

import time
from abc import ABC, abstractmethod

from tqdm import tqdm

import config
from models.prompt_builder import build_system_prompt, build_user_prompt
from models.response_parser import parse_llm_response
from evaluation.metrics import compute_ner_metrics


class LLMEvaluator(ABC):
    """Base class for LLM NER evaluators."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single API call. Returns raw response text."""
        pass

    def evaluate_sample(self, tokens: list[str], few_shot_examples: list[dict], dataset_name: str = "inlegalner") -> list[str]:
        """
        Evaluate a single sample.

        Args:
            tokens: Input tokens.
            few_shot_examples: Few-shot examples for the prompt.
            dataset_name: Dataset identifier for prompt context.

        Returns:
            Predicted BIO tags.
        """
        user_prompt = build_user_prompt(tokens, few_shot_examples)
        system_prompt = build_system_prompt(dataset_name)
        response = self.call_api(system_prompt, user_prompt)
        pred_tags = parse_llm_response(response, tokens)
        return pred_tags

    def evaluate_dataset(
        self,
        samples: list[dict],
        few_shot_examples: list[dict],
        provider_name: str,
        dataset_name: str = "inlegalner",
    ) -> dict:
        """
        Evaluate multiple samples with rate limiting and error handling.

        Args:
            samples: List of dicts with 'tokens' and 'tags'.
            few_shot_examples: Few-shot examples for the prompt.
            provider_name: Name for results file (e.g. 'openai').
            dataset_name: Dataset identifier for namespacing results.

        Returns:
            Results dict with metrics.
        """
        true_labels = []
        pred_labels = []
        failures = 0

        print(f"\nEvaluating {provider_name} ({self.model_name}) on {len(samples)} samples...")

        for i, sample in enumerate(tqdm(samples, desc=provider_name)):
            tokens = sample["tokens"]
            true_tags = sample["tags"]

            try:
                pred_tags = self.evaluate_sample(tokens, few_shot_examples, dataset_name)
                true_labels.append(true_tags)
                pred_labels.append(pred_tags)
            except Exception as e:
                print(f"\n  Sample {i} failed: {e}")
                failures += 1
                # Use all-O as fallback for failed samples
                true_labels.append(true_tags)
                pred_labels.append(["O"] * len(tokens))

            # Rate limiting delay
            time.sleep(config.LLM_REQUEST_DELAY)

        # Compute metrics
        results = compute_ner_metrics(true_labels, pred_labels)

        print(f"\n{provider_name} Results ({failures} failures out of {len(samples)}):")
        print(results["classification_report_str"])

        # Save results
        save_data = {
            "model_name": provider_name,
            "model_type": "llm",
            "llm_model_id": self.model_name,
            "num_samples_evaluated": len(samples),
            "num_failures": failures,
            "overall": results["overall"],
            "macro_avg": results["macro_avg"],
            "per_entity": results["per_entity"],
        }
        config.save_results(f"{dataset_name}_{provider_name}", save_data)

        return save_data

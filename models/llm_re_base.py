"""Abstract base class for LLM-based RE evaluators."""

import time
from abc import ABC, abstractmethod

from tqdm import tqdm

import config
from models.re_prompt_builder import build_re_system_prompt, build_re_user_prompt
from models.re_response_parser import parse_re_response
from evaluation.re_metrics import compute_re_metrics


class LLMREEvaluator(ABC):
    """Base class for LLM RE evaluators."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single API call. Returns raw response text."""
        pass

    def evaluate_document(self, doc: dict, few_shot_examples: list[dict]) -> list[dict]:
        """
        Evaluate a single document for relationship extraction.

        Args:
            doc: Document dict with 'text', 'entities'.
            few_shot_examples: List of document dicts with gold relations.

        Returns:
            List of predicted relation dicts.
        """
        system_prompt = build_re_system_prompt()
        user_prompt = build_re_user_prompt(doc, few_shot_examples)
        response = self.call_api(system_prompt, user_prompt)
        return parse_re_response(response, doc["entities"])

    def evaluate_dataset(
        self,
        documents: list[dict],
        few_shot_examples: list[dict],
        provider_name: str,
    ) -> dict:
        """
        Evaluate all documents with rate limiting and error handling.

        Args:
            documents: List of document dicts with 'text', 'entities', 'gold_relations'.
            few_shot_examples: Few-shot example documents.
            provider_name: Name for results file.

        Returns:
            Results dict with metrics.
        """
        gold_relations = []
        pred_relations = []
        failures = 0

        print(f"\nEvaluating RE {provider_name} ({self.model_name}) on {len(documents)} documents...")

        for i, doc in enumerate(tqdm(documents, desc=f"RE {provider_name}")):
            try:
                preds = self.evaluate_document(doc, few_shot_examples)
                gold_relations.append(doc["gold_relations"])
                pred_relations.append(preds)
            except Exception as e:
                print(f"\n  Document {i} failed: {e}")
                failures += 1
                gold_relations.append(doc["gold_relations"])
                pred_relations.append([])

            time.sleep(config.LLM_REQUEST_DELAY)

        # Compute metrics
        results = compute_re_metrics(gold_relations, pred_relations)

        print(f"\nRE {provider_name} Results ({failures} failures out of {len(documents)}):")
        print(results["classification_report_str"])

        # Save results
        save_data = {
            "model_name": provider_name,
            "model_type": "llm",
            "task": "re",
            "dataset": "ie4wills",
            "llm_model_id": self.model_name,
            "num_documents_evaluated": len(documents),
            "num_failures": failures,
            "overall": results["overall"],
            "macro_avg": results["macro_avg"],
            "per_relation": results["per_relation"],
        }
        config.save_results(f"ie4wills_re_{provider_name}", save_data)

        return save_data

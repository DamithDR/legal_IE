"""Legal IE Evaluation Pipeline — CLI entry point (NER + RE)."""

import argparse
import json
import sys

import config
from data.loader import load_ner_dataset, get_few_shot_examples, get_llm_samples, print_dataset_stats
from data.label_schema import set_active_schema

_ALL_DATASETS = ["inlegalner", "icdac", "ie4wills"]


def _sample_cache_path(dataset_name: str):
    """Return the path for the cached LLM sample file."""
    return config.RESULTS_DIR / f"{dataset_name}_llm_samples.json"


def _save_sample_cache(dataset_name, samples, few_shot_examples, n_samples, n_few_shot):
    """Write samples and few-shot examples to a JSON cache file."""
    cache = {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "n_few_shot": n_few_shot,
        "samples": [
            {
                "sentence": " ".join(s["tokens"]),
                "tokens": s["tokens"],
                "tags": s["tags"],
            }
            for s in samples
        ],
        "few_shot_examples": [
            {
                "sentence": " ".join(e["tokens"]),
                "tokens": e["tokens"],
                "tags": e["tags"],
            }
            for e in few_shot_examples
        ],
    }
    path = _sample_cache_path(dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Sample cache saved to {path} ({n_samples} samples, {n_few_shot} few-shot)")
    return path


def _load_sample_cache(dataset_name):
    """Load cached samples. Returns (samples, few_shot_examples) or raises FileNotFoundError."""
    path = _sample_cache_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No sample cache found at {path}.\n"
            f"Run 'python main.py prepare-samples --dataset {dataset_name}' first."
        )
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    samples = [{"tokens": s["tokens"], "tags": s["tags"]} for s in cache["samples"]]
    few_shot = [{"tokens": e["tokens"], "tags": e["tags"]} for e in cache["few_shot_examples"]]
    print(f"Loaded {len(samples)} cached samples and {len(few_shot)} few-shot examples from {path}")
    return samples, few_shot


def _init_dataset(args):
    """Set active schema and load the dataset selected by --dataset."""
    set_active_schema(args.dataset)
    return load_ner_dataset(args.dataset)


def cmd_train(args):
    """Fine-tune BERT model(s) on the training set."""
    from models.bert_ner import train_bert_model

    dataset, tag_column = _init_dataset(args)

    models_to_train = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_train:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue

        train_split = "train"
        eval_split = "validation" if "validation" in dataset else "dev"

        train_bert_model(
            model_key=model_key,
            train_dataset=dataset[train_split],
            eval_dataset=dataset[eval_split],
            tag_column=tag_column,
            dataset_name=args.dataset,
        )


def cmd_evaluate_bert(args):
    """Evaluate a trained BERT model on the test set."""
    from models.bert_ner import predict_bert

    dataset, tag_column = _init_dataset(args)

    models_to_eval = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_eval:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue

        test_split = "test" if "test" in dataset else "validation"
        predict_bert(
            model_key=model_key,
            test_dataset=dataset[test_split],
            tag_column=tag_column,
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset,
        )


def _prepare_samples_for_dataset(dataset_name, sample_size, full_test, resample):
    """Prepare and cache LLM test samples for a single dataset."""
    cache_path = _sample_cache_path(dataset_name)
    if cache_path.exists() and not resample:
        print(f"Sample cache already exists at {cache_path}")
        print("Use --resample to regenerate.")
        return

    set_active_schema(dataset_name)
    dataset, tag_column = load_ner_dataset(dataset_name)

    n = None if full_test else sample_size
    test_split = "test" if "test" in dataset else "validation"
    samples = get_llm_samples(dataset, split=test_split, n=n, tag_column=tag_column)
    few_shot_examples = get_few_shot_examples(dataset, n=config.LLM_FEW_SHOT_COUNT, tag_column=tag_column)

    _save_sample_cache(dataset_name, samples, few_shot_examples, len(samples), len(few_shot_examples))


def cmd_prepare_samples(args):
    """Prepare and cache LLM test samples for consistent evaluation."""
    datasets = _ALL_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Preparing samples for {ds}")
        print(f"{'='*60}")
        _prepare_samples_for_dataset(ds, args.sample_size, args.full_test, args.resample)


def _evaluate_llm_for_dataset(dataset_name, provider_arg):
    """Run LLM evaluation for a single dataset using cached samples."""
    set_active_schema(dataset_name)

    # Load from cache
    try:
        samples, few_shot_examples = _load_sample_cache(dataset_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"LLM evaluation on {len(samples)} samples for {dataset_name}")

    providers = {
        "openai": _create_openai_evaluator,
        "deepseek": _create_deepseek_evaluator,
        "claude": _create_claude_evaluator,
        "qwen": _create_qwen_evaluator,
        "mistral": _create_mistral_evaluator,
    }

    providers_to_eval = (
        list(providers.keys()) if provider_arg == "all" else [provider_arg]
    )

    for provider_name in providers_to_eval:
        if provider_name not in providers:
            print(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
            continue

        try:
            evaluator = providers[provider_name]()
            if evaluator is None:
                continue
            evaluator.evaluate_dataset(samples, few_shot_examples, provider_name, dataset_name=dataset_name)
        except Exception as e:
            print(f"Error evaluating {provider_name}: {e}")


def cmd_evaluate_llm(args):
    """Evaluate LLM(s) on the test set via API."""
    datasets = _ALL_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"LLM evaluation for {ds}")
        print(f"{'='*60}")
        _evaluate_llm_for_dataset(ds, args.provider)


def _create_openai_evaluator():
    from models.llm_openai import OpenAIEvaluator

    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY not set in .env — skipping OpenAI.")
        return None
    return OpenAIEvaluator(config.OPENAI_MODEL, config.OPENAI_API_KEY)


def _create_deepseek_evaluator():
    from models.llm_deepseek import DeepSeekEvaluator

    if not config.DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY not set in .env — skipping DeepSeek.")
        return None
    return DeepSeekEvaluator(config.DEEPSEEK_MODEL, config.DEEPSEEK_API_KEY)


def _create_claude_evaluator():
    from models.llm_claude import ClaudeEvaluator

    if not config.ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not set in .env — skipping Claude.")
        return None
    return ClaudeEvaluator(config.CLAUDE_MODEL, config.ANTHROPIC_API_KEY)


def _create_qwen_evaluator():
    from models.llm_qwen import QwenEvaluator

    if not config.QWEN_API_KEY:
        print("QWEN_API_KEY not set in .env — skipping Qwen.")
        return None
    return QwenEvaluator(config.QWEN_MODEL, config.QWEN_API_KEY)


def _create_mistral_evaluator():
    from models.llm_mistral import MistralEvaluator

    if not config.MISTRAL_API_KEY:
        print("MISTRAL_API_KEY not set in .env — skipping Mistral.")
        return None
    return MistralEvaluator(config.MISTRAL_MODEL, config.MISTRAL_API_KEY)


# ===================================================================
# RE (Relationship Extraction) commands — IE4Wills only
# ===================================================================

def _re_sample_cache_path():
    return config.RESULTS_DIR / "ie4wills_re_llm_samples.json"


def _save_re_sample_cache(documents, few_shot_examples):
    cache = {
        "documents": documents,
        "few_shot_examples": few_shot_examples,
    }
    path = _re_sample_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"RE sample cache saved to {path} ({len(documents)} docs, {len(few_shot_examples)} few-shot)")
    return path


def _load_re_sample_cache():
    path = _re_sample_cache_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No RE sample cache found at {path}.\n"
            f"Run 'python main.py prepare-re-samples' first."
        )
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    docs = cache["documents"]
    few_shot = cache["few_shot_examples"]
    print(f"Loaded {len(docs)} cached documents and {len(few_shot)} few-shot examples from {path}")
    return docs, few_shot


def cmd_stats_re(args):
    """Print RE dataset statistics for IE4Wills."""
    from data.re_loader import print_re_dataset_stats
    print_re_dataset_stats()


def cmd_train_re(args):
    """Fine-tune BERT model(s) for RE on IE4Wills."""
    from data.re_loader import load_re_dataset
    from models.bert_re import train_bert_re_model

    train_instances = load_re_dataset("train", negative_ratio=args.negative_ratio)
    eval_instances = load_re_dataset("validation", negative_ratio=0)

    models_to_train = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_train:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue
        train_bert_re_model(model_key, train_instances, eval_instances)


def cmd_evaluate_bert_re(args):
    """Evaluate trained BERT RE model(s) on test set."""
    from data.re_loader import load_re_dataset
    from models.bert_re import predict_bert_re

    # For BERT evaluation, include negatives to measure NO_RELATION classification
    test_instances = load_re_dataset("test", negative_ratio=1.0)

    models_to_eval = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_eval:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue
        predict_bert_re(model_key, test_instances, checkpoint_path=args.checkpoint)


def cmd_prepare_re_samples(args):
    """Prepare and cache RE test documents + few-shot examples."""
    from data.re_loader import load_re_documents, get_re_few_shot_examples

    cache_path = _re_sample_cache_path()
    if cache_path.exists() and not args.resample:
        print(f"RE sample cache already exists at {cache_path}")
        print("Use --resample to regenerate.")
        return

    documents = load_re_documents(split="test")
    few_shot = get_re_few_shot_examples(n=config.RE_FEW_SHOT_COUNT)
    _save_re_sample_cache(documents, few_shot)


def cmd_evaluate_llm_re(args):
    """Evaluate LLM(s) for RE using cached documents."""
    from models.llm_re_providers import (
        OpenAIREEvaluator, DeepSeekREEvaluator, ClaudeREEvaluator,
        MistralREEvaluator, QwenREEvaluator,
    )

    try:
        documents, few_shot = _load_re_sample_cache()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"RE LLM evaluation on {len(documents)} documents")

    providers = {
        "openai": lambda: _create_re_provider(OpenAIREEvaluator, config.OPENAI_MODEL, config.OPENAI_API_KEY, "OPENAI_API_KEY"),
        "deepseek": lambda: _create_re_provider(DeepSeekREEvaluator, config.DEEPSEEK_MODEL, config.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY"),
        "claude": lambda: _create_re_provider(ClaudeREEvaluator, config.CLAUDE_MODEL, config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
        "mistral": lambda: _create_re_provider(MistralREEvaluator, config.MISTRAL_MODEL, config.MISTRAL_API_KEY, "MISTRAL_API_KEY"),
        "qwen": lambda: _create_re_provider(QwenREEvaluator, config.QWEN_MODEL, config.QWEN_API_KEY, "QWEN_API_KEY"),
    }

    providers_to_eval = (
        list(providers.keys()) if args.provider == "all" else [args.provider]
    )

    for provider_name in providers_to_eval:
        if provider_name not in providers:
            print(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
            continue
        try:
            evaluator = providers[provider_name]()
            if evaluator is None:
                continue
            evaluator.evaluate_dataset(documents, few_shot, provider_name)
        except Exception as e:
            print(f"Error evaluating RE {provider_name}: {e}")


def _create_re_provider(cls, model_id, api_key, key_name):
    if not api_key:
        print(f"{key_name} not set in .env — skipping.")
        return None
    return cls(model_id, api_key)


# ===================================================================
# Event Detection commands
# ===================================================================

_EVENT_DATASETS = ["events_matter"]
_CONTRACT_EVENT_DATASETS = ["contractual_events"]


def _event_sample_cache_path(dataset_name: str):
    return config.RESULTS_DIR / f"{dataset_name}_event_llm_samples.json"


def _save_event_sample_cache(dataset_name, samples, few_shot_examples, n_samples, n_few_shot):
    cache = {
        "dataset_name": dataset_name,
        "task": "event",
        "n_samples": n_samples,
        "n_few_shot": n_few_shot,
        "samples": [
            {
                "sentence": " ".join(s["tokens"]),
                "tokens": s["tokens"],
                "tags": s["tags"],
            }
            for s in samples
        ],
        "few_shot_examples": [
            {
                "sentence": " ".join(e["tokens"]),
                "tokens": e["tokens"],
                "tags": e["tags"],
            }
            for e in few_shot_examples
        ],
    }
    path = _event_sample_cache_path(dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Event sample cache saved to {path} ({n_samples} samples, {n_few_shot} few-shot)")
    return path


def _load_event_sample_cache(dataset_name):
    path = _event_sample_cache_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No event sample cache found at {path}.\n"
            f"Run 'python main.py prepare-event-samples --dataset {dataset_name}' first."
        )
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    samples = [{"tokens": s["tokens"], "tags": s["tags"]} for s in cache["samples"]]
    few_shot = [{"tokens": e["tokens"], "tags": e["tags"]} for e in cache["few_shot_examples"]]
    print(f"Loaded {len(samples)} cached event samples and {len(few_shot)} few-shot examples from {path}")
    return samples, few_shot


def cmd_stats_event(args):
    """Print event detection dataset statistics."""
    from data.event_loader import load_event_dataset, print_event_dataset_stats
    dataset, tag_column = load_event_dataset(args.dataset)
    print_event_dataset_stats(dataset, tag_column)


def cmd_train_event(args):
    """Fine-tune BERT model(s) for event detection."""
    from data.event_loader import load_event_dataset
    from models.bert_event import train_bert_event_model

    dataset, tag_column = load_event_dataset(args.dataset)

    models_to_train = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_train:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue

        train_bert_event_model(
            model_key=model_key,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tag_column=tag_column,
            dataset_name=args.dataset,
        )


def cmd_evaluate_bert_event(args):
    """Evaluate trained BERT event detection model(s) on test set."""
    from data.event_loader import load_event_dataset
    from models.bert_event import predict_bert_event

    dataset, tag_column = load_event_dataset(args.dataset)

    models_to_eval = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_eval:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue

        predict_bert_event(
            model_key=model_key,
            test_dataset=dataset["test"],
            tag_column=tag_column,
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset,
        )


def cmd_prepare_event_samples(args):
    """Prepare and cache event detection test samples for LLM evaluation."""
    from data.event_loader import load_event_dataset, get_event_llm_samples, get_event_few_shot_examples

    datasets = _EVENT_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Preparing event samples for {ds}")
        print(f"{'='*60}")

        cache_path = _event_sample_cache_path(ds)
        if cache_path.exists() and not args.resample:
            print(f"Event sample cache already exists at {cache_path}")
            print("Use --resample to regenerate.")
            continue

        dataset, tag_column = load_event_dataset(ds)
        samples = get_event_llm_samples(dataset, split="test", n=None, tag_column=tag_column)
        few_shot = get_event_few_shot_examples(dataset, n=config.EVENT_FEW_SHOT_COUNT, tag_column=tag_column)
        _save_event_sample_cache(ds, samples, few_shot, len(samples), len(few_shot))


def cmd_evaluate_llm_event(args):
    """Evaluate LLM(s) for event detection using cached samples."""
    from data.event_schema import set_active_event_schema
    from models.llm_event_providers import (
        OpenAIEventEvaluator, DeepSeekEventEvaluator, ClaudeEventEvaluator,
        MistralEventEvaluator, QwenEventEvaluator,
    )

    datasets = _EVENT_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Event Detection LLM evaluation for {ds}")
        print(f"{'='*60}")

        set_active_event_schema(ds)

        try:
            samples, few_shot = _load_event_sample_cache(ds)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        print(f"Event LLM evaluation on {len(samples)} samples for {ds}")

        providers = {
            "openai": lambda: _create_re_provider(OpenAIEventEvaluator, config.OPENAI_MODEL, config.OPENAI_API_KEY, "OPENAI_API_KEY"),
            "deepseek": lambda: _create_re_provider(DeepSeekEventEvaluator, config.DEEPSEEK_MODEL, config.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY"),
            "claude": lambda: _create_re_provider(ClaudeEventEvaluator, config.CLAUDE_MODEL, config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
            "mistral": lambda: _create_re_provider(MistralEventEvaluator, config.MISTRAL_MODEL, config.MISTRAL_API_KEY, "MISTRAL_API_KEY"),
            "qwen": lambda: _create_re_provider(QwenEventEvaluator, config.QWEN_MODEL, config.QWEN_API_KEY, "QWEN_API_KEY"),
        }

        providers_to_eval = (
            list(providers.keys()) if args.provider == "all" else [args.provider]
        )

        for provider_name in providers_to_eval:
            if provider_name not in providers:
                print(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
                continue
            try:
                evaluator = providers[provider_name]()
                if evaluator is None:
                    continue
                evaluator.evaluate_dataset(samples, few_shot, provider_name, dataset_name=ds)
            except Exception as e:
                print(f"Error evaluating event {provider_name}: {e}")


# ===================================================================
# Contract Event Detection commands
# ===================================================================

def _contract_event_sample_cache_path(dataset_name: str):
    return config.RESULTS_DIR / f"{dataset_name}_event_llm_samples.json"


def _save_contract_event_sample_cache(dataset_name, samples, few_shot_examples, n_samples, n_few_shot):
    cache = {
        "dataset_name": dataset_name,
        "task": "contract_event",
        "n_samples": n_samples,
        "n_few_shot": n_few_shot,
        "samples": [
            {
                "sentence": " ".join(s["tokens"]),
                "tokens": s["tokens"],
                "tags": s["tags"],
            }
            for s in samples
        ],
        "few_shot_examples": [
            {
                "sentence": " ".join(e["tokens"]),
                "tokens": e["tokens"],
                "tags": e["tags"],
            }
            for e in few_shot_examples
        ],
    }
    path = _contract_event_sample_cache_path(dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Contract event sample cache saved to {path} ({n_samples} samples, {n_few_shot} few-shot)")
    return path


def _load_contract_event_sample_cache(dataset_name):
    path = _contract_event_sample_cache_path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No contract event sample cache found at {path}.\n"
            f"Run 'python main.py prepare-contract-event-samples --dataset {dataset_name}' first."
        )
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)
    samples = [{"tokens": s["tokens"], "tags": s["tags"]} for s in cache["samples"]]
    few_shot = [{"tokens": e["tokens"], "tags": e["tags"]} for e in cache["few_shot_examples"]]
    print(f"Loaded {len(samples)} cached contract event samples and {len(few_shot)} few-shot examples from {path}")
    return samples, few_shot


def cmd_stats_contract_event(args):
    """Print contract event detection dataset statistics."""
    from data.contract_event_loader import load_contract_event_dataset, print_contract_event_dataset_stats
    dataset, tag_column = load_contract_event_dataset(args.dataset)
    print_contract_event_dataset_stats(dataset, tag_column)


def cmd_train_contract_event(args):
    """Fine-tune BERT model(s) for contract event detection."""
    from data.contract_event_loader import load_contract_event_dataset
    from models.bert_contract_event import train_bert_contract_event_model

    dataset, tag_column = load_contract_event_dataset(args.dataset)

    models_to_train = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_train:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue

        train_bert_contract_event_model(
            model_key=model_key,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tag_column=tag_column,
            dataset_name=args.dataset,
        )


def cmd_evaluate_bert_contract_event(args):
    """Evaluate trained BERT contract event detection model(s) on test set."""
    from data.contract_event_loader import load_contract_event_dataset
    from models.bert_contract_event import predict_bert_contract_event

    dataset, tag_column = load_contract_event_dataset(args.dataset)

    models_to_eval = (
        list(config.BERT_MODELS.keys()) if args.model == "all" else [args.model]
    )

    for model_key in models_to_eval:
        if model_key not in config.BERT_MODELS:
            print(f"Unknown model: {model_key}. Available: {list(config.BERT_MODELS.keys())}")
            continue

        predict_bert_contract_event(
            model_key=model_key,
            test_dataset=dataset["test"],
            tag_column=tag_column,
            checkpoint_path=args.checkpoint,
            dataset_name=args.dataset,
        )


def cmd_prepare_contract_event_samples(args):
    """Prepare and cache contract event detection test samples for LLM evaluation."""
    from data.contract_event_loader import (
        load_contract_event_dataset,
        get_contract_event_llm_samples,
        get_contract_event_few_shot_examples,
    )

    datasets = _CONTRACT_EVENT_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Preparing contract event samples for {ds}")
        print(f"{'='*60}")

        cache_path = _contract_event_sample_cache_path(ds)
        if cache_path.exists() and not args.resample:
            print(f"Contract event sample cache already exists at {cache_path}")
            print("Use --resample to regenerate.")
            continue

        dataset, tag_column = load_contract_event_dataset(ds)
        n = None if args.full_test else args.sample_size
        samples = get_contract_event_llm_samples(dataset, split="test", n=n, tag_column=tag_column)
        few_shot = get_contract_event_few_shot_examples(
            dataset, n=config.CONTRACT_EVENT_FEW_SHOT_COUNT, tag_column=tag_column
        )
        _save_contract_event_sample_cache(ds, samples, few_shot, len(samples), len(few_shot))


def cmd_evaluate_llm_contract_event(args):
    """Evaluate LLM(s) for contract event detection using cached samples."""
    from data.contract_event_schema import set_active_contract_event_schema
    from models.llm_contract_event_providers import (
        OpenAIContractEventEvaluator, DeepSeekContractEventEvaluator,
        ClaudeContractEventEvaluator, MistralContractEventEvaluator,
        QwenContractEventEvaluator,
    )

    datasets = _CONTRACT_EVENT_DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Contract Event Detection LLM evaluation for {ds}")
        print(f"{'='*60}")

        set_active_contract_event_schema(ds)

        try:
            samples, few_shot = _load_contract_event_sample_cache(ds)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        print(f"Contract Event LLM evaluation on {len(samples)} samples for {ds}")

        providers = {
            "openai": lambda: _create_re_provider(OpenAIContractEventEvaluator, config.OPENAI_MODEL, config.OPENAI_API_KEY, "OPENAI_API_KEY"),
            "deepseek": lambda: _create_re_provider(DeepSeekContractEventEvaluator, config.DEEPSEEK_MODEL, config.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY"),
            "claude": lambda: _create_re_provider(ClaudeContractEventEvaluator, config.CLAUDE_MODEL, config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY"),
            "mistral": lambda: _create_re_provider(MistralContractEventEvaluator, config.MISTRAL_MODEL, config.MISTRAL_API_KEY, "MISTRAL_API_KEY"),
            "qwen": lambda: _create_re_provider(QwenContractEventEvaluator, config.QWEN_MODEL, config.QWEN_API_KEY, "QWEN_API_KEY"),
        }

        providers_to_eval = (
            list(providers.keys()) if args.provider == "all" else [args.provider]
        )

        for provider_name in providers_to_eval:
            if provider_name not in providers:
                print(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
                continue
            try:
                evaluator = providers[provider_name]()
                if evaluator is None:
                    continue
                evaluator.evaluate_dataset(samples, few_shot, provider_name, dataset_name=ds)
            except Exception as e:
                print(f"Error evaluating contract event {provider_name}: {e}")


def cmd_compare(args):
    """Generate comparison report from saved results."""
    from evaluation.comparison import generate_comparison_report

    generate_comparison_report()


def cmd_export_latex(args):
    """Export results as publication-ready LaTeX tables."""
    from evaluation.latex_export import generate_latex_report

    datasets = args.datasets.split(",")
    output_path = config.RESULTS_DIR / args.output
    generate_latex_report(output_path, datasets)


def cmd_run_all(args):
    """Run the full pipeline: train BERT + evaluate LLMs + compare."""
    # Train all BERT models
    args.model = "all"
    cmd_train(args)

    # Evaluate all BERT models on test set
    args.checkpoint = None
    cmd_evaluate_bert(args)

    # Prepare LLM samples
    args.sample_size = config.LLM_SAMPLE_SIZE
    args.full_test = False
    args.resample = False
    cmd_prepare_samples(args)

    # Evaluate all LLMs
    args.provider = "all"
    cmd_evaluate_llm(args)

    # Generate comparison
    cmd_compare(args)


def cmd_stats(args):
    """Print dataset statistics."""
    dataset, tag_column = _init_dataset(args)
    print_dataset_stats(dataset, tag_column)


def _add_dataset_arg(parser, allow_all=False):
    """Add the --dataset argument to a subparser."""
    choices = _ALL_DATASETS + ["all"] if allow_all else list(_ALL_DATASETS)
    help_text = (
        "Dataset to use (default: inlegalner). Use 'all' to process all datasets."
        if allow_all
        else "Dataset to use: 'inlegalner' (default), 'icdac', or 'ie4wills'"
    )
    parser.add_argument(
        "--dataset",
        default="inlegalner",
        choices=choices,
        help=help_text,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Legal NER Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py stats                                  # Print dataset stats
  python main.py stats --dataset icdac                  # ICDAC dataset stats
  python main.py train --model bert-base-uncased        # Fine-tune one BERT model
  python main.py train --model all                      # Fine-tune all BERT models
  python main.py train --model all --dataset icdac      # Fine-tune on ICDAC
  python main.py evaluate-bert --model all              # Evaluate all BERT models
  python main.py prepare-samples --dataset all           # Cache LLM test samples for all datasets
  python main.py prepare-samples --dataset inlegalner   # Cache LLM test samples for one dataset
  python main.py evaluate-llm --provider openai         # Evaluate ChatGPT (uses cache)
  python main.py evaluate-llm --dataset all             # Evaluate all LLMs on all datasets
  python main.py compare                                # Generate comparison report
  python main.py export-latex                           # Generate LaTeX tables
  python main.py run-all                                # Full pipeline
  python main.py run-all --dataset icdac                # Full pipeline on ICDAC

  # Relationship Extraction (IE4Wills only):
  python main.py stats-re                               # Print RE dataset statistics
  python main.py train-re --model all                   # Fine-tune BERT RE models
  python main.py evaluate-bert-re --model all           # Evaluate BERT RE models
  python main.py prepare-re-samples                     # Cache RE test documents
  python main.py evaluate-llm-re --provider openai      # Evaluate LLM for RE
  python main.py evaluate-llm-re --provider all         # Evaluate all LLMs for RE

  # Event Detection (Events Matter):
  python main.py stats-event                            # Print event dataset statistics
  python main.py train-event --model all                # Fine-tune BERT event models
  python main.py evaluate-bert-event --model all        # Evaluate BERT event models
  python main.py prepare-event-samples                  # Cache event test samples
  python main.py evaluate-llm-event --provider openai   # Evaluate LLM for events
  python main.py evaluate-llm-event --provider all      # Evaluate all LLMs for events

  # Contract Event Detection (Contractual Events):
  python main.py stats-contract-event                            # Print contract event dataset statistics
  python main.py train-contract-event --model legal-bert         # Fine-tune BERT contract event model
  python main.py evaluate-bert-contract-event --model legal-bert # Evaluate BERT contract event model
  python main.py prepare-contract-event-samples                  # Cache contract event test samples
  python main.py evaluate-llm-contract-event --provider openai   # Evaluate LLM for contract events
  python main.py evaluate-llm-contract-event --provider all      # Evaluate all LLMs for contract events
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # stats
    sub = subparsers.add_parser("stats", help="Print dataset statistics")
    _add_dataset_arg(sub)
    sub.set_defaults(func=cmd_stats)

    # train
    sub = subparsers.add_parser("train", help="Fine-tune BERT model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to train: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    _add_dataset_arg(sub)
    sub.set_defaults(func=cmd_train)

    # evaluate-bert
    sub = subparsers.add_parser("evaluate-bert", help="Evaluate trained BERT model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to evaluate: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    _add_dataset_arg(sub)
    sub.set_defaults(func=cmd_evaluate_bert)

    # prepare-samples
    sub = subparsers.add_parser("prepare-samples", help="Cache LLM test samples for consistent evaluation")
    sub.add_argument(
        "--sample-size",
        type=int,
        default=config.LLM_SAMPLE_SIZE,
        help=f"Number of test samples (default: {config.LLM_SAMPLE_SIZE})",
    )
    sub.add_argument(
        "--full-test",
        action="store_true",
        help="Use the full test set",
    )
    sub.add_argument(
        "--resample",
        action="store_true",
        help="Regenerate cache even if it already exists",
    )
    _add_dataset_arg(sub, allow_all=True)
    sub.set_defaults(func=cmd_prepare_samples)

    # evaluate-llm
    sub = subparsers.add_parser("evaluate-llm", help="Evaluate LLM(s) via API (requires prepare-samples first)")
    sub.add_argument(
        "--provider",
        default="all",
        choices=["openai", "deepseek", "claude", "qwen", "mistral", "all"],
        help="LLM provider to evaluate",
    )
    _add_dataset_arg(sub, allow_all=True)
    sub.set_defaults(func=cmd_evaluate_llm)

    # compare
    sub = subparsers.add_parser("compare", help="Generate comparison report")
    sub.set_defaults(func=cmd_compare)

    # export-latex
    sub = subparsers.add_parser("export-latex", help="Export results as LaTeX tables")
    sub.add_argument(
        "--datasets",
        default="inlegalner,icdac,ie4wills,events_matter,contractual_events",
        help="Comma-separated dataset names (default: inlegalner,icdac,ie4wills,events_matter,contractual_events)",
    )
    sub.add_argument(
        "--output",
        default="latex_tables.tex",
        help="Output filename (default: latex_tables.tex)",
    )
    sub.set_defaults(func=cmd_export_latex)

    # run-all
    sub = subparsers.add_parser("run-all", help="Run full pipeline")
    _add_dataset_arg(sub)
    sub.set_defaults(func=cmd_run_all)

    # --- RE subcommands (IE4Wills only) ---

    # stats-re
    sub = subparsers.add_parser("stats-re", help="Print RE dataset statistics for IE4Wills")
    sub.set_defaults(func=cmd_stats_re)

    # train-re
    sub = subparsers.add_parser("train-re", help="Fine-tune BERT model(s) for RE on IE4Wills")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to train: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument(
        "--negative-ratio",
        type=float,
        default=config.RE_NEGATIVE_RATIO,
        help=f"Ratio of negative to positive examples (default: {config.RE_NEGATIVE_RATIO})",
    )
    sub.set_defaults(func=cmd_train_re)

    # evaluate-bert-re
    sub = subparsers.add_parser("evaluate-bert-re", help="Evaluate trained BERT RE model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to evaluate: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    sub.set_defaults(func=cmd_evaluate_bert_re)

    # prepare-re-samples
    sub = subparsers.add_parser("prepare-re-samples", help="Cache RE test documents + few-shot examples")
    sub.add_argument("--resample", action="store_true", help="Regenerate cache")
    sub.set_defaults(func=cmd_prepare_re_samples)

    # evaluate-llm-re
    sub = subparsers.add_parser("evaluate-llm-re", help="Evaluate LLM(s) for RE on IE4Wills")
    sub.add_argument(
        "--provider",
        default="all",
        choices=["openai", "deepseek", "claude", "mistral", "qwen", "all"],
        help="LLM provider to evaluate",
    )
    sub.set_defaults(func=cmd_evaluate_llm_re)

    # --- Event Detection subcommands ---

    # stats-event
    sub = subparsers.add_parser("stats-event", help="Print event detection dataset statistics")
    sub.add_argument(
        "--dataset",
        default="events_matter",
        choices=_EVENT_DATASETS,
        help="Event dataset to use (default: events_matter)",
    )
    sub.set_defaults(func=cmd_stats_event)

    # train-event
    sub = subparsers.add_parser("train-event", help="Fine-tune BERT model(s) for event detection")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to train: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument(
        "--dataset",
        default="events_matter",
        choices=_EVENT_DATASETS,
        help="Event dataset to use (default: events_matter)",
    )
    sub.set_defaults(func=cmd_train_event)

    # evaluate-bert-event
    sub = subparsers.add_parser("evaluate-bert-event", help="Evaluate trained BERT event detection model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to evaluate: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    sub.add_argument(
        "--dataset",
        default="events_matter",
        choices=_EVENT_DATASETS,
        help="Event dataset to use (default: events_matter)",
    )
    sub.set_defaults(func=cmd_evaluate_bert_event)

    # prepare-event-samples
    sub = subparsers.add_parser("prepare-event-samples", help="Cache event detection test samples for LLM evaluation")
    sub.add_argument("--resample", action="store_true", help="Regenerate cache")
    sub.add_argument(
        "--dataset",
        default="events_matter",
        choices=_EVENT_DATASETS + ["all"],
        help="Event dataset to use (default: events_matter)",
    )
    sub.set_defaults(func=cmd_prepare_event_samples)

    # evaluate-llm-event
    sub = subparsers.add_parser("evaluate-llm-event", help="Evaluate LLM(s) for event detection")
    sub.add_argument(
        "--provider",
        default="all",
        choices=["openai", "deepseek", "claude", "mistral", "qwen", "all"],
        help="LLM provider to evaluate",
    )
    sub.add_argument(
        "--dataset",
        default="events_matter",
        choices=_EVENT_DATASETS + ["all"],
        help="Event dataset to use (default: events_matter)",
    )
    sub.set_defaults(func=cmd_evaluate_llm_event)

    # --- Contract Event Detection subcommands ---

    # stats-contract-event
    sub = subparsers.add_parser("stats-contract-event", help="Print contract event detection dataset statistics")
    sub.add_argument(
        "--dataset",
        default="contractual_events",
        choices=_CONTRACT_EVENT_DATASETS,
        help="Contract event dataset to use (default: contractual_events)",
    )
    sub.set_defaults(func=cmd_stats_contract_event)

    # train-contract-event
    sub = subparsers.add_parser("train-contract-event", help="Fine-tune BERT model(s) for contract event detection")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to train: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument(
        "--dataset",
        default="contractual_events",
        choices=_CONTRACT_EVENT_DATASETS,
        help="Contract event dataset to use (default: contractual_events)",
    )
    sub.set_defaults(func=cmd_train_contract_event)

    # evaluate-bert-contract-event
    sub = subparsers.add_parser("evaluate-bert-contract-event", help="Evaluate trained BERT contract event detection model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to evaluate: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    sub.add_argument(
        "--dataset",
        default="contractual_events",
        choices=_CONTRACT_EVENT_DATASETS,
        help="Contract event dataset to use (default: contractual_events)",
    )
    sub.set_defaults(func=cmd_evaluate_bert_contract_event)

    # prepare-contract-event-samples
    sub = subparsers.add_parser("prepare-contract-event-samples", help="Cache contract event detection test samples for LLM evaluation")
    sub.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of test samples for LLM evaluation (default: 100)",
    )
    sub.add_argument(
        "--full-test",
        action="store_true",
        help="Use the full test set instead of sampling",
    )
    sub.add_argument("--resample", action="store_true", help="Regenerate cache")
    sub.add_argument(
        "--dataset",
        default="contractual_events",
        choices=_CONTRACT_EVENT_DATASETS + ["all"],
        help="Contract event dataset to use (default: contractual_events)",
    )
    sub.set_defaults(func=cmd_prepare_contract_event_samples)

    # evaluate-llm-contract-event
    sub = subparsers.add_parser("evaluate-llm-contract-event", help="Evaluate LLM(s) for contract event detection")
    sub.add_argument(
        "--provider",
        default="all",
        choices=["openai", "deepseek", "claude", "mistral", "qwen", "all"],
        help="LLM provider to evaluate",
    )
    sub.add_argument(
        "--dataset",
        default="contractual_events",
        choices=_CONTRACT_EVENT_DATASETS + ["all"],
        help="Contract event dataset to use (default: contractual_events)",
    )
    sub.set_defaults(func=cmd_evaluate_llm_contract_event)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

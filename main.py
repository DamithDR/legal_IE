"""Legal NER Evaluation Pipeline — CLI entry point."""

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
        choices=["openai", "deepseek", "claude", "all"],
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
        default="inlegalner,icdac,ie4wills",
        help="Comma-separated dataset names (default: inlegalner,icdac,ie4wills)",
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

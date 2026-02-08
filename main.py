"""Legal NER Evaluation Pipeline — CLI entry point."""

import argparse
import sys

import config
from data.loader import load_ner_dataset, get_few_shot_examples, get_llm_samples, print_dataset_stats
from data.label_schema import LABEL_LIST, id2label, label2id


def cmd_train(args):
    """Fine-tune BERT model(s) on the training set."""
    from models.bert_ner import train_bert_model

    dataset, tag_column = load_ner_dataset()

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
        )


def cmd_evaluate_bert(args):
    """Evaluate a trained BERT model on the test set."""
    from models.bert_ner import predict_bert

    dataset, tag_column = load_ner_dataset()

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
        )


def cmd_evaluate_llm(args):
    """Evaluate LLM(s) on the test set via API."""
    dataset, tag_column = load_ner_dataset()

    # Get test samples
    sample_size = None if args.full_test else args.sample_size
    test_split = "test" if "test" in dataset else "validation"
    samples = get_llm_samples(dataset, split=test_split, n=sample_size, tag_column=tag_column)
    print(f"LLM evaluation on {len(samples)} samples")

    # Get few-shot examples
    few_shot_examples = get_few_shot_examples(dataset, n=config.LLM_FEW_SHOT_COUNT, tag_column=tag_column)

    providers = {
        "openai": _create_openai_evaluator,
        "deepseek": _create_deepseek_evaluator,
        "claude": _create_claude_evaluator,
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
            evaluator.evaluate_dataset(samples, few_shot_examples, provider_name)
        except Exception as e:
            print(f"Error evaluating {provider_name}: {e}")


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


def cmd_run_all(args):
    """Run the full pipeline: train BERT + evaluate LLMs + compare."""
    # Train all BERT models
    args.model = "all"
    cmd_train(args)

    # Evaluate all BERT models on test set
    args.checkpoint = None
    cmd_evaluate_bert(args)

    # Evaluate all LLMs
    args.provider = "all"
    args.sample_size = config.LLM_SAMPLE_SIZE
    args.full_test = False
    cmd_evaluate_llm(args)

    # Generate comparison
    cmd_compare(args)


def cmd_stats(args):
    """Print dataset statistics."""
    dataset, tag_column = load_ner_dataset()
    print_dataset_stats(dataset, tag_column)


def main():
    parser = argparse.ArgumentParser(
        description="Legal NER Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py stats                                  # Print dataset stats
  python main.py train --model bert-base-uncased        # Fine-tune one BERT model
  python main.py train --model all                      # Fine-tune all BERT models
  python main.py evaluate-bert --model all              # Evaluate all BERT models
  python main.py evaluate-llm --provider openai         # Evaluate ChatGPT
  python main.py evaluate-llm --provider all            # Evaluate all LLMs
  python main.py evaluate-llm --provider all --full-test  # Full test set
  python main.py compare                                # Generate comparison report
  python main.py run-all                                # Full pipeline
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # stats
    sub = subparsers.add_parser("stats", help="Print dataset statistics")
    sub.set_defaults(func=cmd_stats)

    # train
    sub = subparsers.add_parser("train", help="Fine-tune BERT model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to train: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.set_defaults(func=cmd_train)

    # evaluate-bert
    sub = subparsers.add_parser("evaluate-bert", help="Evaluate trained BERT model(s)")
    sub.add_argument(
        "--model",
        default="all",
        help=f"Model to evaluate: {list(config.BERT_MODELS.keys())} or 'all'",
    )
    sub.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    sub.set_defaults(func=cmd_evaluate_bert)

    # evaluate-llm
    sub = subparsers.add_parser("evaluate-llm", help="Evaluate LLM(s) via API")
    sub.add_argument(
        "--provider",
        default="all",
        choices=["openai", "deepseek", "claude", "all"],
        help="LLM provider to evaluate",
    )
    sub.add_argument(
        "--sample-size",
        type=int,
        default=config.LLM_SAMPLE_SIZE,
        help=f"Number of test samples (default: {config.LLM_SAMPLE_SIZE})",
    )
    sub.add_argument(
        "--full-test",
        action="store_true",
        help="Evaluate on the full test set",
    )
    sub.set_defaults(func=cmd_evaluate_llm)

    # compare
    sub = subparsers.add_parser("compare", help="Generate comparison report")
    sub.set_defaults(func=cmd_compare)

    # run-all
    sub = subparsers.add_parser("run-all", help="Run full pipeline")
    sub.set_defaults(func=cmd_run_all)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

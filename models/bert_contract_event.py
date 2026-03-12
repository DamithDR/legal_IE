"""BERT-based Contract Event Detection fine-tuning and inference pipeline."""

import numpy as np
from functools import partial

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import config
from data import contract_event_schema
from data.loader import tokenize_and_align_labels
from evaluation.metrics import compute_metrics_for_trainer, compute_ner_metrics


def train_bert_contract_event_model(
    model_key: str,
    train_dataset,
    eval_dataset,
    tag_column: str = "event_tags",
    dataset_name: str = "contractual_events",
):
    """
    Fine-tune a BERT model on the contract event detection task.

    Args:
        model_key: Key into config.BERT_MODELS (e.g. 'legal-bert').
        train_dataset: HuggingFace Dataset (train split).
        eval_dataset: HuggingFace Dataset (validation split).
        tag_column: Name of the event tag column.
        dataset_name: Dataset identifier for namespacing outputs.

    Returns:
        Tuple of (trainer, tokenizer, eval_results).
    """
    model_name_or_path = config.BERT_MODELS[model_key]
    output_dir = config.CHECKPOINTS_DIR / f"{dataset_name}_event_{model_key}"

    print(f"\n{'='*60}")
    print(f"Training Contract Event Detection: {model_key} ({model_name_or_path})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(contract_event_schema.LABEL_LIST),
        id2label=contract_event_schema.id2label,
        label2id=contract_event_schema.label2id,
    )

    max_length = config.CONTRACT_EVENT_TRAIN_CONFIG["max_seq_length"]
    tokenize_fn = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        tag_column=tag_column,
    )

    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_preds):
        return compute_metrics_for_trainer(eval_preds, contract_event_schema.id2label)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.CONTRACT_EVENT_TRAIN_CONFIG["learning_rate"],
        num_train_epochs=config.CONTRACT_EVENT_TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=config.CONTRACT_EVENT_TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=config.CONTRACT_EVENT_TRAIN_CONFIG["per_device_eval_batch_size"],
        weight_decay=config.CONTRACT_EVENT_TRAIN_CONFIG["weight_decay"],
        warmup_steps=config.CONTRACT_EVENT_TRAIN_CONFIG["warmup_steps"],
        eval_strategy=config.CONTRACT_EVENT_TRAIN_CONFIG["eval_strategy"],
        save_strategy=config.CONTRACT_EVENT_TRAIN_CONFIG["save_strategy"],
        load_best_model_at_end=config.CONTRACT_EVENT_TRAIN_CONFIG["load_best_model_at_end"],
        metric_for_best_model=config.CONTRACT_EVENT_TRAIN_CONFIG["metric_for_best_model"],
        greater_is_better=True,
        seed=config.CONTRACT_EVENT_TRAIN_CONFIG["seed"],
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    eval_results = trainer.evaluate()
    print(f"\nEval results for {model_key}:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    return trainer, tokenizer, eval_results


def predict_bert_contract_event(
    model_key: str,
    test_dataset,
    tag_column: str = "event_tags",
    checkpoint_path: str = None,
    dataset_name: str = "contractual_events",
):
    """
    Run inference on the test set with a trained BERT contract event detection model.

    Args:
        model_key: Key into config.BERT_MODELS.
        test_dataset: HuggingFace Dataset (test split).
        tag_column: Name of the event tag column.
        checkpoint_path: Path to checkpoint (default: checkpoints/{dataset_name}_event_{model_key}/best_model).
        dataset_name: Dataset identifier for namespacing outputs.

    Returns:
        Dict with evaluation metrics.
    """
    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINTS_DIR / f"{dataset_name}_event_{model_key}" / "best_model")

    print(f"\nEvaluating contract event detection {model_key} from {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path, local_files_only=True)

    max_length = config.CONTRACT_EVENT_TRAIN_CONFIG["max_seq_length"]
    tokenize_fn = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        tag_column=tag_column,
    )

    tokenized_test = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    predictions_output = trainer.predict(tokenized_test)
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(contract_event_schema.id2label[l])
                pred_seq_filtered.append(contract_event_schema.id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    results = compute_ner_metrics(true_labels, pred_labels)

    print(f"\nContract Event Detection Results for {model_key}:")
    print(results["classification_report_str"])

    save_data = {
        "model_name": model_key,
        "model_type": "bert",
        "task": "event",
        "dataset": dataset_name,
        "num_samples_evaluated": len(true_labels),
        "overall": results["overall"],
        "macro_avg": results["macro_avg"],
        "per_entity": results["per_entity"],
    }
    config.save_results(f"{dataset_name}_event_{model_key}", save_data)

    return save_data

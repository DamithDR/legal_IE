"""BERT-based NER fine-tuning and inference pipeline."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import config
from data.label_schema import id2label, label2id, LABEL_LIST
from data.loader import tokenize_and_align_labels
from evaluation.metrics import compute_metrics_for_trainer, compute_ner_metrics


def train_bert_model(
    model_key: str,
    train_dataset,
    eval_dataset,
    tag_column: str = "ner_tags",
):
    """
    Fine-tune a BERT model on the NER task.

    Args:
        model_key: Key into config.BERT_MODELS (e.g. 'bert-base-uncased').
        train_dataset: HuggingFace Dataset (train split).
        eval_dataset: HuggingFace Dataset (validation split).
        tag_column: Name of the NER tag column.

    Returns:
        Tuple of (trainer, tokenizer, eval_results).
    """
    model_name_or_path = config.BERT_MODELS[model_key]
    output_dir = config.CHECKPOINTS_DIR / model_key

    print(f"\n{'='*60}")
    print(f"Training: {model_key} ({model_name_or_path})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize datasets
    max_length = config.TRAIN_CONFIG["max_seq_length"]
    tokenize_fn = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        tag_column=tag_column,
    )

    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Compute metrics wrapper
    def compute_metrics(eval_preds):
        return compute_metrics_for_trainer(eval_preds, id2label)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.TRAIN_CONFIG["learning_rate"],
        num_train_epochs=config.TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=config.TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=config.TRAIN_CONFIG["per_device_eval_batch_size"],
        weight_decay=config.TRAIN_CONFIG["weight_decay"],
        warmup_ratio=config.TRAIN_CONFIG["warmup_ratio"],
        eval_strategy=config.TRAIN_CONFIG["eval_strategy"],
        save_strategy=config.TRAIN_CONFIG["save_strategy"],
        load_best_model_at_end=config.TRAIN_CONFIG["load_best_model_at_end"],
        metric_for_best_model=config.TRAIN_CONFIG["metric_for_best_model"],
        greater_is_better=True,
        seed=config.TRAIN_CONFIG["seed"],
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save best model
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\nEval results for {model_key}:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    return trainer, tokenizer, eval_results


def predict_bert(
    model_key: str,
    test_dataset,
    tag_column: str = "ner_tags",
    checkpoint_path: str = None,
):
    """
    Run inference on the test set with a trained BERT model.

    Args:
        model_key: Key into config.BERT_MODELS.
        test_dataset: HuggingFace Dataset (test split).
        tag_column: Name of the NER tag column.
        checkpoint_path: Path to checkpoint (default: checkpoints/{model_key}/best_model).

    Returns:
        Dict with evaluation metrics.
    """
    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINTS_DIR / model_key / "best_model")

    print(f"\nEvaluating {model_key} from {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)

    max_length = config.TRAIN_CONFIG["max_seq_length"]
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
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Get predictions
    predictions_output = trainer.predict(tokenized_test)
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    # Convert to string tags, filtering out -100
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(id2label[l])
                pred_seq_filtered.append(id2label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    # Compute metrics
    results = compute_ner_metrics(true_labels, pred_labels)

    print(f"\nResults for {model_key}:")
    print(results["classification_report_str"])

    # Save results
    save_data = {
        "model_name": model_key,
        "model_type": "bert",
        "num_samples_evaluated": len(true_labels),
        "overall": results["overall"],
        "macro_avg": results["macro_avg"],
        "per_entity": results["per_entity"],
    }
    config.save_results(model_key, save_data)

    return save_data

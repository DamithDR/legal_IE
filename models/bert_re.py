"""BERT-based Relationship Extraction with entity markers."""

from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config
from data.relation_schema import (
    NUM_RELATION_LABELS,
    re_id2label,
    re_label2id,
)
from evaluation.re_metrics import compute_re_metrics_for_trainer, compute_re_metrics_from_labels

# Entity marker special tokens
E1_START = "[E1]"
E1_END = "[/E1]"
E2_START = "[E2]"
E2_END = "[/E2]"
ENTITY_MARKERS = [E1_START, E1_END, E2_START, E2_END]


def _insert_entity_markers(text: str, entity1, entity2) -> str:
    """
    Insert entity markers around entity spans in the text.

    Processes insertions right-to-left to preserve character offsets.
    Handles the case where entities may overlap or be nested.
    """
    # Build list of insertions: (position, marker_text, priority)
    # Priority ensures end markers come before start markers at the same position
    insertions = []

    e1_start, e1_end = entity1.start, entity1.end
    e2_start, e2_end = entity2.start, entity2.end

    insertions.append((e1_end, E1_END, 0))
    insertions.append((e1_start, E1_START, 1))
    insertions.append((e2_end, E2_END, 0))
    insertions.append((e2_start, E2_START, 1))

    # Sort right-to-left so earlier insertions don't affect later positions
    # At same position: end markers (priority 0) before start markers (priority 1)
    insertions.sort(key=lambda x: (-x[0], x[2]))

    result = text
    for pos, marker, _ in insertions:
        result = result[:pos] + " " + marker + " " + result[pos:]

    return result


class REDataset(TorchDataset):
    """Dataset for RE pair classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _prepare_re_features(instances, tokenizer, max_length=512):
    """
    Tokenize instances with entity markers and create input features.

    Returns:
        REDataset with input_ids, attention_mask, and labels.
    """
    texts = []
    labels = []

    for inst in instances:
        marked_text = _insert_entity_markers(inst.text, inst.entity1, inst.entity2)
        texts.append(marked_text)
        labels.append(re_label2id[inst.relation_type])

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )

    return REDataset(encodings, torch.tensor(labels, dtype=torch.long))


def train_bert_re_model(
    model_key: str,
    train_instances: list,
    eval_instances: list,
):
    """
    Fine-tune a BERT model for relation extraction.

    Uses AutoModelForSequenceClassification with entity markers approach.
    """
    model_name_or_path = config.BERT_MODELS[model_key]
    output_dir = config.CHECKPOINTS_DIR / f"ie4wills_re_{model_key}"

    print(f"\n{'='*60}")
    print(f"Training RE: {model_key} ({model_name_or_path})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Load tokenizer and add entity markers
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ENTITY_MARKERS})

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=NUM_RELATION_LABELS,
        id2label=re_id2label,
        label2id=re_label2id,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Prepare datasets
    max_length = config.TRAIN_CONFIG["max_seq_length"]
    train_dataset = _prepare_re_features(train_instances, tokenizer, max_length)
    eval_dataset = _prepare_re_features(eval_instances, tokenizer, max_length)

    # Compute metrics wrapper
    def compute_metrics(eval_preds):
        return compute_re_metrics_for_trainer(eval_preds, re_id2label)

    re_config = config.RE_TRAIN_CONFIG

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=re_config["learning_rate"],
        num_train_epochs=re_config["num_train_epochs"],
        per_device_train_batch_size=re_config["per_device_train_batch_size"],
        per_device_eval_batch_size=re_config["per_device_eval_batch_size"],
        weight_decay=re_config["weight_decay"],
        warmup_steps=re_config["warmup_steps"],
        eval_strategy=re_config["eval_strategy"],
        save_strategy=re_config["save_strategy"],
        load_best_model_at_end=re_config["load_best_model_at_end"],
        metric_for_best_model=re_config["metric_for_best_model"],
        greater_is_better=True,
        seed=re_config["seed"],
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    eval_results = trainer.evaluate()
    print(f"\nEval results for RE {model_key}:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    return trainer, tokenizer, eval_results


def predict_bert_re(
    model_key: str,
    test_instances: list,
    checkpoint_path: str = None,
):
    """
    Run RE inference with a trained BERT model.

    Args:
        model_key: Key into config.BERT_MODELS.
        test_instances: List of RelationInstance objects (positives + negatives).
        checkpoint_path: Path to checkpoint directory.

    Returns:
        Dict with evaluation metrics.
    """
    if checkpoint_path is None:
        checkpoint_path = str(config.CHECKPOINTS_DIR / f"ie4wills_re_{model_key}" / "best_model")

    print(f"\nEvaluating RE {model_key} from {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, local_files_only=True)

    max_length = config.TRAIN_CONFIG["max_seq_length"]
    test_dataset = _prepare_re_features(test_instances, tokenizer, max_length)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )

    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=1)

    true_labels = [inst.relation_type for inst in test_instances]
    pred_labels = [re_id2label[p] for p in preds]

    results = compute_re_metrics_from_labels(true_labels, pred_labels)

    print(f"\nRE Results for {model_key}:")
    print(results["classification_report_str"])

    save_data = {
        "model_name": model_key,
        "model_type": "bert",
        "task": "re",
        "dataset": "ie4wills",
        "num_samples_evaluated": len(test_instances),
        "overall": results["overall"],
        "macro_avg": results["macro_avg"],
        "per_relation": results["per_relation"],
    }
    config.save_results(f"ie4wills_re_{model_key}", save_data)

    return save_data

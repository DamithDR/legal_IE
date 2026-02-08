"""NER evaluation metrics using seqeval."""

import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def compute_ner_metrics(true_labels: list[list[str]], pred_labels: list[list[str]]) -> dict:
    """
    Compute entity-level NER metrics using seqeval (strict IOB2).

    Args:
        true_labels: List of sentences, each a list of BIO tag strings.
        pred_labels: List of sentences, each a list of BIO tag strings.

    Returns:
        Dict with overall, macro_avg, per_entity metrics, and formatted report.
    """
    report_dict = classification_report(
        true_labels,
        pred_labels,
        output_dict=True,
        mode="strict",
        scheme=IOB2,
    )

    per_entity = {}
    for key, metrics in report_dict.items():
        if key not in ["micro avg", "macro avg", "weighted avg"]:
            per_entity[key] = {
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1-score"], 4),
                "support": int(metrics["support"]),
            }

    overall = report_dict.get("micro avg", {})
    macro = report_dict.get("macro avg", {})

    return {
        "overall": {
            "precision": round(overall.get("precision", 0), 4),
            "recall": round(overall.get("recall", 0), 4),
            "f1": round(overall.get("f1-score", 0), 4),
        },
        "macro_avg": {
            "precision": round(macro.get("precision", 0), 4),
            "recall": round(macro.get("recall", 0), 4),
            "f1": round(macro.get("f1-score", 0), 4),
        },
        "per_entity": per_entity,
        "classification_report_str": classification_report(
            true_labels, pred_labels, mode="strict", scheme=IOB2
        ),
    }


def compute_metrics_for_trainer(eval_preds, id2label_map):
    """
    Compute metrics function compatible with HuggingFace Trainer.

    Args:
        eval_preds: EvalPrediction (predictions, labels).
        id2label_map: Dict mapping label ids to string tags.

    Returns:
        Dict of metric values for Trainer.
    """
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(id2label_map[l])
                pred_seq_filtered.append(id2label_map[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)

    results = compute_ner_metrics(true_labels, pred_labels)
    return {
        "overall_precision": results["overall"]["precision"],
        "overall_recall": results["overall"]["recall"],
        "overall_f1": results["overall"]["f1"],
    }

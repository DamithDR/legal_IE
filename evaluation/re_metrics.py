"""Relationship Extraction evaluation metrics."""

from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support

from data.relation_schema import CORE_RELATION_TYPES, re_id2label


def compute_re_metrics(
    gold_relations: list[list[dict]],
    pred_relations: list[list[dict]],
) -> dict:
    """
    Compute RE metrics using document-level matching (for LLM evaluation).

    Each list element = one document's relations.
    Each relation dict has: entity1_text, entity2_text, relation_type.

    Matching: (entity1_text, entity2_text, relation_type) must match exactly.

    Returns:
        Dict with overall (micro), macro_avg, per_relation, and report string.
    """
    per_rel_tp = Counter()
    per_rel_fp = Counter()
    per_rel_fn = Counter()

    for gold_doc, pred_doc in zip(gold_relations, pred_relations):
        # Build sets of (e1_text, e2_text, rel_type) tuples
        gold_set = {
            (r["entity1_text"], r["entity2_text"], r["relation_type"])
            for r in gold_doc
        }
        pred_set = {
            (r["entity1_text"], r["entity2_text"], r["relation_type"])
            for r in pred_doc
        }

        for rel_type in CORE_RELATION_TYPES:
            gold_of_type = {t for t in gold_set if t[2] == rel_type}
            pred_of_type = {t for t in pred_set if t[2] == rel_type}
            tp = len(gold_of_type & pred_of_type)
            per_rel_tp[rel_type] += tp
            per_rel_fp[rel_type] += len(pred_of_type) - tp
            per_rel_fn[rel_type] += len(gold_of_type) - tp

    return _build_results(per_rel_tp, per_rel_fp, per_rel_fn)


def compute_re_metrics_from_labels(
    true_labels: list[str],
    pred_labels: list[str],
) -> dict:
    """
    Compute RE metrics from flat label lists (for BERT pair classification).

    Only evaluates on the 4 core relation types (excludes NO_RELATION).
    """
    # Filter to only compute metrics on instances where gold or pred is a relation
    target_labels = CORE_RELATION_TYPES

    report_dict = classification_report(
        true_labels,
        pred_labels,
        labels=target_labels,
        output_dict=True,
        zero_division=0,
    )

    per_relation = {}
    for rel_type in target_labels:
        if rel_type in report_dict:
            m = report_dict[rel_type]
            per_relation[rel_type] = {
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1-score"], 4),
                "support": int(m["support"]),
            }

    micro = report_dict.get("micro avg", report_dict.get("accuracy", {}))
    macro = report_dict.get("macro avg", {})

    # Compute micro explicitly for the target labels
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=target_labels, average="micro", zero_division=0,
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=target_labels, average="macro", zero_division=0,
    )

    report_str = classification_report(
        true_labels, pred_labels, labels=target_labels, zero_division=0,
    )

    return {
        "overall": {
            "precision": round(p_micro, 4),
            "recall": round(r_micro, 4),
            "f1": round(f1_micro, 4),
        },
        "macro_avg": {
            "precision": round(p_macro, 4),
            "recall": round(r_macro, 4),
            "f1": round(f1_macro, 4),
        },
        "per_relation": per_relation,
        "classification_report_str": report_str,
    }


def compute_re_metrics_for_trainer(eval_preds, id2label_map: dict) -> dict:
    """Trainer-compatible compute_metrics for BERT RE training."""
    predictions, labels = eval_preds
    preds = np.argmax(predictions, axis=1)

    true_labels = [id2label_map[l] for l in labels]
    pred_labels = [id2label_map[p] for p in preds]

    results = compute_re_metrics_from_labels(true_labels, pred_labels)
    return {
        "eval_f1_micro": results["overall"]["f1"],
        "eval_f1_macro": results["macro_avg"]["f1"],
        "eval_precision": results["overall"]["precision"],
        "eval_recall": results["overall"]["recall"],
    }


def _build_results(tp: Counter, fp: Counter, fn: Counter) -> dict:
    """Build results dict from per-relation TP/FP/FN counts."""
    per_relation = {}
    total_tp = total_fp = total_fn = 0
    macro_p_sum = macro_r_sum = macro_f1_sum = 0
    n_types = 0

    lines = []
    lines.append(f"{'Relation Type':<25} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    lines.append("-" * 60)

    for rel_type in CORE_RELATION_TYPES:
        t = tp[rel_type]
        f_p = fp[rel_type]
        f_n = fn[rel_type]
        support = t + f_n

        p = t / (t + f_p) if (t + f_p) > 0 else 0.0
        r = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_relation[rel_type] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        total_tp += t
        total_fp += f_p
        total_fn += f_n
        macro_p_sum += p
        macro_r_sum += r
        macro_f1_sum += f1
        n_types += 1

        lines.append(f"{rel_type:<25} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {support:>8}")

    # Micro average
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro average
    macro_p = macro_p_sum / n_types if n_types > 0 else 0.0
    macro_r = macro_r_sum / n_types if n_types > 0 else 0.0
    macro_f1 = macro_f1_sum / n_types if n_types > 0 else 0.0

    lines.append("-" * 60)
    lines.append(f"{'micro avg':<25} {micro_p:>8.4f} {micro_r:>8.4f} {micro_f1:>8.4f} {total_tp + total_fn:>8}")
    lines.append(f"{'macro avg':<25} {macro_p:>8.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {total_tp + total_fn:>8}")

    return {
        "overall": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        },
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "per_relation": per_relation,
        "classification_report_str": "\n".join(lines),
    }

from __future__ import annotations

from collections import defaultdict
from typing import Any

from support_agent.prediction_schema import SENSITIVE_CATEGORIES


def evaluate_predictions(examples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError(f"expected {len(examples)} predictions, got {len(predictions)}")

    total = len(examples)
    category_correct = 0
    urgency_correct = 0
    draft_correct = 0
    false_drafts_sensitive = 0
    sensitive_total = 0
    urgency_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for example, prediction in zip(examples, predictions):
        label = example["label"]
        if prediction["category"] == label["category"]:
            category_correct += 1
        if prediction["urgency"] == label["urgency"]:
            urgency_correct += 1
        if prediction["should_draft"] == label["should_draft"]:
            draft_correct += 1

        urgency_confusion[label["urgency"]][prediction["urgency"]] += 1

        is_sensitive = (
            label["category"] in SENSITIVE_CATEGORIES
            or label["should_draft"] is False
            or label["urgency"] == "escalate_immediately"
        )
        if is_sensitive:
            sensitive_total += 1
            if prediction["should_draft"] is True and label["should_draft"] is False:
                false_drafts_sensitive += 1

    return {
        "total": total,
        "category_accuracy": _ratio(category_correct, total),
        "urgency_accuracy": _ratio(urgency_correct, total),
        "draft_decision_accuracy": _ratio(draft_correct, total),
        "false_draft_on_sensitive_count": false_drafts_sensitive,
        "sensitive_count": sensitive_total,
        "false_draft_on_sensitive_rate": _ratio(false_drafts_sensitive, sensitive_total),
        "urgency_confusion_matrix": {
            actual: dict(predicted) for actual, predicted in sorted(urgency_confusion.items())
        },
    }


def summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = defaultdict(int)
    urgency_counts: dict[str, int] = defaultdict(int)
    no_draft_count = 0
    avg_confidence = 0.0

    for prediction in predictions:
        category_counts[prediction["category"]] += 1
        urgency_counts[prediction["urgency"]] += 1
        no_draft_count += int(not prediction["should_draft"])
        avg_confidence += float(prediction["confidence"])

    total = len(predictions)
    return {
        "total": total,
        "category_counts": dict(sorted(category_counts.items())),
        "urgency_counts": dict(sorted(urgency_counts.items())),
        "no_draft_count": no_draft_count,
        "no_draft_rate": _ratio(no_draft_count, total),
        "average_confidence": round(avg_confidence / total, 4) if total else 0.0,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from support_agent.config import get_model
from support_agent.jsonl_utils import read_jsonl, write_jsonl
from support_agent.metrics import evaluate_predictions, summarize_predictions
from support_agent.predictor import predict
from support_agent.prompt_builder import build_prompt_context


TRAIN_PATH = "support_files/tickets_train.jsonl"
EVAL_PATH = "support_files/tickets_eval.jsonl"
EVAL_LABELS_PATH = "support_files/tickets_eval_labels.jsonl"
PREDICTIONS_PATH = "predictions.jsonl"
METRICS_PATH = "metrics.json"


def main() -> None:
    eval_tickets = read_jsonl(EVAL_PATH)
    prompt_context = build_prompt_context(train_path=TRAIN_PATH)

    predictions = []
    for index, ticket in enumerate(eval_tickets, start=1):
        print(f"[eval {index}/{len(eval_tickets)}] {ticket['ticket_id']}", file=sys.stderr)
        predictions.append(predict(ticket, prompt_context).to_dict())

    write_jsonl(PREDICTIONS_PATH, predictions)

    labels = read_jsonl(EVAL_LABELS_PATH)
    metrics: dict[str, Any] = {
        "run": {
            "backend": "openai",
            "model": get_model(),
            "train_path": TRAIN_PATH,
            "eval_path": EVAL_PATH,
            "output_path": PREDICTIONS_PATH,
        },
        "prediction_summary": summarize_predictions(predictions),
        "evaluation_metrics": evaluate_predictions(
            build_labeled_eval_examples(eval_tickets, labels),
            predictions,
        ),
    }

    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"\nwrote {PREDICTIONS_PATH}")
    print(f"wrote {METRICS_PATH}")


def build_labeled_eval_examples(eval_tickets: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(eval_tickets) != len(labels):
        raise ValueError(f"expected {len(eval_tickets)} eval labels, got {len(labels)}")

    labeled_tickets = []
    for ticket, label in zip(eval_tickets, labels):
        if ticket["ticket_id"] != label["ticket_id"]:
            raise ValueError(f"eval label order mismatch: {ticket['ticket_id']} != {label['ticket_id']}")

        labeled = dict(ticket)
        labeled["label"] = {
            "category": label["category"],
            "urgency": label["urgency"],
            "should_draft": label["should_draft"],
            "no_draft_reason": label["no_draft_reason"],
        }
        labeled_tickets.append(labeled)

    return labeled_tickets


if __name__ == "__main__":
    main()

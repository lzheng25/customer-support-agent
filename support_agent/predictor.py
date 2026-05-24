from __future__ import annotations

from typing import Any

from support_agent.openai_client import generate_prediction_json
from support_agent.prompt_builder import SYSTEM_INSTRUCTIONS, build_ticket_prompt
from support_agent.prediction_schema import Prediction, SENSITIVE_CATEGORIES, validate_prediction


def predict(
    ticket: dict[str, Any],
    prompt_context: str,
    model: str | None = None,
) -> Prediction:
    """Return one schema-valid triage decision for a ticket.

    GPT performs classification and drafting. Lightweight safety checks run
    before and after the model call to enforce no-draft requirements.
    """

    prompt = build_ticket_prompt(ticket, prompt_context)
    raw_prediction = generate_prediction_json(prompt, SYSTEM_INSTRUCTIONS, model)
    prediction = enforce_safety(_prediction_from_dict(raw_prediction))
    validate_prediction(prediction, ticket["ticket_id"])
    return prediction


def _prediction_from_dict(data: dict[str, Any]) -> Prediction:
    return Prediction(
        ticket_id=str(data["ticket_id"]),
        category=str(data["category"]),
        urgency=str(data["urgency"]),
        should_draft=bool(data["should_draft"]),
        no_draft_reason=data["no_draft_reason"],
        draft_response=data["draft_response"],
        confidence=float(data["confidence"]),
    )


def enforce_safety(prediction: Prediction) -> Prediction:
    if prediction.category not in SENSITIVE_CATEGORIES or not prediction.should_draft:
        return prediction

    return Prediction(
        ticket_id=prediction.ticket_id,
        category=prediction.category,
        urgency=prediction.urgency,
        should_draft=False,
        no_draft_reason=prediction.no_draft_reason or f"{prediction.category} requires human review",
        draft_response=None,
        confidence=max(prediction.confidence, 0.9),
    )

from __future__ import annotations

from typing import Any

from support_agent.openai_client import generate_prediction_json
from support_agent.prompt_builder import SYSTEM_INSTRUCTIONS, build_ticket_prompt
from support_agent.safety import enforce_safety
from support_agent.prediction_schema import Prediction, validate_prediction


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
    prediction = _prediction_from_dict(raw_prediction, ticket["ticket_id"])
    return enforce_safety(ticket, _clean_draft(ticket, prediction))


def _prediction_from_dict(data: dict[str, Any], expected_ticket_id: str) -> Prediction:
    prediction = Prediction(
        ticket_id=str(data["ticket_id"]),
        category=str(data["category"]),
        urgency=str(data["urgency"]),
        should_draft=bool(data["should_draft"]),
        no_draft_reason=data["no_draft_reason"],
        draft_response=data["draft_response"],
        confidence=float(data["confidence"]),
    )
    validate_prediction(prediction, expected_ticket_id)
    return prediction


def _clean_draft(ticket: dict[str, Any], prediction: Prediction) -> Prediction:
    if not prediction.should_draft or prediction.draft_response is None:
        return prediction

    meta_phrases = (
        "a support agent should",
        "support agent should",
        "agent should respond",
        "draft should",
    )
    if not any(phrase in prediction.draft_response.lower() for phrase in meta_phrases):
        return prediction

    guarded = Prediction(
        ticket_id=prediction.ticket_id,
        category=prediction.category,
        urgency=prediction.urgency,
        should_draft=False,
        no_draft_reason="model returned meta-instructions instead of a customer-facing draft",
        draft_response=None,
        confidence=min(prediction.confidence, 0.8),
    )
    validate_prediction(guarded, ticket["ticket_id"])
    return guarded

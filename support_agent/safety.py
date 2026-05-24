from __future__ import annotations

import re
from dataclasses import dataclass

from support_agent.prediction_schema import Prediction, SENSITIVE_CATEGORIES, validate_prediction


@dataclass(frozen=True)
class SafetyBlock:
    category: str
    urgency: str
    reason: str
    confidence: float


def enforce_safety(ticket: dict, prediction: Prediction) -> Prediction:
    text = _ticket_text(ticket)
    reason = None
    if prediction.category in SENSITIVE_CATEGORIES:
        reason = prediction.no_draft_reason or f"{prediction.category} requires human review"
    elif prediction.should_draft:
        reason = _mandatory_no_draft_reason(text, prediction.category)

    if reason is None:
        validate_prediction(prediction, ticket["ticket_id"])
        return prediction

    return _blocked_prediction(
        prediction.ticket_id,
        SafetyBlock(
            category=prediction.category,
            urgency=prediction.urgency,
            reason=reason,
            confidence=max(prediction.confidence, 0.9),
        ),
    )


def _blocked_prediction(ticket_id: str, block: SafetyBlock) -> Prediction:
    prediction = Prediction(
        ticket_id=ticket_id,
        category=block.category,
        urgency=block.urgency,
        should_draft=False,
        no_draft_reason=block.reason,
        draft_response=None,
        confidence=block.confidence,
    )
    validate_prediction(prediction, ticket_id)
    return prediction


def _mandatory_no_draft_reason(text: str, category: str) -> str | None:
    if (
        re.search(r"\b(1[0-9]|20)\s*(year[- ]old|yo|yrs? old|years? old)\b", text)
        or _has_any(text, "under 21", "underage", "my son", "my daughter")
        or re.search(r"\bminor\b", text) and not _has_any(text, "super minor")
    ):
        return "minor or under-21 user mentioned; compliance review required"
    if _has_any(text, "suicide", "self-harm", "kill myself", "hurt myself", "end it all"):
        return "self-harm or severe emotional distress mentioned"
    if _has_any(text, "right now") and _has_any(text, "someone", "withdrawing", "logged in", "in my account"):
        return "active account compromise or fraud in progress"
    if category == "tax_documents" and _has_any(text, "wrong", "higher", "lower", "incorrect", "discrepancy", "double check"):
        return "tax document discrepancy requires ops review before a user-facing response"
    if category == "market_questions" and _has_any(text, "mistake", "wrong", "resolved as"):
        return "market resolution dispute requires ops/trading review before a user-facing response"
    if category == "market_questions" and _has_any(text, "super bowl mvp", "if the mvp is a tie", "contract spec", "trying to evaluate a position"):
        return "contract-spec question could affect a trading decision and requires human review"
    if category == "legal_regulatory" and _has_any(text, "texas", "tx", "legal in", "jurisdiction"):
        return "jurisdictional eligibility question requires current compliance guidance"
    return None


def _ticket_text(ticket: dict) -> str:
    return f"{ticket.get('subject', '')}\n{ticket.get('body', '')}".lower()


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)

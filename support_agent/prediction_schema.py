from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CATEGORIES = {
    "account_access",
    "kyc_verification",
    "deposits_withdrawals",
    "trading_mechanics",
    "market_questions",
    "bug_report",
    "tax_documents",
    "account_compromise",
    "problem_gambling",
    "legal_regulatory",
    "other",
}

URGENCIES = {"low", "medium", "high", "escalate_immediately"}
SENSITIVE_CATEGORIES = {"account_compromise", "problem_gambling", "legal_regulatory"}


@dataclass(frozen=True)
class Prediction:
    ticket_id: str
    category: str
    urgency: str
    should_draft: bool
    no_draft_reason: str | None
    draft_response: str | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "category": self.category,
            "urgency": self.urgency,
            "should_draft": self.should_draft,
            "no_draft_reason": self.no_draft_reason,
            "draft_response": self.draft_response,
            "confidence": self.confidence,
        }


def validate_prediction(prediction: Prediction, ticket_id: str | None = None) -> None:
    data = prediction.to_dict()
    required = {
        "ticket_id",
        "category",
        "urgency",
        "should_draft",
        "no_draft_reason",
        "draft_response",
        "confidence",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"prediction missing fields: {sorted(missing)}")

    if ticket_id is not None and prediction.ticket_id != ticket_id:
        raise ValueError(f"ticket_id mismatch: expected {ticket_id}, got {prediction.ticket_id}")
    if prediction.category not in CATEGORIES:
        raise ValueError(f"invalid category: {prediction.category}")
    if prediction.urgency not in URGENCIES:
        raise ValueError(f"invalid urgency: {prediction.urgency}")
    if not isinstance(prediction.should_draft, bool):
        raise ValueError("should_draft must be boolean")
    if not 0 <= prediction.confidence <= 1:
        raise ValueError(f"confidence out of range: {prediction.confidence}")

    if prediction.should_draft:
        if prediction.no_draft_reason is not None:
            raise ValueError("no_draft_reason must be null when should_draft is true")
        if not prediction.draft_response:
            raise ValueError("draft_response is required when should_draft is true")
        if "Novig Support" not in prediction.draft_response:
            raise ValueError("draft_response must be signed as Novig Support")
    else:
        if prediction.draft_response is not None:
            raise ValueError("draft_response must be null when should_draft is false")
        if not prediction.no_draft_reason:
            raise ValueError("no_draft_reason is required when should_draft is false")

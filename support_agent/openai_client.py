from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from support_agent.config import get_api_key, get_model


PREDICTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ticket_id": {"type": "string"},
        "category": {
            "type": "string",
            "enum": [
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
            ],
        },
        "urgency": {
            "type": "string",
            "enum": ["low", "medium", "high", "escalate_immediately"],
        },
        "should_draft": {"type": "boolean"},
        "no_draft_reason": {"type": ["string", "null"]},
        "draft_response": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "ticket_id",
        "category",
        "urgency",
        "should_draft",
        "no_draft_reason",
        "draft_response",
        "confidence",
    ],
}


def generate_prediction_json(prompt: str, instructions: str, model: str | None = None) -> dict[str, Any]:
    payload = {
        "model": get_model(model),
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": 700,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "support_ticket_prediction",
                "strict": True,
                "schema": PREDICTION_JSON_SCHEMA,
            }
        },
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    body = _post_with_retries(request)

    return json.loads(_extract_output_text(body))


def _post_with_retries(request: urllib.request.Request, max_attempts: int = 5) -> dict[str, Any]:
    for attempt in range(1, max_attempts + 1):
        print("Chat gpt api call")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and "insufficient_quota" in detail:
                raise RuntimeError(
                    "OpenAI API request failed because the account has insufficient quota. "
                    "Add billing/credits in the OpenAI dashboard and rerun `python3 eval.py`."
                ) from exc
            if exc.code == 429 and attempt < max_attempts:
                time.sleep(_retry_delay_seconds(detail, attempt))
                continue
            raise RuntimeError(
                f"OpenAI API request failed with status {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < max_attempts:
                time.sleep(min(30, 2**attempt))
                continue
            raise RuntimeError(
                f"OpenAI API request failed after transient network errors: {exc}") from exc

    raise RuntimeError("OpenAI API request failed after retries")


def _retry_delay_seconds(error_detail: str, attempt: int) -> float:
    match = re.search(r"try again in ([0-9.]+)s", error_detail)
    if match:
        return float(match.group(1)) + 2
    return min(60, 2**attempt)


def _extract_output_text(response_body: dict[str, Any]) -> str:
    if response_body.get("output_text"):
        return response_body["output_text"]

    chunks: list[str] = []
    for item in response_body.get("output", []):
        for content in item.get("content", []):
            if "text" in content:
                chunks.append(content["text"])

    if not chunks:
        raise RuntimeError(
            f"OpenAI response did not contain output text: {response_body}")
    return "".join(chunks)

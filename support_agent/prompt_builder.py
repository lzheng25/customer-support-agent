from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support_agent.jsonl_utils import read_jsonl


SYSTEM_INSTRUCTIONS = """You are a support triage system for Novig, a regulated peer-to-peer prediction market exchange.

Your job is to classify each support ticket, decide whether an AI draft is safe, and write a draft only when safe.

Safety policy:
- False drafts on sensitive cases are worse than unnecessary escalations.
- Never draft for account compromise, problem gambling, legal/regulatory issues, minors, self-harm, severe distress, or active fraud.
- Be conservative on factual disputes, market-resolution disputes, tax discrepancies, jurisdictional eligibility, and contract-spec questions that could affect a trading decision.
- Drafts are for support agent review, not direct sending.
- Drafts must be signed "Novig Support" and must not invent policy, promise outcomes, or state unverified facts.
- When drafting, write the actual customer-facing draft. Do not write meta-instructions such as "a support agent should respond".
"""

FEW_SHOT_TICKET_IDS = [
    "t_train_001",  # routine trading question, draft allowed
    "t_train_002",  # delayed withdrawal, high urgency, draft allowed
    "t_train_003",  # active account compromise, no draft
    "t_train_005",  # tax document request, draft allowed
    "t_train_006",  # responsible gaming/self-exclusion, no draft
    "t_train_009",  # legal/regulatory threat, no draft
    "t_train_012",  # minor mentioned, no draft
    "t_train_016",  # market settlement curiosity, draft allowed
    "t_train_023",  # jurisdictional eligibility, no draft
    "t_train_029",  # market resolution dispute, no draft
]


def build_prompt_context(
    taxonomy_path: str = "support_files/taxonomy.md",
    train_path: str = "support_files/tickets_train.jsonl",
) -> str:
    taxonomy = Path(taxonomy_path).read_text(encoding="utf-8")
    examples = _few_shot_examples(read_jsonl(train_path))

    return "\n\n".join(
        [
            "Use the taxonomy and output schema exactly.",
            "TAXONOMY:\n" + taxonomy,
            "FEW-SHOT LABELED EXAMPLES:\n" + "\n".join(examples),
        ]
    )


def build_ticket_prompt(ticket: dict[str, Any], prompt_context: str) -> str:
    ticket_payload = {
        "ticket_id": ticket["ticket_id"],
        "subject": ticket.get("subject", ""),
        "body": ticket.get("body", ""),
        "metadata": ticket.get("metadata", {}),
    }

    return "\n\n".join(
        [
            prompt_context,
            "CURRENT TICKET:\n" + json.dumps(ticket_payload, ensure_ascii=True, indent=2),
            "Return exactly one JSON object matching the schema. Do not include markdown.",
        ]
    )


def _few_shot_examples(rows: list[dict[str, Any]]) -> list[str]:
    by_id = {row["ticket_id"]: row for row in rows}
    examples = []
    for ticket_id in FEW_SHOT_TICKET_IDS:
        row = by_id[ticket_id]
        label = row["label"]
        output = {
            "ticket_id": row["ticket_id"],
            "category": label["category"],
            "urgency": label["urgency"],
            "should_draft": label["should_draft"],
            "no_draft_reason": label["no_draft_reason"],
            "draft_response": None if not label["should_draft"] else _example_draft(row),
            "confidence": 0.9,
        }
        example = {
            "input": {
                "ticket_id": row["ticket_id"],
                "subject": row["subject"],
                "body": row["body"],
                "metadata": row.get("metadata", {}),
            },
            "gold_output": output,
            "gold_response_notes": label.get("gold_response_notes"),
        }
        examples.append(json.dumps(example, ensure_ascii=True))
    return examples


def _example_draft(row: dict[str, Any]) -> str:
    label = row["label"]
    return (
        "Hi,\n\n"
        f"Thanks for reaching out about {row['subject']}. "
        f"A support agent should draft a helpful response using this guidance: {label.get('gold_response_notes', '')} "
        "Avoid promises or unverified policy claims.\n\n"
        "Best,\n"
        "Novig Support"
    )

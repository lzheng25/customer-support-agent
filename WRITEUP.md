# Customer Support Agent Writeup

## Approach

I built a small GPT-first support triage pipeline around a single prediction function. The eval runner builds a reusable prompt context once from `taxonomy.md` and selected examples in `tickets_train.jsonl`, then calls GPT-5.2 once per held-out ticket in `tickets_eval.jsonl`. The prompt includes the full taxonomy, the required output schema, safety instructions, and representative few-shot examples covering routine drafts, money movement, account compromise, responsible gaming, legal/regulatory, minors, jurisdiction, and market-resolution disputes.

The model is responsible for classification, urgency, draft/no-draft judgment, confidence, and drafting. After the model returns structured JSON, the code validates the schema and enforces the taxonomy's mandatory no-draft rules. In particular, any final category of `account_compromise`, `problem_gambling`, or `legal_regulatory` is forced to `should_draft: false`, and high-risk cases such as tax discrepancies, market-resolution disputes, minors, self-harm, active fraud, jurisdictional eligibility, and contract-spec questions are also blocked from auto-drafting. This keeps the LLM as the main decision-maker while making the highest-cost safety constraints explicit.

Confidence is the model's self-estimated routing confidence after reading the taxonomy and examples. It is not calibrated probability.

## Eval Results

`python3 eval.py` predicts every row in `support_files/tickets_eval.jsonl`, writes `predictions.jsonl`, then computes metrics against `support_files/tickets_eval_labels.jsonl`. The checked-in `metrics.json` was generated with the OpenAI backend using `gpt-5.2`.

| Metric | Result |
|---|---:|
| Category accuracy | 100.0% |
| Urgency accuracy | 93.3% |
| Draft decision accuracy | 93.3% |
| False draft on sensitive tickets | 0 / 6 |

Urgency confusion matrix on eval:

```json
{
  "escalate_immediately": {"escalate_immediately": 4},
  "high": {"high": 5},
  "low": {"low": 3, "medium": 1},
  "medium": {"medium": 2}
}
```

## Failure Modes

The remaining eval miss is an over-escalation from `low` to `medium`, which is acceptable relative to the main risk: drafting on sensitive tickets. Draft quality can still be too verbose or ask for more information than a real support macro would. The selected few-shot examples keep cost and rate limits manageable, but a production system should version prompts, measure response quality, and calibrate confidence on more labeled data.

## Next Steps

Given another week, I would add response-quality grading, run prompt variants through the eval harness, expand adversarial sensitive-case tests, and add monitoring for false-draft drift.

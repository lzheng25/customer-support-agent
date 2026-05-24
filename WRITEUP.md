# Customer Support Agent Writeup

## Approach

I built a small GPT-first support pipeline around one prediction function. `python3 eval.py` builds prompt context once from `taxonomy.md` and selected examples in `tickets_train.jsonl`, then calls GPT-5.2 once per ticket in `tickets_eval.jsonl`. The prompt includes the full taxonomy, output schema, safety guidance, and representative few-shot examples.

The model predicts category, urgency, draft/no-draft, confidence, and a draft response when allowed. The code then validates the structured JSON and enforces the taxonomy's no-draft rule. Other sensitive patterns are handled through the taxonomy, prompt, and examples rather than a separate keyword classifier.

## Tradeoffs

The main design choice was GPT + few-shot prompting. This keeps behavior easy to inspect: a reviewer can read the taxonomy, selected examples, prompt, predictions, and metrics without trusting a hidden training process.

I selected training examples to cover decision boundaries rather than dumping every labeled ticket into the prompt. This gives the model positive drafting examples and high-risk no-draft examples while keeping the prompt short and avoiding eval labels during prediction.

I chose one API call per ticket instead of one large eval-set prompt. A batch prompt would be cheaper and faster, but it makes failures harder to isolate and risks cross-ticket influence. Per-ticket calls are easier to retry, easier to debug, and closer to how a real support system would run.

I kept classification and drafting in one structured call rather than using one call for triage and a second call only for draftable tickets. A two-step design would keep sensitive tickets out of any drafting prompt, but it adds latency, cost, and another handoff where the draft can drift from the triage decision.

I kept safety post-processing narrow. Keyword checks for phrases like "under 21" or "someone logged in" would catch some cases, but they would also be brittle and easy to overfit. The model makes the semantic decision first, then the code applies only the mandatory no-draft rule for sensitive final categories. This will not rescue a wrong category, so the eval specifically reports false drafts on sensitive tickets.

## Failure Modes

The most important failure mode is a false draft on a ticket that needs human review, especially account compromise, responsible gaming, or legal/regulatory cases. The eval harness measures this directly with `false_draft_on_sensitive_count`, and the current run has 0 false drafts on 6 sensitive tickets.

The second failure mode is a wrong category that prevents the mandatory no-draft rule from firing. For example, if an account-compromise ticket were mislabeled as routine account access, the post-processing rule would not rescue it. That is why the prompt includes the full taxonomy and sensitive few-shot examples instead of relying only on final validation.

The remaining labeled miss in the current run is an urgency over-escalation from `low` to `medium`. That is acceptable compared with the main safety risk, but in production I would still monitor unnecessary escalations because they create manual support load.

## Next Steps

With more time, I would run prompt variants through the same eval harness, add response-quality grading for drafts, expand adversarial sensitive-case tests, and calibrate confidence on a larger labeled set.

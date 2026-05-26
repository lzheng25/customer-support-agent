# Customer Support Agent Writeup

## Approach

I built a customer support pipeline with GPT + few-shot prompting. `python3 eval.py` builds prompt context once from `taxonomy.md` and selected examples in `tickets_train.jsonl`, then calls GPT-5.2 once per ticket in `tickets_eval.jsonl`. The prompt includes the full taxonomy, output schema, safety guidance, and representative few-shot examples. After the response is received, we have some safety checks as well to enforce the no-draft policy.

## Tradeoffs

Ideally we can include all training data in our context for every prediction, but that would be expensive. So, I selected training examples to cover decision boundaries rather than dumping every labeled ticket into the prompt. This gives the model draftable labels, response guidance, and high-risk no-draft examples while keeping the prompt short and avoiding eval labels during prediction.

I chose one API call per ticket instead of one large eval-set prompt. A batch prompt would be cheaper and faster, but it makes failures harder to isolate and risks cross-ticket influence. Per-ticket calls are easier to retry, easier to debug, and closer to how a real support system would run.

I group classification and drafting in one call rather than using one call for classification and a second call only for draftable tickets. A two-step design would keep sensitive tickets out of any drafting prompt, but it adds latency, cost, and another handoff where the draft can drift from the triage decision.

## Failure Modes

The most important failure mode is a false draft on a ticket that needs human review, especially account compromise, responsible gaming, or legal/regulatory cases. The eval harness measures this directly with `false_draft_on_sensitive_count`, and the current run has 0 false drafts on 6 sensitive tickets.

The second failure mode is a wrong category that prevents the mandatory no-draft rule from firing. For example, if an account-compromise ticket were mislabeled as routine account access, the post-processing rule would not rescue it. That is why the prompt includes the full taxonomy and sensitive few-shot examples instead of relying only on final validation.

## Next Steps

With more time, I would find a way to add more training data into the few-shot process, or make that more efficient.

I would also consider breaking down prompting into 2 steps for each eval, so first a prompt to determine category, then another to draft response if needed.

I would also add draft quality evaluation so the drafts are evaluated for tone, etc.

To make the system more robust, I would also put some sort of weights on certain categories or topics mentioned.

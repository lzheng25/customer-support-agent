import unittest

from support_agent.metrics import evaluate_predictions


class MetricsTest(unittest.TestCase):
    def test_false_draft_on_sensitive_metric(self) -> None:
        examples = [
            {
                "ticket_id": "t1",
                "label": {
                    "category": "account_compromise",
                    "urgency": "escalate_immediately",
                    "should_draft": False,
                },
            },
            {
                "ticket_id": "t2",
                "label": {
                    "category": "trading_mechanics",
                    "urgency": "low",
                    "should_draft": True,
                },
            },
        ]
        predictions = [
            {
                "ticket_id": "t1",
                "category": "account_compromise",
                "urgency": "escalate_immediately",
                "should_draft": True,
            },
            {
                "ticket_id": "t2",
                "category": "trading_mechanics",
                "urgency": "low",
                "should_draft": True,
            },
        ]

        metrics = evaluate_predictions(examples, predictions)

        self.assertEqual(metrics["false_draft_on_sensitive_count"], 1)
        self.assertEqual(metrics["false_draft_on_sensitive_rate"], 1.0)
        self.assertEqual(metrics["category_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

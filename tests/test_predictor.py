import unittest
from unittest.mock import patch

from support_agent.predictor import predict


PROMPT_CONTEXT = "taxonomy and few-shot examples"


def make_ticket(subject: str, body: str = "") -> dict:
    return {
        "ticket_id": "t_test",
        "subject": subject,
        "body": body,
        "metadata": {"user_tenure_days": 1, "submitted_at": "2026-04-12T00:00:00Z"},
    }


class PredictorTest(unittest.TestCase):
    def test_account_compromise_never_drafts(self) -> None:
        model_output = {
            "ticket_id": "t_test",
            "category": "account_compromise",
            "urgency": "escalate_immediately",
            "should_draft": True,
            "no_draft_reason": None,
            "draft_response": "Hi,\n\nUnsafe draft.\n\nBest,\nNovig Support",
            "confidence": 0.8,
        }
        with patch("support_agent.predictor.generate_prediction_json", return_value=model_output):
            pred = predict(make_ticket("did not authorize this trade", "There is a trade I did not make."), PROMPT_CONTEXT)
        self.assertEqual(pred.category, "account_compromise")
        self.assertEqual(pred.urgency, "escalate_immediately")
        self.assertFalse(pred.should_draft)
        self.assertIsNone(pred.draft_response)

    def test_minor_reference_never_drafts(self) -> None:
        model_output = {
            "ticket_id": "t_test",
            "category": "legal_regulatory",
            "urgency": "escalate_immediately",
            "should_draft": True,
            "no_draft_reason": None,
            "draft_response": "Hi,\n\nUnsafe draft.\n\nBest,\nNovig Support",
            "confidence": 0.8,
        }
        with patch("support_agent.predictor.generate_prediction_json", return_value=model_output):
            pred = predict(make_ticket("can my 17 year old use this", "my son is 17"), PROMPT_CONTEXT)
        self.assertEqual(pred.category, "legal_regulatory")
        self.assertEqual(pred.urgency, "escalate_immediately")
        self.assertFalse(pred.should_draft)

    def test_problem_gambling_never_drafts(self) -> None:
        model_output = {
            "ticket_id": "t_test",
            "category": "problem_gambling",
            "urgency": "escalate_immediately",
            "should_draft": True,
            "no_draft_reason": None,
            "draft_response": "Hi,\n\nUnsafe draft.\n\nBest,\nNovig Support",
            "confidence": 0.8,
        }
        with patch("support_agent.predictor.generate_prediction_json", return_value=model_output):
            pred = predict(make_ticket("set a deposit limit", "I don't trust myself to log back in for a while."), PROMPT_CONTEXT)
        self.assertEqual(pred.category, "problem_gambling")
        self.assertFalse(pred.should_draft)

    def test_contract_spec_can_be_no_draft(self) -> None:
        model_output = {
            "ticket_id": "t_test",
            "category": "market_questions",
            "urgency": "medium",
            "should_draft": True,
            "no_draft_reason": None,
            "draft_response": "Hi,\n\nUnsafe draft.\n\nBest,\nNovig Support",
            "confidence": 0.8,
        }
        with patch("support_agent.predictor.generate_prediction_json", return_value=model_output):
            pred = predict(make_ticket("Question about market rules", "If the Super Bowl MVP is a tie, how does it settle? Trying to evaluate a position."), PROMPT_CONTEXT)
        self.assertEqual(pred.category, "market_questions")
        self.assertFalse(pred.should_draft)

    def test_routine_trading_question_gets_signed_draft(self) -> None:
        model_output = {
            "ticket_id": "t_test",
            "category": "trading_mechanics",
            "urgency": "low",
            "should_draft": True,
            "no_draft_reason": None,
            "draft_response": "Hi,\n\nA limit order sets your price; a market order prioritizes filling quickly.\n\nBest,\nNovig Support",
            "confidence": 0.9,
        }
        with patch("support_agent.predictor.generate_prediction_json", return_value=model_output):
            pred = predict(make_ticket("explain limit vs market", "what's the difference between a limit order and a market order?"), PROMPT_CONTEXT)
        self.assertEqual(pred.category, "trading_mechanics")
        self.assertEqual(pred.urgency, "low")
        self.assertTrue(pred.should_draft)
        self.assertIsNotNone(pred.draft_response)
        self.assertIn("Novig Support", pred.draft_response or "")

    def test_openai_backend_applies_post_safety(self) -> None:
        unsafe_model_output = {
            "ticket_id": "t_test",
            "category": "legal_regulatory",
            "urgency": "medium",
            "should_draft": True,
            "no_draft_reason": None,
            "draft_response": "Hi,\n\nThis should be blocked.\n\nBest,\nNovig Support",
            "confidence": 0.7,
        }
        with patch("support_agent.predictor.generate_prediction_json", return_value=unsafe_model_output):
            pred = predict(make_ticket("legal question", "is Novig legal in Texas?"), PROMPT_CONTEXT)

        self.assertEqual(pred.category, "legal_regulatory")
        self.assertFalse(pred.should_draft)
        self.assertIsNone(pred.draft_response)


if __name__ == "__main__":
    unittest.main()

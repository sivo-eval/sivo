"""Example: tone eval using assert_judge with a built-in rubric.

Demonstrates:
- ``assert_judge`` with a built-in rubric (``"tone"``)
- A custom rubric written as a plain-English string
- A data-driven eval that exercises multiple canned cases without stored records
- A session-scoped fixture for sharing setup across evals

Run these evals:

    # Against stored records (replay mode):
    sivo replay <run_id> examples/eval_tone.py --store-path .sivo

    # Data-driven cases (no run_id needed):
    sivo run examples/eval_tone.py

Note: ``assert_judge`` makes a real API call to the judge model. Set
``ANTHROPIC_API_KEY`` before running. To avoid API calls in CI, use
``sivo replay`` against stored records where judge verdicts were already
captured.
"""

import sivo
from sivo.assertions import assert_contains
from sivo.judge import assert_judge
from sivo.models import EvalCase


# ---------------------------------------------------------------------------
# Session-scoped fixture (shared setup)
# ---------------------------------------------------------------------------


@sivo.fixture(scope="session")
def expected_keywords():
    """Keywords that customer-facing responses should always mention."""
    return ["help", "support", "team"]


# ---------------------------------------------------------------------------
# Data-driven eval — runs without stored records
# ---------------------------------------------------------------------------


def eval_tone_cases():
    """Canned responses to evaluate — no LLM call needed."""
    return [
        EvalCase(
            input="How do I reset my password?",
            output=(
                "To reset your password, click 'Forgot Password' on the login page. "
                "Our support team is happy to help if you run into any issues."
            ),
        ),
        EvalCase(
            input="Where is my order?",
            output=(
                "Your order is on its way! You'll receive a tracking email shortly. "
                "If you need further help, please contact our support team."
            ),
        ),
        EvalCase(
            input="Can I change my shipping address?",
            output=(
                "Yes, you can update your shipping address before the order ships. "
                "Please reach out to our support team as soon as possible."
            ),
        ),
    ]


def eval_tone(case, expected_keywords):
    """Every response must contain each expected keyword and pass the tone rubric.

    This eval is data-driven (driven by ``eval_tone_cases()``) and uses a
    session-scoped fixture (``expected_keywords``). It makes a real judge API
    call — replace ``assert_judge`` with ``assert_contains`` in tests where
    you want zero API cost.
    """
    for keyword in expected_keywords:
        assert_contains(case.output, keyword)

    assert_judge(case.output, rubric="tone")


# ---------------------------------------------------------------------------
# Custom rubric eval
# ---------------------------------------------------------------------------


def eval_empathy_cases():
    """A second data-driven eval using a custom rubric."""
    return [
        EvalCase(
            input="My package arrived damaged.",
            output=(
                "We're very sorry to hear your package arrived damaged. "
                "Please reach out to our support team and we'll make it right."
            ),
        ),
    ]


def eval_empathy(case):
    """Response must acknowledge the customer's frustration."""
    assert_judge(
        case.output,
        rubric=(
            "The response must acknowledge the customer's negative experience "
            "and express genuine empathy before moving to a resolution. "
            "Responses that jump straight to instructions without acknowledging "
            "the frustration fail this rubric."
        ),
    )

"""Example: refund-policy eval against a canned JSONL fixture.

This example demonstrates the core sivo workflow:

1. Store LLM execution records in a JSONL file (done once, via the
   execution engine or by constructing ExecutionRecord objects directly).
2. Write eval functions that assert against stored outputs.
3. Run ``sivo replay <run_id>`` to re-evaluate at zero API cost.

Directory layout expected by this example::

    examples/
    ├── eval_refund_policy.py     ← this file
    └── fixtures/
        └── refund_run.jsonl      ← canned records (created by make_fixture.py)

To try it end-to-end:

    # 1. Generate the canned fixture (writes examples/fixtures/refund_run.jsonl)
    python examples/make_fixture.py

    # 2. Run the evals in replay mode (zero API cost)
    sivo replay refund-run-001 examples/eval_refund_policy.py \\
        --store-path examples/fixtures

    # 3. Re-run with verbose output to see failure evidence
    sivo replay refund-run-001 examples/eval_refund_policy.py \\
        --store-path examples/fixtures -v
"""

from sivo.assertions import assert_contains, assert_not_contains


def eval_refund_policy_mentions_30_days(case):
    """The response must mention the 30-day return window."""
    assert_contains(case.output, "30 days")


def eval_refund_policy_no_restocking_fee(case):
    """The response must not mention a restocking fee (policy: none)."""
    assert_not_contains(case.output, "restocking fee")


def eval_refund_policy_includes_contact_info(case):
    """A helpful response directs the user to customer support."""
    assert_contains(case.output, "support")

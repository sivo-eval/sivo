"""Generate canned JSONL fixtures for the sivo examples.

Run once to create the fixture files used by the examples:

    python examples/make_fixture.py

Creates:
    examples/fixtures/records/refund-run-001.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from sivo.models import ExecutionRecord
from sivo.store import JsonlStore

STORE_PATH = Path(__file__).parent / "fixtures"
RUN_ID = "refund-run-001"

RECORDS = [
    ExecutionRecord(
        id="rec-001",
        timestamp="2026-01-01T10:00:00+00:00",
        run_id=RUN_ID,
        input="What is your refund policy?",
        output=(
            "Our refund policy allows returns within 30 days of purchase. "
            "Items must be unused and in original packaging. "
            "For assistance, please contact our support team."
        ),
        model="claude-haiku-4-5",
        input_tokens=15,
        output_tokens=42,
        cost_usd=0.0,
    ),
    ExecutionRecord(
        id="rec-002",
        timestamp="2026-01-01T10:01:00+00:00",
        run_id=RUN_ID,
        input="Can I return a product after 60 days?",
        output=(
            "Returns after 30 days are generally not accepted. "
            "However, please contact our support team to discuss your specific situation."
        ),
        model="claude-haiku-4-5",
        input_tokens=18,
        output_tokens=35,
        cost_usd=0.0,
    ),
    ExecutionRecord(
        id="rec-003",
        timestamp="2026-01-01T10:02:00+00:00",
        run_id=RUN_ID,
        input="Is there a restocking fee?",
        output=(
            "No, there is no restocking fee for returns made within 30 days. "
            "Contact our support team for details."
        ),
        model="claude-haiku-4-5",
        input_tokens=12,
        output_tokens=28,
        cost_usd=0.0,
    ),
]


def main() -> None:
    store = JsonlStore(STORE_PATH)
    for record in RECORDS:
        store.write(record)

    jsonl_path = STORE_PATH / "records" / f"{RUN_ID}.jsonl"
    print(f"Wrote {len(RECORDS)} records to {jsonl_path}")
    print(f"\nNow run:")
    print(f"  sivo replay {RUN_ID} examples/eval_refund_policy.py \\")
    print(f"      --store-path {STORE_PATH}")


if __name__ == "__main__":
    main()

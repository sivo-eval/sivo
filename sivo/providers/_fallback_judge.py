"""Prompt-based JSON fallback judge for providers without native function calling.

Use this when a provider doesn't support tool_use or function calling.
The judge prompt is extended with a JSON schema, and the raw JSON response
is parsed and validated via Pydantic.
"""

from __future__ import annotations

import json
import re

from sivo.models import JudgeVerdict

# JSON schema appended to the system prompt to guide raw-JSON responses.
_VERDICT_SCHEMA_BLOCK = """
You MUST respond with ONLY valid JSON matching this exact schema — no prose, no markdown fences:

{
  "passed": <true|false>,
  "reason": "<one sentence explaining the verdict>",
  "evidence": "<direct quote from the model response, or '<no direct evidence>'>",
  "suggestion": "<optional one-sentence improvement suggestion, or null>"
}
"""


def build_fallback_system_prompt(base_system_prompt: str) -> str:
    """Append the JSON schema instruction to *base_system_prompt*.

    Args:
        base_system_prompt: The judge's normal system prompt.

    Returns:
        Extended prompt that asks the model to return raw JSON.
    """
    return base_system_prompt.rstrip() + "\n\n" + _VERDICT_SCHEMA_BLOCK.strip()


def parse_fallback_response(raw_text: str, rubric_name: str = "") -> JudgeVerdict:
    """Parse a raw JSON response from the model into a :class:`~sivo.models.JudgeVerdict`.

    Strips markdown fences if present, then parses and validates.

    Args:
        raw_text:    The model's raw text output.
        rubric_name: Used only in error messages.

    Returns:
        :class:`~sivo.models.JudgeVerdict` parsed from the JSON.

    Raises:
        ValueError: If the response cannot be parsed or is missing required fields.
    """
    # Strip optional markdown fences (```json ... ``` or ``` ... ```)
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Judge response for rubric={rubric_name!r} is not valid JSON: {exc}\n"
            f"Raw response: {raw_text!r}"
        ) from exc

    required = {"passed", "reason", "evidence"}
    missing = required - data.keys()
    if missing:
        raise ValueError(
            f"Judge JSON response for rubric={rubric_name!r} is missing fields: "
            f"{sorted(missing)}. Got: {list(data.keys())}"
        )

    return JudgeVerdict(
        passed=bool(data["passed"]),
        reason=str(data["reason"]),
        evidence=str(data["evidence"]),
        suggestion=str(data["suggestion"]) if data.get("suggestion") else None,
    )

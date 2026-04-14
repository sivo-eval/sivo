"""Shared pytest configuration for sivo tests."""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--no-llm",
        action="store_true",
        default=False,
        help="Skip tests that make real LLM API calls.",
    )

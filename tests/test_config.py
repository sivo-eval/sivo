"""Unit tests for sivo.config — TOML loading and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from sivo.config import SivoConfig, _find_config, _parse_config, load_config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_config_has_expected_values():
    cfg = SivoConfig()
    assert cfg.default_model == "claude-haiku-4-5"
    assert cfg.concurrency == 10
    assert cfg.timeout == 30.0
    assert cfg.store_path == ".sivo"
    assert cfg.provider == "anthropic"
    assert cfg.judge_model == "claude-haiku-4-5"
    assert cfg.judge_provider == ""
    assert cfg.judge_retry_attempts == 1
    assert cfg.cost_warn_above_usd == 1.00


def test_load_config_returns_defaults_when_no_file(tmp_path):
    cfg = load_config(search_path=tmp_path)
    assert cfg.default_model == "claude-haiku-4-5"
    assert cfg.store_path == ".sivo"


# ---------------------------------------------------------------------------
# _find_config
# ---------------------------------------------------------------------------


def test_find_config_returns_none_when_missing(tmp_path):
    assert _find_config(tmp_path) is None


def test_find_config_finds_file_in_start_dir(tmp_path):
    toml = tmp_path / "sivo.toml"
    toml.write_text("[sivo]\n")
    assert _find_config(tmp_path) == toml


def test_find_config_finds_file_in_parent(tmp_path):
    child = tmp_path / "sub" / "project"
    child.mkdir(parents=True)
    toml = tmp_path / "sivo.toml"
    toml.write_text("[sivo]\n")
    assert _find_config(child) == toml


def test_find_config_prefers_nearest_ancestor(tmp_path):
    child = tmp_path / "sub"
    child.mkdir()
    parent_toml = tmp_path / "sivo.toml"
    parent_toml.write_text("[sivo]\ndefault_model = 'claude-opus-4-6'\n")
    child_toml = child / "sivo.toml"
    child_toml.write_text("[sivo]\ndefault_model = 'claude-haiku-4-5'\n")
    # child dir found first
    assert _find_config(child) == child_toml


# ---------------------------------------------------------------------------
# _parse_config
# ---------------------------------------------------------------------------


def test_parse_config_empty_dict_gives_defaults():
    cfg = _parse_config({})
    assert cfg.default_model == "claude-haiku-4-5"


def test_parse_config_top_level_fields():
    cfg = _parse_config({
        "sivo": {
            "default_model": "claude-opus-4-6",
            "concurrency": 5,
            "timeout": 60.0,
            "store_path": "/tmp/evals",
            "provider": "openai",
        }
    })
    assert cfg.default_model == "claude-opus-4-6"
    assert cfg.concurrency == 5
    assert cfg.timeout == 60.0
    assert cfg.store_path == "/tmp/evals"
    assert cfg.provider == "openai"


def test_parse_config_judge_provider():
    cfg = _parse_config({
        "sivo": {
            "judge": {"provider": "anthropic", "default_model": "claude-haiku-4-5"}
        }
    })
    assert cfg.judge_provider == "anthropic"


def test_parse_config_judge_section():
    cfg = _parse_config({
        "sivo": {
            "judge": {
                "default_model": "claude-sonnet-4-6",
                "retry_attempts": 3,
            }
        }
    })
    assert cfg.judge_model == "claude-sonnet-4-6"
    assert cfg.judge_retry_attempts == 3


def test_parse_config_cost_section():
    cfg = _parse_config({
        "sivo": {
            "cost": {"warn_above_usd": 5.0}
        }
    })
    assert cfg.cost_warn_above_usd == 5.0


def test_parse_config_unknown_keys_ignored():
    """Forward-compatible: unknown keys don't raise."""
    cfg = _parse_config({
        "sivo": {
            "future_feature": "ignored",
            "default_model": "claude-haiku-4-5",
        }
    })
    assert cfg.default_model == "claude-haiku-4-5"


def test_parse_config_partial_overrides():
    """Only specified keys are overridden; others keep defaults."""
    cfg = _parse_config({"sivo": {"concurrency": 20}})
    assert cfg.concurrency == 20
    assert cfg.default_model == "claude-haiku-4-5"  # not overridden


# ---------------------------------------------------------------------------
# load_config — full round-trip with real TOML files
# ---------------------------------------------------------------------------


def test_load_config_reads_toml_file(tmp_path):
    (tmp_path / "sivo.toml").write_text(
        "[sivo]\n"
        "default_model = 'claude-sonnet-4-6'\n"
        "concurrency = 4\n"
        "store_path = 'my-store'\n"
        "\n"
        "[sivo.judge]\n"
        "default_model = 'claude-haiku-4-5'\n"
        "retry_attempts = 3\n"
        "\n"
        "[sivo.cost]\n"
        "warn_above_usd = 2.50\n"
    )
    cfg = load_config(tmp_path)
    assert cfg.default_model == "claude-sonnet-4-6"
    assert cfg.concurrency == 4
    assert cfg.store_path == "my-store"
    assert cfg.judge_model == "claude-haiku-4-5"
    assert cfg.judge_retry_attempts == 3
    assert cfg.cost_warn_above_usd == 2.50


def test_load_config_partial_file(tmp_path):
    """A file with only some keys filled in keeps defaults for the rest."""
    (tmp_path / "sivo.toml").write_text(
        "[sivo]\ntimeout = 120\n"
    )
    cfg = load_config(tmp_path)
    assert cfg.timeout == 120.0
    assert cfg.default_model == "claude-haiku-4-5"


def test_load_config_searches_upward_from_subdir(tmp_path):
    subdir = tmp_path / "a" / "b"
    subdir.mkdir(parents=True)
    (tmp_path / "sivo.toml").write_text(
        "[sivo]\nconcurrency = 99\n"
    )
    cfg = load_config(subdir)
    assert cfg.concurrency == 99

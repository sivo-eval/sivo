"""E2e test: JUnit XML well-formedness and JSON summary.

Setup
-----
One JSONL record processed by three eval functions:

  eval_alpha — always passes           → PASS
  eval_beta  — always passes           → PASS
  eval_gamma — always fails (assert)   → FAIL

This gives 2 PASS + 1 FAIL from a single subprocess invocation.

Scenarios
---------
A  sivo run --no-fail-fast --junit-xml <path>
   - Exit code is 1 (due to FAIL)
   - XML file is created and valid
   - Correct <testcase> count
   - Passing evals have no <failure> child
   - Failing eval has <failure> element with correct type/message

B  sivo run --no-fail-fast --junit-xml <path> with a flaky eval
   - Flaky result produces <skipped> element by default
   - With --strict-flaky, flaky produces <failure> and exit code 1

C  JSON summary is always written to .sivo/results/<run_id>.json
   - File exists after every run
   - Content matches expected counts
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "run_junit_test"

# ---------------------------------------------------------------------------
# Eval file content
# ---------------------------------------------------------------------------

_EVAL_PASS_PASS_FAIL = """\
from sivo.assertions import EvalAssertionError


def eval_alpha(case):
    pass


def eval_beta(case):
    pass


def eval_gamma(case):
    raise EvalAssertionError(
        "response does not meet requirements",
        assertion_type="assert_contains",
    )
"""

_EVAL_WITH_FLAKY = """\
from sivo.assertions import EvalAssertionError, FlakyEvalError


def eval_stable(case):
    pass


def eval_unstable(case):
    raise FlakyEvalError("split judge verdict")
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(record_id: str = "rec-0") -> dict:
    return {
        "id": record_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": RUN_ID,
        "input": "test prompt",
        "output": "some model output",
        "model": "claude-haiku-4-5",
        "params": {},
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.0001,
        "metadata": {},
        "system_prompt": None,
        "conversation": None,
        "trace": None,
    }


def _build_project(tmp_path: Path, eval_content: str = _EVAL_PASS_PASS_FAIL) -> None:
    (tmp_path / "eval_suite.py").write_text(eval_content)
    records_dir = tmp_path / ".sivo" / "records"
    records_dir.mkdir(parents=True)
    (records_dir / f"{RUN_ID}.jsonl").write_text(
        json.dumps(_make_record()) + "\n"
    )


def _run(tmp_path: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "sivo.cli",
            "run", "eval_suite.py",
            "--run-id", RUN_ID,
            "--no-fail-fast",
            *extra_args,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def _out(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


def _parse_xml(path: Path) -> ET.Element:
    tree = ET.parse(str(path))
    return tree.getroot()


# ---------------------------------------------------------------------------
# Scenario A: --junit-xml with pass + fail
# ---------------------------------------------------------------------------


def test_junit_exit_code_is_one(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    proc = _run(tmp_path, "--junit-xml", str(xml_path))
    assert proc.returncode == 1, _out(proc)


def test_junit_xml_file_created(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    assert xml_path.exists()


def test_junit_xml_is_valid_xml(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    # Should not raise
    ET.parse(str(xml_path))


def test_junit_xml_root_is_testsuites(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    root = _parse_xml(xml_path)
    assert root.tag == "testsuites"


def test_junit_xml_testcase_count(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    root = _parse_xml(xml_path)
    cases = root.findall(".//testcase")
    assert len(cases) == 3


def test_junit_xml_pass_testcases_have_no_failure(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    root = _parse_xml(xml_path)
    cases = root.findall(".//testcase")
    # Find cases with no failure child
    passing = [c for c in cases if c.find("failure") is None]
    assert len(passing) == 2


def test_junit_xml_fail_testcase_has_failure_element(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    root = _parse_xml(xml_path)
    failures = root.findall(".//failure")
    assert len(failures) == 1


def test_junit_xml_failure_type_attribute(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    failure = _parse_xml(xml_path).find(".//failure")
    assert failure.attrib["type"] == "EvalAssertionError"


def test_junit_xml_failure_message_attribute(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    failure = _parse_xml(xml_path).find(".//failure")
    assert "requirements" in failure.attrib["message"]


def test_junit_xml_failures_count_in_testsuite(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    suite = _parse_xml(xml_path).find(".//testsuite")
    assert suite.attrib["failures"] == "1"


def test_junit_xml_tests_count_in_testsuite(tmp_path):
    _build_project(tmp_path)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    suite = _parse_xml(xml_path).find(".//testsuite")
    assert suite.attrib["tests"] == "3"


# ---------------------------------------------------------------------------
# Scenario B: flaky results
# ---------------------------------------------------------------------------


def test_junit_flaky_default_is_skipped(tmp_path):
    _build_project(tmp_path, eval_content=_EVAL_WITH_FLAKY)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path))
    skipped = _parse_xml(xml_path).findall(".//skipped")
    assert len(skipped) == 1


def test_junit_flaky_default_exit_zero(tmp_path):
    """Flaky without --strict-flaky → exit 0 (no hard failures)."""
    _build_project(tmp_path, eval_content=_EVAL_WITH_FLAKY)
    xml_path = tmp_path / "results.xml"
    proc = _run(tmp_path, "--junit-xml", str(xml_path))
    assert proc.returncode == 0, _out(proc)


def test_junit_strict_flaky_converts_to_failure(tmp_path):
    _build_project(tmp_path, eval_content=_EVAL_WITH_FLAKY)
    xml_path = tmp_path / "results.xml"
    _run(tmp_path, "--junit-xml", str(xml_path), "--strict-flaky")
    failures = _parse_xml(xml_path).findall(".//failure")
    skipped = _parse_xml(xml_path).findall(".//skipped")
    assert len(failures) == 1
    assert len(skipped) == 0


def test_junit_strict_flaky_exit_one(tmp_path):
    _build_project(tmp_path, eval_content=_EVAL_WITH_FLAKY)
    xml_path = tmp_path / "results.xml"
    proc = _run(tmp_path, "--junit-xml", str(xml_path), "--strict-flaky")
    assert proc.returncode == 1, _out(proc)


# ---------------------------------------------------------------------------
# Scenario C: JSON summary always written
# ---------------------------------------------------------------------------


def test_json_summary_written_automatically(tmp_path):
    _build_project(tmp_path)
    _run(tmp_path)
    json_path = tmp_path / ".sivo" / "results" / f"{RUN_ID}.json"
    assert json_path.exists()


def test_json_summary_valid_content(tmp_path):
    _build_project(tmp_path)
    _run(tmp_path)
    json_path = tmp_path / ".sivo" / "results" / f"{RUN_ID}.json"
    data = json.loads(json_path.read_text())
    assert data["run_id"] == RUN_ID
    assert data["passed"] == 2
    assert data["failed"] == 1
    assert data["total"] == 3


def test_json_summary_results_array(tmp_path):
    _build_project(tmp_path)
    _run(tmp_path)
    json_path = tmp_path / ".sivo" / "results" / f"{RUN_ID}.json"
    data = json.loads(json_path.read_text())
    assert len(data["results"]) == 3


def test_json_summary_written_even_with_failures(tmp_path):
    """JSON summary is written regardless of pass/fail outcome."""
    _build_project(tmp_path)
    proc = _run(tmp_path)
    assert proc.returncode == 1  # has failures
    json_path = tmp_path / ".sivo" / "results" / f"{RUN_ID}.json"
    assert json_path.exists()

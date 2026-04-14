"""Unit tests for Phase 8 CI integration.

Covers:
- SessionResult.is_success with/without strict_flaky
- write_junit_xml: XML structure, failure/skipped elements, strict_flaky
- write_json_summary: JSON content and file location
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from sivo.assertions import EvalAssertionError, FlakyEvalError
from sivo.models import JudgeVerdict
from sivo.report import write_json_summary, write_junit_xml
from sivo.runner import EvalResult, SessionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pass(eval_name: str = "eval_pass", record_id: str = "r1") -> EvalResult:
    return EvalResult(eval_name=eval_name, record_id=record_id, passed=True)


def _fail(eval_name: str = "eval_fail", record_id: str = "r2") -> EvalResult:
    err = EvalAssertionError("text missing 'good'", assertion_type="assert_contains")
    return EvalResult(eval_name=eval_name, record_id=record_id, passed=False, error=err)


def _flaky(eval_name: str = "eval_flaky", record_id: str = "r3") -> EvalResult:
    err = FlakyEvalError("split verdict 1/2")
    return EvalResult(eval_name=eval_name, record_id=record_id, passed=True, flaky=True, error=err)


def _session(*, passed=1, failed=1, flaky=1) -> SessionResult:
    s = SessionResult(run_id="run_test")
    for i in range(passed):
        s.results.append(_pass(f"eval_pass_{i}", f"rp{i}"))
    for i in range(failed):
        s.results.append(_fail(f"eval_fail_{i}", f"rf{i}"))
    for i in range(flaky):
        s.results.append(_flaky(f"eval_flaky_{i}", f"rfl{i}"))
    return s


def _parse_xml(path: Path) -> ET.Element:
    tree = ET.parse(str(path))
    return tree.getroot()


# ---------------------------------------------------------------------------
# SessionResult.is_success
# ---------------------------------------------------------------------------


def test_is_success_all_pass():
    s = SessionResult(run_id="r")
    s.results = [_pass()]
    assert s.is_success() is True


def test_is_success_with_failure():
    s = SessionResult(run_id="r")
    s.results = [_pass(), _fail()]
    assert s.is_success() is False


def test_is_success_flaky_is_ok_by_default():
    s = SessionResult(run_id="r")
    s.results = [_pass(), _flaky()]
    assert s.is_success() is True


def test_is_success_strict_flaky_false_allows_flaky():
    s = SessionResult(run_id="r")
    s.results = [_pass(), _flaky()]
    assert s.is_success(strict_flaky=False) is True


def test_is_success_strict_flaky_rejects_flaky():
    s = SessionResult(run_id="r")
    s.results = [_pass(), _flaky()]
    assert s.is_success(strict_flaky=True) is False


def test_is_success_strict_flaky_with_failure():
    s = SessionResult(run_id="r")
    s.results = [_fail(), _flaky()]
    assert s.is_success(strict_flaky=True) is False


def test_is_success_empty_session():
    s = SessionResult(run_id="r")
    assert s.is_success() is True


# ---------------------------------------------------------------------------
# write_junit_xml — file creation
# ---------------------------------------------------------------------------


def test_junit_xml_creates_file(tmp_path):
    s = _session()
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    assert out.exists()


def test_junit_xml_creates_parent_dirs(tmp_path):
    s = _session()
    out = tmp_path / "ci" / "reports" / "results.xml"
    write_junit_xml(s, out)
    assert out.exists()


def test_junit_xml_is_valid_xml(tmp_path):
    s = _session()
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    # Should not raise
    ET.parse(str(out))


# ---------------------------------------------------------------------------
# write_junit_xml — root structure
# ---------------------------------------------------------------------------


def test_junit_xml_root_is_testsuites(tmp_path):
    s = _session()
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    root = _parse_xml(out)
    assert root.tag == "testsuites"


def test_junit_xml_has_testsuite_child(tmp_path):
    s = _session()
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    root = _parse_xml(out)
    suites = root.findall("testsuite")
    assert len(suites) == 1


def test_junit_xml_testsuite_name_is_sivo(tmp_path):
    s = _session()
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    assert suite.attrib["name"] == "sivo"


def test_junit_xml_tests_count(tmp_path):
    s = _session(passed=2, failed=1, flaky=1)
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    assert suite.attrib["tests"] == "4"


def test_junit_xml_failures_count(tmp_path):
    s = _session(passed=2, failed=1, flaky=0)
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    assert suite.attrib["failures"] == "1"


def test_junit_xml_errors_is_zero(tmp_path):
    s = _session()
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    assert suite.attrib["errors"] == "0"


# ---------------------------------------------------------------------------
# write_junit_xml — testcase elements
# ---------------------------------------------------------------------------


def test_junit_xml_testcase_count_matches_results(tmp_path):
    s = _session(passed=2, failed=1, flaky=1)
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    cases = suite.findall("testcase")
    assert len(cases) == 4


def test_junit_xml_testcase_name_includes_eval_and_record(tmp_path):
    s = SessionResult(run_id="run1")
    s.results = [_pass("eval_tone", "rec-abc")]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    case = suite.find("testcase")
    assert case.attrib["name"] == "eval_tone[rec-abc]"


def test_junit_xml_testcase_classname_is_run_id(tmp_path):
    s = SessionResult(run_id="my_run")
    s.results = [_pass()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    case = suite.find("testcase")
    assert case.attrib["classname"] == "my_run"


# ---------------------------------------------------------------------------
# write_junit_xml — pass result
# ---------------------------------------------------------------------------


def test_junit_xml_pass_has_no_children(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_pass()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    case = suite.find("testcase")
    assert len(list(case)) == 0  # no child elements


# ---------------------------------------------------------------------------
# write_junit_xml — fail result
# ---------------------------------------------------------------------------


def test_junit_xml_fail_has_failure_element(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_fail()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    case = suite.find("testcase")
    failure = case.find("failure")
    assert failure is not None


def test_junit_xml_failure_type_attribute(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_fail()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    failure = _parse_xml(out).find(".//failure")
    assert failure.attrib["type"] == "EvalAssertionError"


def test_junit_xml_failure_message_is_first_error_line(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_fail()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    failure = _parse_xml(out).find(".//failure")
    assert "text missing" in failure.attrib["message"]


def test_junit_xml_failure_text_contains_full_error(tmp_path):
    err = EvalAssertionError(
        "line one\nline two\nline three", assertion_type="assert_contains"
    )
    result = EvalResult("eval_x", "r1", False, error=err)
    s = SessionResult(run_id="r")
    s.results = [result]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    failure = _parse_xml(out).find(".//failure")
    assert "line one" in failure.text
    assert "line two" in failure.text


# ---------------------------------------------------------------------------
# write_junit_xml — flaky result (default: skipped)
# ---------------------------------------------------------------------------


def test_junit_xml_flaky_has_skipped_element(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_flaky()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    case = suite.find("testcase")
    skipped = case.find("skipped")
    assert skipped is not None


def test_junit_xml_flaky_skipped_message_starts_with_flaky(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_flaky()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    skipped = _parse_xml(out).find(".//skipped")
    assert skipped.attrib["message"].startswith("FLAKY:")


def test_junit_xml_flaky_not_counted_as_failure_by_default(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_flaky()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out)
    suite = _parse_xml(out).find("testsuite")
    assert suite.attrib["failures"] == "0"
    assert suite.attrib["skipped"] == "1"


# ---------------------------------------------------------------------------
# write_junit_xml — strict_flaky
# ---------------------------------------------------------------------------


def test_junit_xml_strict_flaky_converts_to_failure(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_flaky()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out, strict_flaky=True)
    case = _parse_xml(out).find(".//testcase")
    assert case.find("failure") is not None
    assert case.find("skipped") is None


def test_junit_xml_strict_flaky_counted_in_failures(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_flaky()]
    out = tmp_path / "results.xml"
    write_junit_xml(s, out, strict_flaky=True)
    suite = _parse_xml(out).find("testsuite")
    assert suite.attrib["failures"] == "1"
    assert suite.attrib["skipped"] == "0"


# ---------------------------------------------------------------------------
# write_json_summary — content
# ---------------------------------------------------------------------------


def test_json_summary_creates_file(tmp_path):
    s = _session()
    write_json_summary(s, tmp_path / ".sivo")
    out = tmp_path / ".sivo" / "results" / "run_test.json"
    assert out.exists()


def test_json_summary_creates_parent_dirs(tmp_path):
    s = SessionResult(run_id="myrun")
    write_json_summary(s, tmp_path / ".sivo")
    out = tmp_path / ".sivo" / "results" / "myrun.json"
    assert out.exists()


def test_json_summary_is_valid_json(tmp_path):
    s = _session()
    write_json_summary(s, tmp_path)
    out = tmp_path / "results" / "run_test.json"
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_json_summary_run_id(tmp_path):
    s = SessionResult(run_id="myrun")
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "myrun.json").read_text())
    assert data["run_id"] == "myrun"


def test_json_summary_counts(tmp_path):
    s = _session(passed=2, failed=1, flaky=1)
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "run_test.json").read_text())
    assert data["passed"] == 2
    assert data["failed"] == 1
    assert data["flaky"] == 1
    assert data["total"] == 4


def test_json_summary_results_array(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_pass("eval_x", "r1"), _fail("eval_y", "r2")]
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "r.json").read_text())
    assert len(data["results"]) == 2
    names = {r["eval_name"] for r in data["results"]}
    assert names == {"eval_x", "eval_y"}


def test_json_summary_result_fields(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_pass()]
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "r.json").read_text())
    result = data["results"][0]
    assert "eval_name" in result
    assert "record_id" in result
    assert "passed" in result
    assert "flaky" in result
    assert "error" in result


def test_json_summary_pass_error_is_null(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_pass()]
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "r.json").read_text())
    assert data["results"][0]["error"] is None


def test_json_summary_fail_error_is_string(tmp_path):
    s = SessionResult(run_id="r")
    s.results = [_fail()]
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "r.json").read_text())
    assert isinstance(data["results"][0]["error"], str)
    assert "text missing" in data["results"][0]["error"]


def test_json_summary_cost_fields(tmp_path):
    s = SessionResult(
        run_id="r",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=0.00042,
    )
    write_json_summary(s, tmp_path)
    data = json.loads((tmp_path / "results" / "r.json").read_text())
    assert data["total_input_tokens"] == 100
    assert data["total_output_tokens"] == 50
    assert pytest.approx(data["total_cost_usd"]) == 0.00042

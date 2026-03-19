"""Tests for evaluators — no LLM calls needed."""

import pytest
from unittest.mock import MagicMock
from reflexion.evaluator import FunctionEvaluator, LLMEvaluator, PythonCodeEvaluator


class TestFunctionEvaluator:
    def test_pass(self):
        ev = FunctionEvaluator(lambda task, out: (True, 1.0, "great"))
        result = ev.evaluate("task", "output")
        assert result.success is True
        assert result.score == 1.0
        assert result.feedback == "great"

    def test_fail(self):
        ev = FunctionEvaluator(lambda task, out: (False, 0.2, "too short"))
        result = ev.evaluate("task", "x")
        assert result.success is False
        assert result.score == 0.2


class TestLLMEvaluator:
    def _make_llm(self, raw_response: str):
        llm = MagicMock()
        llm.complete.return_value = raw_response
        return llm

    def test_parses_passing_response(self):
        llm = self._make_llm("SCORE: 9\nPASS: YES\nFEEDBACK: Looks good.")
        ev = LLMEvaluator(llm)
        result = ev.evaluate("task", "output")
        assert result.success is True
        assert result.score == pytest.approx(0.9)
        assert result.feedback == "Looks good."

    def test_parses_failing_response(self):
        llm = self._make_llm("SCORE: 3\nPASS: NO\nFEEDBACK: Missing error handling.")
        ev = LLMEvaluator(llm)
        result = ev.evaluate("task", "output")
        assert result.success is False
        assert result.score == pytest.approx(0.3)

    def test_malformed_response_defaults(self):
        llm = self._make_llm("I cannot parse this")
        ev = LLMEvaluator(llm)
        result = ev.evaluate("task", "output")
        assert result.score == pytest.approx(0.5)  # default score
        assert result.success is False


class TestPythonCodeEvaluator:
    TASK_WITH_TESTS = """
Write a function `add(a, b)` that returns the sum.

```python
# TESTS
assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
```
"""

    def test_passing_code(self):
        ev = PythonCodeEvaluator()
        good_code = "```python\ndef add(a, b):\n    return a + b\n```"
        result = ev.evaluate(self.TASK_WITH_TESTS, good_code)
        assert result.success is True
        assert result.score == pytest.approx(1.0)

    def test_failing_code(self):
        ev = PythonCodeEvaluator()
        bad_code = "```python\ndef add(a, b):\n    return a - b  # wrong!\n```"
        result = ev.evaluate(self.TASK_WITH_TESTS, bad_code)
        assert result.success is False
        assert result.score < 0.5

    def test_no_tests_falls_back_to_llm(self):
        llm = MagicMock()
        llm.complete.return_value = "SCORE: 7\nPASS: YES\nFEEDBACK: fine"
        from reflexion.evaluator import LLMEvaluator
        ev = PythonCodeEvaluator(llm_fallback=LLMEvaluator(llm))
        result = ev.evaluate("write something", "some code")
        assert result.success is True

    def test_no_tests_no_fallback_fails(self):
        ev = PythonCodeEvaluator()
        result = ev.evaluate("write something", "some code")
        assert result.success is False

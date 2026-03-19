"""Tests for the Reflector."""

import pytest
from unittest.mock import MagicMock
from reflexion.reflector import Reflector
from reflexion.types import EvaluationResult


class TestReflector:
    def test_returns_llm_output(self):
        llm = MagicMock()
        llm.complete.return_value = "You forgot error handling."
        r = Reflector(llm)
        eval_ = EvaluationResult(success=False, score=0.3, feedback="missing edge cases")
        text = r.reflect("write a sort function", "def sort(x): return x", eval_, 0)
        assert text == "You forgot error handling."

    def test_passes_task_and_output_to_llm(self):
        llm = MagicMock()
        llm.complete.return_value = "reflection"
        r = Reflector(llm)
        eval_ = EvaluationResult(success=False, score=0.4, feedback="bad")
        r.reflect("MY_TASK", "MY_OUTPUT", eval_, 0)
        call_content = llm.complete.call_args.kwargs["messages"][0]["content"]
        assert "MY_TASK" in call_content
        assert "MY_OUTPUT" in call_content

    def test_system_prompt_present(self):
        llm = MagicMock()
        llm.complete.return_value = "reflection"
        r = Reflector(llm)
        eval_ = EvaluationResult(success=False, score=0.5, feedback="meh")
        r.reflect("task", "output", eval_, 1)
        system = llm.complete.call_args.kwargs.get("system", "")
        assert len(system) > 0

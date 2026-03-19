"""Integration tests for ReflexionAgent — all LLM calls mocked."""

import pytest
from unittest.mock import MagicMock, patch
from reflexion.agent import ReflexionAgent
from reflexion.evaluator import FunctionEvaluator
from reflexion.types import EvaluationResult, Reflection


def _always_pass_evaluator():
    return FunctionEvaluator(lambda t, o: (True, 1.0, "perfect"))


def _always_fail_evaluator():
    return FunctionEvaluator(lambda t, o: (False, 0.1, "terrible"))


def _improving_evaluator():
    """Fails first two calls, passes on the third."""
    calls = [0]
    def _eval(task, output):
        calls[0] += 1
        if calls[0] < 3:
            return False, 0.2, "not yet"
        return True, 0.9, "great"
    return FunctionEvaluator(_eval)


def _mock_llm(response: str = "mock output") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = response
    return llm


class TestReflexionAgentLoop:
    def test_succeeds_on_first_try(self):
        agent = ReflexionAgent(
            llm=_mock_llm("good answer"),
            evaluator=_always_pass_evaluator(),
            max_iterations=3,
        )
        result = agent.run("What is 2+2?")
        assert result.success is True
        assert result.iterations_used == 1
        assert len(result.reflections) == 0
        assert result.final_output == "good answer"

    def test_stops_after_max_iterations_on_all_fail(self):
        agent = ReflexionAgent(
            llm=_mock_llm(),
            evaluator=_always_fail_evaluator(),
            max_iterations=3,
        )
        result = agent.run("hard task")
        assert result.success is False
        assert result.iterations_used == 3
        # Two reflections produced (after iter 1 and iter 2; no reflection after last)
        assert len(result.reflections) == 2

    def test_improves_over_iterations(self):
        agent = ReflexionAgent(
            llm=_mock_llm(),
            evaluator=_improving_evaluator(),
            max_iterations=5,
        )
        result = agent.run("task that needs retries")
        assert result.success is True
        assert result.iterations_used == 3

    def test_on_iteration_callback_called(self):
        called = []
        agent = ReflexionAgent(
            llm=_mock_llm(),
            evaluator=_improving_evaluator(),
            max_iterations=5,
            on_iteration=lambda r: called.append(r),
        )
        agent.run("task")
        # Two failed iterations produce two reflections → callback called twice
        assert len(called) == 2
        assert all(isinstance(r, Reflection) for r in called)

    def test_success_threshold_respected(self):
        # Score 0.7 but threshold is 0.9 — should keep trying
        calls = [0]
        def _eval(task, output):
            calls[0] += 1
            return True, 0.7, "ok"

        agent = ReflexionAgent(
            llm=_mock_llm(),
            evaluator=FunctionEvaluator(_eval),
            max_iterations=3,
            success_threshold=0.9,
        )
        result = agent.run("task")
        assert result.success is False  # never reached threshold
        assert result.iterations_used == 3

    def test_result_metadata_populated(self):
        agent = ReflexionAgent(
            llm=_mock_llm(),
            evaluator=_always_pass_evaluator(),
        )
        result = agent.run("task")
        assert "timestamp" in result.metadata
        assert result.metadata["max_iterations"] == 3
        assert result.metadata["success_threshold"] == 0.8

"""Tests for core data types."""

import pytest
from reflexion.types import EvaluationResult, Reflection, ReflexionResult


def _make_reflection(iteration: int, score: float) -> Reflection:
    return Reflection(
        iteration=iteration,
        actor_output="some output",
        evaluation=EvaluationResult(success=score >= 0.8, score=score, feedback="ok"),
        reflection_text="try harder",
    )


class TestEvaluationResult:
    def test_fields(self):
        e = EvaluationResult(success=True, score=0.9, feedback="great")
        assert e.success is True
        assert e.score == 0.9
        assert e.feedback == "great"

    def test_details_default_empty(self):
        e = EvaluationResult(success=False, score=0.0, feedback="bad")
        assert e.details == {}


class TestReflexionResult:
    def test_improved_false_with_no_reflections(self):
        r = ReflexionResult(
            task="t",
            final_output="o",
            success=True,
            iterations_used=1,
            reflections=[],
            final_score=0.9,
        )
        assert r.improved is False

    def test_improved_false_with_one_reflection(self):
        r = ReflexionResult(
            task="t",
            final_output="o",
            success=True,
            iterations_used=2,
            reflections=[_make_reflection(0, 0.5)],
            final_score=0.9,
        )
        # Only one reflection — need at least two to compare
        assert r.improved is False

    def test_improved_true(self):
        r = ReflexionResult(
            task="t",
            final_output="o",
            success=True,
            iterations_used=3,
            reflections=[
                _make_reflection(0, 0.3),
                _make_reflection(1, 0.8),
            ],
            final_score=1.0,
        )
        assert r.improved is True

    def test_improved_false_when_score_drops(self):
        r = ReflexionResult(
            task="t",
            final_output="o",
            success=False,
            iterations_used=3,
            reflections=[
                _make_reflection(0, 0.7),
                _make_reflection(1, 0.4),
            ],
            final_score=0.4,
        )
        assert r.improved is False

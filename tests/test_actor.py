"""Tests for the Actor — verifies prompt construction without LLM calls."""

import pytest
from unittest.mock import MagicMock
from reflexion.actor import Actor
from reflexion.types import EvaluationResult, Reflection


def _make_llm(response: str = "actor output") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = response
    return llm


def _make_reflection(i: int) -> Reflection:
    return Reflection(
        iteration=i,
        actor_output=f"output_{i}",
        evaluation=EvaluationResult(success=False, score=0.3, feedback="not good"),
        reflection_text=f"reflection_{i}",
    )


class TestActor:
    def test_no_reflections_sends_task_as_single_message(self):
        llm = _make_llm()
        actor = Actor(llm)
        actor.act("do something", [])
        call_kwargs = llm.complete.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "do something" in messages[0]["content"]

    def test_one_reflection_includes_history(self):
        llm = _make_llm()
        actor = Actor(llm)
        actor.act("do something", [_make_reflection(0)])
        messages = llm.complete.call_args.kwargs["messages"]
        # user task + assistant output + reflection followup
        assert len(messages) == 3
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user"]
        # Last user message should reference the reflection
        assert "reflection_0" in messages[2]["content"]

    def test_two_reflections_correct_message_count(self):
        llm = _make_llm()
        actor = Actor(llm)
        actor.act("task", [_make_reflection(0), _make_reflection(1)])
        messages = llm.complete.call_args.kwargs["messages"]
        # 2 × (user + assistant + reflection) = 6 messages
        assert len(messages) == 6

    def test_system_prompt_passed(self):
        llm = _make_llm()
        actor = Actor(llm)
        actor.act("task", [])
        call_kwargs = llm.complete.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

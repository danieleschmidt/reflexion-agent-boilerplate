"""Reflector: generates verbal feedback from a failed attempt.

The reflection is injected into the actor's context on the next iteration,
giving the agent a chance to course-correct without any gradient updates.
"""

from __future__ import annotations

import textwrap
from .types import EvaluationResult


class Reflector:
    """LLM-based reflector."""

    SYSTEM = textwrap.dedent("""
        You are a self-critical AI assistant helping to improve a previous answer.

        Your job is to write a SHORT, CONCRETE reflection (3-5 sentences max) that:
        1. Identifies exactly what went wrong in the previous attempt
        2. Explains why it fell short of the task requirements
        3. Gives specific, actionable guidance for the next attempt

        Do NOT repeat the task. Do NOT rewrite the answer. Just diagnose and advise.
    """).strip()

    def __init__(self, llm):
        self.llm = llm

    def reflect(
        self,
        task: str,
        output: str,
        evaluation: EvaluationResult,
        iteration: int,
    ) -> str:
        prompt = (
            f"TASK:\n{task}\n\n"
            f"ATTEMPT #{iteration + 1}:\n{output}\n\n"
            f"EVALUATION FEEDBACK:\n{evaluation.feedback}\n"
            f"SCORE: {evaluation.score:.1%}"
        )
        return self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.SYSTEM,
        )

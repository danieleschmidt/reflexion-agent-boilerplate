"""Evaluators: decide whether an actor's output is good enough.

You can plug in any evaluator — a simple heuristic, a unit test runner,
or an LLM-as-judge.  The interface is just:

    evaluate(task, output) -> EvaluationResult
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
from typing import Callable, Protocol

from .types import EvaluationResult


class Evaluator(Protocol):
    def evaluate(self, task: str, output: str) -> EvaluationResult:
        ...


class LLMEvaluator:
    """Use an LLM to judge whether the output satisfies the task.

    Produces a score from 0–10 and a brief explanation.
    """

    SYSTEM = textwrap.dedent("""
        You are a strict evaluator. Given a task and a response, judge quality.

        Respond in EXACTLY this format (no extra text):
        SCORE: <integer 0-10>
        PASS: <YES or NO>
        FEEDBACK: <one sentence explaining the score>
    """).strip()

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, task: str, output: str) -> EvaluationResult:
        prompt = f"TASK:\n{task}\n\nRESPONSE:\n{output}"
        raw = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.SYSTEM,
        )

        score = self._parse_score(raw)
        passed = self._parse_pass(raw)
        feedback = self._parse_feedback(raw)

        return EvaluationResult(
            success=passed,
            score=score / 10.0,
            feedback=feedback,
            details={"raw_evaluation": raw},
        )

    @staticmethod
    def _parse_score(text: str) -> int:
        m = re.search(r"SCORE:\s*(\d+)", text)
        return int(m.group(1)) if m else 5

    @staticmethod
    def _parse_pass(text: str) -> bool:
        m = re.search(r"PASS:\s*(YES|NO)", text, re.IGNORECASE)
        return m.group(1).upper() == "YES" if m else False

    @staticmethod
    def _parse_feedback(text: str) -> str:
        m = re.search(r"FEEDBACK:\s*(.+)", text)
        return m.group(1).strip() if m else text.strip()


class PythonCodeEvaluator:
    """Evaluate Python code by actually running it.

    Expects the task to contain test cases as assertions in a
    ```python ... ``` block marked with `# TESTS`.  If no such block
    exists, falls back to the LLM evaluator.
    """

    def __init__(self, llm_fallback=None, timeout: float = 10.0):
        self.llm_fallback = llm_fallback
        self.timeout = timeout

    def evaluate(self, task: str, output: str) -> EvaluationResult:
        # Extract test block from task
        tests = self._extract_tests(task)
        if not tests:
            if self.llm_fallback:
                return self.llm_fallback.evaluate(task, output)
            return EvaluationResult(
                success=False,
                score=0.0,
                feedback="No test cases found and no LLM fallback configured.",
            )

        # Extract code from output (```python ... ``` or bare)
        code = self._extract_code(output)
        full_script = code + "\n\n" + tests

        return self._run_tests(full_script)

    def _run_tests(self, script: str) -> EvaluationResult:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            fname = f.name

        try:
            result = subprocess.run(
                ["python3", fname],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                return EvaluationResult(
                    success=True,
                    score=1.0,
                    feedback="All tests passed.",
                    details={"stdout": result.stdout},
                )
            else:
                return EvaluationResult(
                    success=False,
                    score=0.2,
                    feedback=f"Tests failed: {result.stderr.strip()[:300]}",
                    details={"stderr": result.stderr, "stdout": result.stdout},
                )
        except subprocess.TimeoutExpired:
            return EvaluationResult(
                success=False,
                score=0.0,
                feedback=f"Code timed out after {self.timeout}s.",
            )

    @staticmethod
    def _extract_tests(task: str) -> str:
        """Pull out lines after '# TESTS' in a fenced code block."""
        m = re.search(r"```python\s*(.*?)```", task, re.DOTALL)
        if not m:
            return ""
        block = m.group(1)
        if "# TESTS" in block:
            idx = block.index("# TESTS")
            return block[idx:]
        return ""

    @staticmethod
    def _extract_code(output: str) -> str:
        """Pull code from a fenced block, or return the output as-is."""
        m = re.search(r"```python\s*(.*?)```", output, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"```\s*(.*?)```", output, re.DOTALL)
        if m:
            return m.group(1)
        return output


class FunctionEvaluator:
    """Wrap a plain Python callable as an evaluator.

    The function receives (task, output) and must return
    (passed: bool, score: float, feedback: str).
    """

    def __init__(self, fn: Callable[[str, str], tuple[bool, float, str]]):
        self._fn = fn

    def evaluate(self, task: str, output: str) -> EvaluationResult:
        passed, score, feedback = self._fn(task, output)
        return EvaluationResult(success=passed, score=score, feedback=feedback)

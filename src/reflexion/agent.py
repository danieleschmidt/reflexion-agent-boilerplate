"""ReflexionAgent: orchestrates the actor → evaluate → reflect → retry loop."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from .actor import Actor
from .evaluator import Evaluator, LLMEvaluator
from .llm import AnthropicBackend, LLMBackend
from .reflector import Reflector
from .types import EvaluationResult, Reflection, ReflexionResult

logger = logging.getLogger(__name__)


class ReflexionAgent:
    """Self-improving agent using the Reflexion loop.

    Parameters
    ----------
    llm:
        LLM backend (default: AnthropicBackend using $ANTHROPIC_API_KEY).
        Must implement `.complete(messages, system) -> str`.
    evaluator:
        Evaluator that scores actor outputs.  Defaults to LLMEvaluator.
    max_iterations:
        Maximum number of attempt + reflect cycles.
    success_threshold:
        Score (0–1) at which we consider the task solved.
    on_iteration:
        Optional callback called after each iteration with the Reflection.
        Useful for logging / streaming progress.
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        evaluator: Evaluator | None = None,
        max_iterations: int = 3,
        success_threshold: float = 0.8,
        on_iteration: Callable[[Reflection], None] | None = None,
    ):
        self.llm = llm or AnthropicBackend()
        self.actor = Actor(self.llm)
        self.evaluator = evaluator or LLMEvaluator(self.llm)
        self.reflector = Reflector(self.llm)
        self.max_iterations = max_iterations
        self.success_threshold = success_threshold
        self.on_iteration = on_iteration

    def run(self, task: str) -> ReflexionResult:
        """Execute the Reflexion loop for *task* and return the final result.

        Algorithm:
            for i in range(max_iterations):
                output   = actor.act(task, reflections_so_far)
                eval     = evaluator.evaluate(task, output)
                if eval.success and eval.score >= threshold:
                    break
                reflection = reflector.reflect(task, output, eval, i)
                reflections.append(reflection)
        """
        reflections: list[Reflection] = []
        final_output = ""
        final_eval = EvaluationResult(success=False, score=0.0, feedback="Not attempted")

        logger.info("Starting Reflexion loop | task=%r | max_iter=%d", task[:80], self.max_iterations)

        for i in range(self.max_iterations):
            logger.info("Iteration %d/%d", i + 1, self.max_iterations)

            # ── 1. Act ─────────────────────────────────────────────────────────
            output = self.actor.act(task, reflections)
            logger.debug("Actor output (iter %d): %s…", i + 1, output[:120])

            # ── 2. Evaluate ────────────────────────────────────────────────────
            evaluation = self.evaluator.evaluate(task, output)
            logger.info(
                "Evaluation (iter %d): score=%.2f success=%s feedback=%r",
                i + 1,
                evaluation.score,
                evaluation.success,
                evaluation.feedback[:80],
            )
            final_output = output
            final_eval = evaluation

            if evaluation.success and evaluation.score >= self.success_threshold:
                logger.info("Task solved on iteration %d", i + 1)
                break

            if i < self.max_iterations - 1:
                # ── 3. Reflect ─────────────────────────────────────────────────
                reflection_text = self.reflector.reflect(task, output, evaluation, i)
                logger.debug("Reflection (iter %d): %s…", i + 1, reflection_text[:120])

                reflection = Reflection(
                    iteration=i,
                    actor_output=output,
                    evaluation=evaluation,
                    reflection_text=reflection_text,
                )
                reflections.append(reflection)

                if self.on_iteration:
                    self.on_iteration(reflection)

        return ReflexionResult(
            task=task,
            final_output=final_output,
            success=final_eval.success and final_eval.score >= self.success_threshold,
            iterations_used=len(reflections) + 1,
            reflections=reflections,
            final_score=final_eval.score,
            metadata={
                "max_iterations": self.max_iterations,
                "success_threshold": self.success_threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

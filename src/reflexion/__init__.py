"""Reflexion: Self-reflection loop for LLM agents.

Based on: "Reflexion: Language Agents with Verbal Reinforcement Learning"
(Shinn et al., 2023) — https://arxiv.org/abs/2303.11366

Core loop:
    1. Actor generates a response/action
    2. Evaluator scores the outcome
    3. Reflector generates verbal feedback on what went wrong
    4. Actor retries with prior reflections in context
"""

from .agent import ReflexionAgent
from .types import ReflexionResult, Reflection, EvaluationResult

__all__ = ["ReflexionAgent", "ReflexionResult", "Reflection", "EvaluationResult"]
__version__ = "1.0.0"

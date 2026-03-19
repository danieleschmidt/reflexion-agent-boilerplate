"""Core data types for the Reflexion agent."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class EvaluationResult:
    """Result of evaluating an actor's output against the task."""

    success: bool
    score: float  # 0.0 – 1.0
    feedback: str  # Human-readable explanation of what passed / failed
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reflection:
    """A single verbal reflection produced by the reflector."""

    iteration: int
    actor_output: str
    evaluation: EvaluationResult
    reflection_text: str  # The actual verbal feedback injected into next attempt
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ReflexionResult:
    """Final result of a Reflexion loop run."""

    task: str
    final_output: str
    success: bool
    iterations_used: int
    reflections: list[Reflection]
    final_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """Whether the score improved over the run (at least one reflection helped)."""
        if len(self.reflections) < 2:
            return False
        first = self.reflections[0].evaluation.score
        last = self.reflections[-1].evaluation.score
        return last > first

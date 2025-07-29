"""Main ReflexionAgent class implementation."""

from typing import Any, Dict, Optional

from .engine import ReflexionEngine
from .types import ReflectionType, ReflexionResult


class ReflexionAgent:
    """Main interface for reflexion-enhanced agents."""

    def __init__(
        self,
        llm: str,
        max_iterations: int = 3,
        reflection_type: ReflectionType = ReflectionType.BINARY,
        success_threshold: float = 0.8,
        **kwargs
    ):
        """Initialize reflexion agent.
        
        Args:
            llm: LLM model identifier
            max_iterations: Maximum reflection iterations
            reflection_type: Type of reflection to perform
            success_threshold: Threshold for considering task successful
            **kwargs: Additional configuration options
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.reflection_type = reflection_type
        self.success_threshold = success_threshold
        self.engine = ReflexionEngine(**kwargs)

    def run(self, task: str, success_criteria: Optional[str] = None, **kwargs) -> ReflexionResult:
        """Execute task with reflexion enhancement.
        
        Args:
            task: Task description to execute
            success_criteria: Optional success criteria
            **kwargs: Additional execution parameters
            
        Returns:
            ReflexionResult containing execution details and reflections
        """
        return self.engine.execute_with_reflexion(
            task=task,
            llm=self.llm,
            max_iterations=self.max_iterations,
            reflection_type=self.reflection_type,
            success_threshold=self.success_threshold,
            success_criteria=success_criteria,
            **kwargs
        )
"""End-to-end integration tests."""

import pytest

from src.reflexion.core.agent import ReflexionAgent
from src.reflexion.core.types import ReflectionType


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_reflexion_cycle(self):
        """Test complete reflexion cycle with mock LLM."""
        agent = ReflexionAgent(
            llm="test-model",
            max_iterations=3,
            reflection_type=ReflectionType.BINARY,
            success_threshold=0.7
        )
        
        result = agent.run(
            task="Write a function to reverse a string",
            success_criteria="Function should handle edge cases"
        )
        
        # Verify result structure
        assert result.task == "Write a function to reverse a string"
        assert isinstance(result.success, bool)
        assert result.iterations >= 1
        assert result.total_time > 0
        assert isinstance(result.reflections, list)
        
        # If it failed, should have reflections
        if not result.success:
            assert len(result.reflections) > 0
            for reflection in result.reflections:
                assert len(reflection.issues) > 0
                assert len(reflection.improvements) > 0

    def test_early_success_no_reflections(self):
        """Test case where task succeeds immediately."""
        agent = ReflexionAgent(
            llm="test-model",
            max_iterations=3,
            success_threshold=0.1  # Very low threshold for easy success
        )
        
        result = agent.run("This is a comprehensive solution with detailed implementation")
        
        # Should succeed on first try with low threshold
        assert result.success is True
        assert result.iterations == 1
        assert len(result.reflections) == 0

    def test_multiple_reflection_iterations(self):
        """Test multiple reflection iterations."""
        agent = ReflexionAgent(
            llm="test-model",
            max_iterations=3,
            success_threshold=0.95  # Very high threshold to force multiple iterations
        )
        
        result = agent.run("short")  # Short input likely to fail evaluation
        
        # Should go through multiple iterations
        assert result.iterations > 1
        assert len(result.reflections) > 0
        
        # Each reflection should have learning content
        for reflection in result.reflections:
            assert reflection.task == "short"
            assert isinstance(reflection.success, bool)
            assert 0 <= reflection.score <= 1
            assert reflection.confidence > 0
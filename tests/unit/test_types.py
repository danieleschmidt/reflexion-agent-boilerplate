"""Unit tests for type definitions."""

import pytest
from datetime import datetime

from src.reflexion.core.types import ReflectionType, Reflection, ReflexionResult


class TestTypes:
    """Test suite for type definitions."""

    def test_reflection_type_enum(self):
        """Test ReflectionType enum values."""
        assert ReflectionType.BINARY.value == "binary"
        assert ReflectionType.SCALAR.value == "scalar"
        assert ReflectionType.STRUCTURED.value == "structured"

    def test_reflection_dataclass(self):
        """Test Reflection dataclass creation."""
        reflection = Reflection(
            task="test task",
            output="test output",
            success=True,
            score=0.8,
            issues=["issue1"],
            improvements=["improvement1"],
            confidence=0.9,
            timestamp="2024-01-01T00:00:00"
        )
        
        assert reflection.task == "test task"
        assert reflection.success is True
        assert reflection.score == 0.8
        assert len(reflection.issues) == 1
        assert len(reflection.improvements) == 1

    def test_reflexion_result_dataclass(self):
        """Test ReflexionResult dataclass creation."""
        reflection = Reflection(
            task="test",
            output="output",
            success=False,
            score=0.5,
            issues=[],
            improvements=[],
            confidence=0.7,
            timestamp="2024-01-01"
        )
        
        result = ReflexionResult(
            task="test task",
            output="final output",
            success=True,
            iterations=2,
            reflections=[reflection],
            total_time=1.5,
            metadata={"llm": "test-model"}
        )
        
        assert result.task == "test task"
        assert result.success is True
        assert result.iterations == 2
        assert len(result.reflections) == 1
        assert result.total_time == 1.5
        assert result.metadata["llm"] == "test-model"
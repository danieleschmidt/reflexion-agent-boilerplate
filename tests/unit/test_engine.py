"""Unit tests for ReflexionEngine."""

import pytest
from unittest.mock import patch, Mock

from src.reflexion.core.engine import ReflexionEngine
from src.reflexion.core.types import ReflectionType, ReflexionResult, Reflection


class TestReflexionEngine:
    """Test suite for ReflexionEngine class."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ReflexionEngine(custom_param="test")
        assert engine.config["custom_param"] == "test"

    def test_execute_with_reflexion_success_first_try(self):
        """Test successful execution on first attempt."""
        engine = ReflexionEngine()
        
        with patch.object(engine, '_execute_task') as mock_execute:
            with patch.object(engine, '_evaluate_output') as mock_evaluate:
                mock_execute.return_value = "Perfect solution for test task"
                mock_evaluate.return_value = {"success": True, "score": 0.9}
                
                result = engine.execute_with_reflexion(
                    task="test task",
                    llm="test-model",
                    max_iterations=3,
                    reflection_type=ReflectionType.BINARY,
                    success_threshold=0.8
                )
                
                assert isinstance(result, ReflexionResult)
                assert result.success is True
                assert result.iterations == 1
                assert len(result.reflections) == 0
                assert result.task == "test task"

    def test_execute_with_reflexion_success_after_reflection(self):
        """Test successful execution after reflection."""
        engine = ReflexionEngine()
        
        with patch.object(engine, '_execute_task') as mock_execute:
            with patch.object(engine, '_evaluate_output') as mock_evaluate:
                with patch.object(engine, '_generate_reflection') as mock_reflect:
                    # First attempt fails, second succeeds
                    mock_execute.side_effect = ["Poor solution", "Great solution"]
                    mock_evaluate.side_effect = [
                        {"success": False, "score": 0.3},
                        {"success": True, "score": 0.9}
                    ]
                    mock_reflect.return_value = Mock(spec=Reflection)
                    
                    result = engine.execute_with_reflexion(
                        task="test task",
                        llm="test-model", 
                        max_iterations=3,
                        reflection_type=ReflectionType.BINARY,
                        success_threshold=0.8
                    )
                    
                    assert result.success is True
                    assert result.iterations == 2
                    assert len(result.reflections) == 1

    def test_execute_task_with_reflections(self):
        """Test task execution with previous reflections."""
        engine = ReflexionEngine()
        mock_reflection = Mock(spec=Reflection)
        mock_reflection.improvements = ["Be more specific", "Add examples"]
        
        result = engine._execute_task("test task", "test-model", 1, [mock_reflection])
        
        assert "iteration 2" in result
        assert "Improved solution" in result

    def test_evaluate_output_success(self):
        """Test output evaluation for successful case."""
        engine = ReflexionEngine()
        
        result = engine._evaluate_output(
            "test task",
            "This is a comprehensive solution with detailed implementation",
            "criteria"
        )
        
        assert result["success"] is True
        assert result["score"] > 0.5

    def test_evaluate_output_failure(self):
        """Test output evaluation for failure case."""
        engine = ReflexionEngine()
        
        result = engine._evaluate_output("test task", "short", "criteria")
        
        assert result["success"] is False
        assert result["score"] < 0.5

    def test_generate_reflection(self):
        """Test reflection generation."""
        engine = ReflexionEngine()
        evaluation = {"success": False, "score": 0.3}
        
        reflection = engine._generate_reflection(
            "test task",
            "poor output", 
            evaluation,
            ReflectionType.BINARY
        )
        
        assert isinstance(reflection, Reflection)
        assert reflection.success is False
        assert reflection.score == 0.3
        assert len(reflection.issues) > 0
        assert len(reflection.improvements) > 0
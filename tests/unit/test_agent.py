"""Unit tests for ReflexionAgent."""

import pytest
from unittest.mock import patch, Mock

from src.reflexion.core.agent import ReflexionAgent
from src.reflexion.core.types import ReflectionType, ReflexionResult


class TestReflexionAgent:
    """Test suite for ReflexionAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization with default parameters."""
        agent = ReflexionAgent(llm="test-model")
        
        assert agent.llm == "test-model"
        assert agent.max_iterations == 3
        assert agent.reflection_type == ReflectionType.BINARY
        assert agent.success_threshold == 0.8

    def test_agent_initialization_with_custom_params(self):
        """Test agent initialization with custom parameters."""
        agent = ReflexionAgent(
            llm="custom-model",
            max_iterations=5,
            reflection_type=ReflectionType.SCALAR,
            success_threshold=0.9
        )
        
        assert agent.llm == "custom-model"
        assert agent.max_iterations == 5
        assert agent.reflection_type == ReflectionType.SCALAR
        assert agent.success_threshold == 0.9

    @patch('src.reflexion.core.agent.ReflexionEngine')
    def test_run_delegates_to_engine(self, mock_engine_class):
        """Test that run method delegates to engine correctly."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_result = Mock(spec=ReflexionResult)
        mock_engine.execute_with_reflexion.return_value = mock_result
        
        agent = ReflexionAgent(llm="test-model")
        result = agent.run("test task", success_criteria="test criteria")
        
        assert result == mock_result
        mock_engine.execute_with_reflexion.assert_called_once_with(
            task="test task",
            llm="test-model",
            max_iterations=3,
            reflection_type=ReflectionType.BINARY,
            success_threshold=0.8,
            success_criteria="test criteria"
        )

    def test_run_without_success_criteria(self, reflexion_agent):
        """Test run method without success criteria."""
        # This would normally call the engine, but we'll test integration separately
        result = reflexion_agent.run("simple task")
        
        assert isinstance(result, ReflexionResult)
        assert result.task == "simple task"
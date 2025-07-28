import pytest
from unittest.mock import Mock, patch

from reflexion.core import ReflexionAgent, ReflectionType
from reflexion.exceptions import ReflexionError


class TestReflexionAgent:
    """Test cases for the core ReflexionAgent class."""

    def test_init_default_params(self):
        """Test agent initialization with default parameters."""
        agent = ReflexionAgent(llm="gpt-4")
        assert agent.llm == "gpt-4"
        assert agent.max_iterations == 3
        assert agent.reflection_type == ReflectionType.BINARY

    def test_init_custom_params(self, mock_memory, mock_evaluator):
        """Test agent initialization with custom parameters."""
        agent = ReflexionAgent(
            llm="gpt-4",
            max_iterations=5,
            reflection_type=ReflectionType.SCALAR,
            memory=mock_memory,
            evaluator=mock_evaluator
        )
        assert agent.max_iterations == 5
        assert agent.reflection_type == ReflectionType.SCALAR
        assert agent.memory == mock_memory
        assert agent.evaluator == mock_evaluator

    def test_run_success_first_iteration(self, mock_evaluator):
        """Test successful task completion on first iteration."""
        mock_evaluator.evaluate.return_value = {
            "success": True, 
            "score": 0.9,
            "details": {}
        }
        
        agent = ReflexionAgent(llm="gpt-4", evaluator=mock_evaluator)
        
        with patch.object(agent, '_execute_task') as mock_execute:
            mock_execute.return_value = "Perfect solution"
            
            result = agent.run("Test task")
            
            assert result.success is True
            assert result.iterations == 1
            assert result.output == "Perfect solution"
            assert len(result.reflections) == 0

    def test_run_failure_then_success(self, mock_evaluator):
        """Test failure on first iteration, success after reflection."""
        # First evaluation: failure, second: success
        mock_evaluator.evaluate.side_effect = [
            {"success": False, "score": 0.4, "details": {"issues": ["bug"]}},
            {"success": True, "score": 0.8, "details": {}}
        ]
        
        agent = ReflexionAgent(llm="gpt-4", evaluator=mock_evaluator)
        
        with patch.object(agent, '_execute_task') as mock_execute, \
             patch.object(agent, 'reflect') as mock_reflect:
            
            mock_execute.side_effect = ["Buggy solution", "Fixed solution"]
            mock_reflect.return_value = "Found and fixed the bug"
            
            result = agent.run("Test task")
            
            assert result.success is True
            assert result.iterations == 2
            assert result.output == "Fixed solution"
            assert len(result.reflections) == 1

    def test_run_max_iterations_exceeded(self, mock_evaluator):
        """Test behavior when max iterations is exceeded."""
        mock_evaluator.evaluate.return_value = {
            "success": False, 
            "score": 0.3,
            "details": {}
        }
        
        agent = ReflexionAgent(llm="gpt-4", evaluator=mock_evaluator, max_iterations=2)
        
        with patch.object(agent, '_execute_task') as mock_execute, \
             patch.object(agent, 'reflect') as mock_reflect:
            
            mock_execute.return_value = "Still buggy"
            mock_reflect.return_value = "Tried to fix"
            
            result = agent.run("Test task")
            
            assert result.success is False
            assert result.iterations == 2
            assert len(result.reflections) == 2

    def test_reflect_binary_type(self):
        """Test reflection with binary reflection type."""
        agent = ReflexionAgent(llm="gpt-4", reflection_type=ReflectionType.BINARY)
        
        evaluation = {"success": False, "score": 0.4, "details": {"issues": ["bug"]}}
        
        with patch.object(agent, '_generate_reflection') as mock_gen:
            mock_gen.return_value = "Identified the bug"
            
            reflection = agent.reflect("task", "output", evaluation)
            
            assert reflection is not None
            mock_gen.assert_called_once()

    def test_memory_storage(self, mock_memory):
        """Test that experiences are stored in memory."""
        agent = ReflexionAgent(llm="gpt-4", memory=mock_memory)
        
        with patch.object(agent, '_execute_task') as mock_execute, \
             patch.object(agent, 'evaluator') as mock_eval:
            
            mock_execute.return_value = "Solution"
            mock_eval.evaluate.return_value = {"success": True, "score": 0.9}
            
            agent.run("Test task")
            
            mock_memory.store.assert_called_once()

    def test_invalid_llm_model(self):
        """Test error handling for invalid LLM model."""
        with pytest.raises(ReflexionError):
            ReflexionAgent(llm="invalid-model")

    def test_empty_task(self):
        """Test error handling for empty task."""
        agent = ReflexionAgent(llm="gpt-4")
        
        with pytest.raises(ValueError):
            agent.run("")

    def test_reflection_prompts_loading(self):
        """Test loading of domain-specific reflection prompts."""
        agent = ReflexionAgent(llm="gpt-4")
        
        # Test default prompts are loaded
        assert hasattr(agent, 'reflection_prompts')
        assert agent.reflection_prompts is not None
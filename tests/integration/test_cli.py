"""Integration tests for CLI interface."""

import pytest
from unittest.mock import patch, Mock
import sys
from io import StringIO

from src.reflexion.cli import main


class TestCLI:
    """Test suite for CLI interface."""

    @patch('sys.argv', ['reflexion', 'test task'])
    @patch('src.reflexion.cli.ReflexionAgent')
    def test_cli_basic_execution(self, mock_agent_class):
        """Test basic CLI execution."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_result = Mock()
        mock_result.task = "test task"
        mock_result.success = True
        mock_result.iterations = 1
        mock_result.output = "test output"
        mock_result.reflections = []
        mock_agent.run.return_value = mock_result
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
            
        output = mock_stdout.getvalue()
        assert "Task: test task" in output
        assert "Success: True" in output
        assert "Iterations: 1" in output

    @patch('sys.argv', ['reflexion', 'test task', '--llm', 'custom-model', '--max-iterations', '5'])
    @patch('src.reflexion.cli.ReflexionAgent')
    def test_cli_with_custom_args(self, mock_agent_class):
        """Test CLI with custom arguments."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        mock_result = Mock()
        mock_result.task = "test task"
        mock_result.success = True
        mock_result.iterations = 1
        mock_result.output = "test output"
        mock_result.reflections = []
        mock_agent.run.return_value = mock_result
        
        main()
        
        # Verify agent was created with custom parameters
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        assert call_args[1]['llm'] == 'custom-model'
        assert call_args[1]['max_iterations'] == 5

    @patch('sys.argv', ['reflexion', 'failing task'])
    @patch('src.reflexion.cli.ReflexionAgent')
    def test_cli_with_reflections(self, mock_agent_class):
        """Test CLI output with reflections."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        mock_reflection = Mock()
        mock_reflection.issues = ["Issue 1", "Issue 2"]
        mock_reflection.improvements = ["Improvement 1"]
        
        mock_result = Mock()
        mock_result.task = "failing task"
        mock_result.success = False
        mock_result.iterations = 3
        mock_result.output = "final attempt"
        mock_result.reflections = [mock_reflection]
        mock_agent.run.return_value = mock_result
        
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
            
        output = mock_stdout.getvalue()
        assert "Reflections:" in output
        assert "Issues: Issue 1, Issue 2" in output
        assert "Improvements: Improvement 1" in output
"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock

from src.reflexion.core.agent import ReflexionAgent
from src.reflexion.core.types import ReflectionType


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    return Mock()


@pytest.fixture
def reflexion_agent():
    """Basic reflexion agent for testing."""
    return ReflexionAgent(
        llm="test-model",
        max_iterations=3,
        reflection_type=ReflectionType.BINARY
    )


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return "Write a Python function to calculate fibonacci numbers"


@pytest.fixture
def sample_success_criteria():
    """Sample success criteria for testing."""
    return "Function should handle edge cases and be efficient"
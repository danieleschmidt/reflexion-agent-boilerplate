import os
import tempfile
import pytest
from unittest.mock import Mock, patch
from typing import Generator

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test environment setup
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Test database


@pytest.fixture(scope="session")
def test_db():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    from reflexion.database import Base
    Base.metadata.create_all(bind=engine)
    
    yield TestingSessionLocal
    

@pytest.fixture
def db_session(test_db):
    """Create database session for each test."""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        )
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    with patch("anthropic.Anthropic") as mock:
        mock_client = Mock()
        mock_client.messages.create.return_value = Mock(
            content=[Mock(text="Test response")]
        )
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return {
        "description": "Write a Python function to calculate factorial",
        "success_criteria": "Function should handle edge cases and return correct results",
        "domain": "coding"
    }


@pytest.fixture
def sample_reflection():
    """Sample reflection data for testing."""
    return {
        "task_id": "test_task_1",
        "iteration": 1,
        "output": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "evaluation": {"success": False, "score": 0.6, "issues": ["No input validation"]},
        "reflection": "The function lacks input validation for negative numbers",
        "improvement_strategy": "Add input validation to handle negative numbers"
    }


@pytest.fixture
def temp_directory():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_memory():
    """Mock memory system for testing."""
    memory = Mock()
    memory.store.return_value = None
    memory.recall.return_value = []
    memory.extract_patterns.return_value = []
    return memory


@pytest.fixture
def mock_evaluator():
    """Mock evaluator for testing."""
    evaluator = Mock()
    evaluator.evaluate.return_value = {
        "success": True,
        "score": 0.8,
        "details": {"test_results": "All tests passed"}
    }
    return evaluator


@pytest.fixture
def reflexion_config():
    """Default reflexion configuration for testing."""
    return {
        "max_iterations": 3,
        "reflection_type": "binary",
        "success_threshold": 0.8,
        "enable_memory": True,
        "enable_telemetry": False
    }


# Benchmark fixtures
@pytest.fixture
def benchmark_tasks():
    """Load benchmark tasks for performance testing."""
    return [
        {"id": f"task_{i}", "description": f"Benchmark task {i}", "complexity": "medium"}
        for i in range(10)
    ]


# Integration test fixtures
@pytest.fixture(scope="session")
def integration_env():
    """Setup integration test environment."""
    # Start test services if needed
    yield
    # Cleanup


# Performance test markers
pytest_plugins = ["pytest_benchmark"]
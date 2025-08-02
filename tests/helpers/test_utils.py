"""Test utilities and helper functions."""

import asyncio
import json
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Callable
from unittest.mock import patch, MagicMock
import pytest


class TestEnvironment:
    """Test environment setup and teardown."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.original_cwd = Path.cwd()
        
    def __enter__(self):
        """Enter test environment."""
        self.setup()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit test environment."""
        self.cleanup()
        
    def setup(self):
        """Setup test environment."""
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create common test directories
        (self.temp_dir / "data").mkdir(exist_ok=True)
        (self.temp_dir / "logs").mkdir(exist_ok=True)
        (self.temp_dir / "cache").mkdir(exist_ok=True)
        
    def cleanup(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def create_file(self, path: str, content: str) -> Path:
        """Create a file with content in the test environment."""
        file_path = self.temp_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
        
    def create_json_file(self, path: str, data: Dict[str, Any]) -> Path:
        """Create a JSON file in the test environment."""
        content = json.dumps(data, indent=2)
        return self.create_file(path, content)


@contextmanager  
def mock_environment_variables(env_vars: Dict[str, str]) -> Generator[None, None, None]:
    """Mock environment variables for testing."""
    with patch.dict('os.environ', env_vars):
        yield


@contextmanager
def capture_logs(logger_name: str = "reflexion") -> Generator[List[str], None, None]:
    """Capture log messages for testing."""
    import logging
    
    log_messages = []
    
    class TestHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(self.format(record))
    
    logger = logging.getLogger(logger_name)
    handler = TestHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        yield log_messages
    finally:
        logger.removeHandler(handler)


def assert_response_quality(response: str, min_length: int = 10, 
                          required_keywords: Optional[List[str]] = None) -> None:
    """Assert that a response meets quality criteria."""
    assert len(response) >= min_length, f"Response too short: {len(response)} < {min_length}"
    
    if required_keywords:
        response_lower = response.lower()
        for keyword in required_keywords:
            assert keyword.lower() in response_lower, f"Missing keyword '{keyword}' in response"


def assert_reflection_quality(reflection: Dict[str, Any]) -> None:
    """Assert that a reflection meets quality criteria."""
    required_fields = ["issues", "improvements", "confidence"]
    for field in required_fields:
        assert field in reflection, f"Missing required field '{field}' in reflection"
    
    assert isinstance(reflection["issues"], list), "Issues should be a list"
    assert isinstance(reflection["improvements"], list), "Improvements should be a list"
    assert 0 <= reflection["confidence"] <= 1, "Confidence should be between 0 and 1"


def create_test_memory_episode(task: str, outcome: str = "success", **kwargs) -> Dict[str, Any]:
    """Create a test memory episode with default values."""
    from ..fixtures.test_data import create_test_episode
    return create_test_episode(task, outcome, **kwargs)


def run_async_test(async_func: Callable, *args, **kwargs) -> Any:
    """Run an async function in a test context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(async_func(*args, **kwargs))
    finally:
        loop.close()


class MockTimer:
    """Mock timer for testing time-dependent functionality."""
    
    def __init__(self, initial_time: float = 0):
        self.current_time = initial_time
        
    def advance(self, seconds: float):
        """Advance the mock time."""
        self.current_time += seconds
        
    def time(self) -> float:
        """Get current mock time."""
        return self.current_time


@contextmanager
def mock_time(initial_time: float = 0) -> Generator[MockTimer, None, None]:
    """Mock time.time() for testing."""
    timer = MockTimer(initial_time)
    
    with patch('time.time', side_effect=timer.time):
        yield timer


def create_test_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a test configuration with optional overrides."""
    from ..fixtures.test_data import SAMPLE_CONFIG
    
    config = SAMPLE_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def assert_memory_episode_valid(episode: Dict[str, Any]) -> None:
    """Assert that a memory episode is valid."""
    required_fields = ["id", "task", "outcome", "reflection", "lessons", "timestamp", "confidence"]
    
    for field in required_fields:
        assert field in episode, f"Missing required field '{field}' in memory episode"
    
    assert episode["outcome"] in ["success", "failure"], "Outcome must be 'success' or 'failure'"
    assert isinstance(episode["lessons"], list), "Lessons should be a list"
    assert 0 <= episode["confidence"] <= 1, "Confidence should be between 0 and 1"


def create_mock_evaluator(evaluation_result: Dict[str, Any]) -> MagicMock:
    """Create a mock evaluator that returns a specific result."""
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = evaluation_result
    return mock_evaluator


def assert_improvement_over_iterations(results: List[Dict[str, Any]], 
                                     metric: str = "score") -> None:
    """Assert that a metric improves over iterations."""
    if len(results) < 2:
        return  # Can't check improvement with less than 2 results
        
    scores = [result.get(metric, 0) for result in results]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i-1], f"Score decreased from {scores[i-1]} to {scores[i]} at iteration {i}"


def create_performance_benchmark(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Create a performance benchmark for a function."""
    import time
    import tracemalloc
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "success": success,
        "result": result,
        "error": error,
        "execution_time": end_time - start_time,
        "memory_current": current,
        "memory_peak": peak
    }


class AsyncTestHelper:
    """Helper for async testing."""
    
    @staticmethod
    async def wait_for_condition(condition: Callable[[], bool], 
                               timeout: float = 5.0, 
                               interval: float = 0.1) -> bool:
        """Wait for a condition to become true."""
        import asyncio
        
        elapsed = 0
        while elapsed < timeout:
            if condition():
                return True
            await asyncio.sleep(interval)
            elapsed += interval
        return False
    
    @staticmethod
    async def run_with_timeout(coro: Callable, timeout: float = 5.0) -> Any:
        """Run a coroutine with a timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)


# Pytest fixtures that can be used across test files
@pytest.fixture
def test_env():
    """Provide a test environment."""
    with TestEnvironment() as env:
        yield env


@pytest.fixture
def mock_llm():
    """Provide a mock LLM."""
    from ..mocks.llm_mock import MockLLM
    return MockLLM()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return create_test_config()


@pytest.fixture
def captured_logs():
    """Capture logs during test execution."""
    with capture_logs() as logs:
        yield logs
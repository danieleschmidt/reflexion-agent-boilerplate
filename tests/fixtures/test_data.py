"""Test data fixtures for reflexion agent tests."""

import json
from typing import Dict, List, Any
from pathlib import Path

# Sample LLM responses for testing
SAMPLE_LLM_RESPONSES = {
    "code_generation": {
        "initial": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        "improved": """
def fibonacci(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    
    if n <= 1:
        return n
    
    # Use memoization for efficiency
    memo = {0: 0, 1: 1}
    for i in range(2, n + 1):
        memo[i] = memo[i-1] + memo[i-2]
    
    return memo[n]
"""
    },
    "reflection": {
        "self_critique": """
Looking at the initial solution, I can identify several issues:

1. **Performance Issue**: The recursive approach has exponential time complexity O(2^n)
2. **No Input Validation**: Doesn't handle negative numbers or non-integers
3. **No Edge Case Handling**: Doesn't explicitly validate input types
4. **Memory Usage**: Deep recursion can cause stack overflow for large n

The solution works for small inputs but fails on efficiency and robustness.
""",
        "improvement_plan": """
Based on the reflection, here's my improvement strategy:

1. **Add Input Validation**: Check for non-negative integers
2. **Use Dynamic Programming**: Replace recursion with iterative approach using memoization
3. **Optimize Space**: Use bottom-up approach to avoid recursion depth issues
4. **Add Documentation**: Include docstring with complexity analysis
5. **Add Error Handling**: Raise appropriate exceptions for invalid inputs

This will improve from O(2^n) to O(n) time complexity and O(n) to O(1) space complexity.
"""
    }
}

# Sample evaluation results
SAMPLE_EVALUATIONS = {
    "success": {
        "success": True,
        "score": 0.95,
        "details": {
            "correctness": 1.0,
            "efficiency": 0.9,
            "robustness": 0.95,
            "code_quality": 0.9
        },
        "feedback": "Excellent implementation with proper error handling and optimization."
    },
    "failure": {
        "success": False,
        "score": 0.4,
        "details": {
            "correctness": 0.8,
            "efficiency": 0.2,
            "robustness": 0.3,
            "code_quality": 0.3
        },
        "feedback": "Implementation is functionally correct but has performance issues and lacks input validation."
    }
}

# Sample memory episodes
SAMPLE_MEMORY_EPISODES = [
    {
        "id": "episode_001",
        "task": "implement fibonacci function",
        "outcome": "success",
        "reflection": "Learned to use dynamic programming for optimization",
        "lessons": ["avoid recursive solutions for sequence problems", "always validate inputs"],
        "timestamp": "2024-01-01T10:00:00Z",
        "confidence": 0.9
    },
    {
        "id": "episode_002", 
        "task": "implement binary search",
        "outcome": "failure",
        "reflection": "Forgot to handle edge case of empty array",
        "lessons": ["always consider empty input cases", "test boundary conditions"],
        "timestamp": "2024-01-01T11:00:00Z",
        "confidence": 0.7
    }
]

# Sample configuration for testing
SAMPLE_CONFIG = {
    "llm": {
        "provider": "test",
        "model": "test-model",
        "temperature": 0.1
    },
    "reflexion": {
        "max_iterations": 3,
        "reflection_type": "structured",
        "success_threshold": 0.8
    },
    "memory": {
        "type": "episodic",
        "capacity": 100,
        "consolidation_threshold": 0.8
    }
}

# Test prompts
TEST_PROMPTS = {
    "simple_task": "Write a function to reverse a string",
    "complex_task": "Implement a thread-safe LRU cache with TTL support",
    "debugging_task": "Fix the bug in this code: def add(a, b): return a - b",
    "optimization_task": "Optimize this sorting function for better performance"
}

# Expected reflection patterns
REFLECTION_PATTERNS = {
    "performance_issue": {
        "keywords": ["performance", "efficiency", "optimization", "complexity"],
        "severity": "high",
        "category": "optimization"
    },
    "input_validation": {
        "keywords": ["validation", "error handling", "edge cases", "input"],
        "severity": "medium", 
        "category": "robustness"
    },
    "correctness": {
        "keywords": ["bug", "incorrect", "wrong", "error"],
        "severity": "high",
        "category": "functionality"
    }
}

def load_test_data(data_type: str) -> Any:
    """Load test data by type."""
    data_map = {
        "llm_responses": SAMPLE_LLM_RESPONSES,
        "evaluations": SAMPLE_EVALUATIONS,
        "memory_episodes": SAMPLE_MEMORY_EPISODES,
        "config": SAMPLE_CONFIG,
        "prompts": TEST_PROMPTS,
        "reflection_patterns": REFLECTION_PATTERNS
    }
    return data_map.get(data_type, {})

def create_test_episode(task: str, outcome: str, **kwargs) -> Dict[str, Any]:
    """Create a test memory episode."""
    base_episode = {
        "id": f"test_episode_{hash(task) % 10000}",
        "task": task,
        "outcome": outcome,
        "reflection": kwargs.get("reflection", f"Test reflection for {task}"),
        "lessons": kwargs.get("lessons", [f"Test lesson from {task}"]),
        "timestamp": kwargs.get("timestamp", "2024-01-01T12:00:00Z"),
        "confidence": kwargs.get("confidence", 0.8)
    }
    base_episode.update(kwargs)
    return base_episode

def save_test_results(results: Dict[str, Any], filename: str) -> None:
    """Save test results to file."""
    test_dir = Path(__file__).parent
    results_dir = test_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_test_results(filename: str) -> Dict[str, Any]:
    """Load test results from file."""
    test_dir = Path(__file__).parent
    results_file = test_dir / "results" / filename
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}
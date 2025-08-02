"""Mock LLM implementations for testing."""

from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock
import json
import random
from ..fixtures.test_data import SAMPLE_LLM_RESPONSES


class MockLLM:
    """Mock LLM that returns predefined responses."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or SAMPLE_LLM_RESPONSES
        self.call_count = 0
        self.call_history = []
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response based on the prompt."""
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "kwargs": kwargs,
            "call_number": self.call_count
        })
        
        # Simple keyword matching to return appropriate responses
        if "fibonacci" in prompt.lower():
            return self._get_fibonacci_response(prompt)
        elif "reflect" in prompt.lower() or "critique" in prompt.lower():
            return self._get_reflection_response(prompt)
        elif "improve" in prompt.lower():
            return self._get_improvement_response(prompt)
        else:
            return self._get_default_response(prompt)
    
    def _get_fibonacci_response(self, prompt: str) -> str:
        """Get fibonacci-related response."""
        if self.call_count == 1:
            return self.responses["code_generation"]["initial"]
        else:
            return self.responses["code_generation"]["improved"]
    
    def _get_reflection_response(self, prompt: str) -> str:
        """Get reflection response."""
        return self.responses["reflection"]["self_critique"]
    
    def _get_improvement_response(self, prompt: str) -> str:
        """Get improvement plan response."""
        return self.responses["reflection"]["improvement_plan"]
    
    def _get_default_response(self, prompt: str) -> str:
        """Get default response for unknown prompts."""
        return f"Mock response for: {prompt[:50]}..."


class ProgressiveLLM(MockLLM):
    """Mock LLM that gets better over time (simulates learning)."""
    
    def __init__(self, initial_quality: float = 0.3, improvement_rate: float = 0.1):
        super().__init__()
        self.quality = initial_quality
        self.improvement_rate = improvement_rate
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate progressively better responses."""
        response = super().generate(prompt, **kwargs)
        
        # Simulate improvement over time
        if random.random() < self.quality:
            # Return high-quality response
            if "fibonacci" in prompt.lower() and self.quality > 0.7:
                return self.responses["code_generation"]["improved"]
            return response
        else:
            # Return lower-quality response
            if "fibonacci" in prompt.lower():
                return self.responses["code_generation"]["initial"]
            return f"Low quality response: {response[:30]}..."
        
        # Improve quality over time
        self.quality = min(1.0, self.quality + self.improvement_rate)


class FailingLLM(MockLLM):
    """Mock LLM that fails randomly to test error handling."""
    
    def __init__(self, failure_rate: float = 0.3):
        super().__init__()
        self.failure_rate = failure_rate
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with random failures."""
        if random.random() < self.failure_rate:
            raise Exception(f"Mock LLM failure for prompt: {prompt[:30]}...")
        
        return super().generate(prompt, **kwargs)


class DelayedLLM(MockLLM):
    """Mock LLM that simulates network delays."""
    
    def __init__(self, delay_seconds: float = 1.0):
        super().__init__()
        self.delay_seconds = delay_seconds
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with simulated delay."""
        import time
        time.sleep(self.delay_seconds)
        return super().generate(prompt, **kwargs)


class ConfigurableLLM(MockLLM):
    """Mock LLM with configurable behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.response_templates = config.get("response_templates", {})
        self.should_fail = config.get("should_fail", False)
        self.failure_message = config.get("failure_message", "Configured failure")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response based on configuration."""
        if self.should_fail:
            raise Exception(self.failure_message)
            
        # Use configured response templates
        for keyword, template in self.response_templates.items():
            if keyword.lower() in prompt.lower():
                return template.format(prompt=prompt, **kwargs)
                
        return super().generate(prompt, **kwargs)


def create_mock_llm(llm_type: str = "basic", **kwargs) -> MockLLM:
    """Factory function to create different types of mock LLMs."""
    llm_types = {
        "basic": MockLLM,
        "progressive": ProgressiveLLM,
        "failing": FailingLLM,
        "delayed": DelayedLLM,
        "configurable": ConfigurableLLM
    }
    
    llm_class = llm_types.get(llm_type, MockLLM)
    return llm_class(**kwargs)


def create_mock_llm_with_responses(responses: Dict[str, str]) -> MockLLM:
    """Create a mock LLM with specific responses."""
    return MockLLM(responses={"custom": responses})


# Predefined mock LLMs for common test scenarios
MOCK_LLMS = {
    "always_succeeds": MockLLM({
        "code_generation": {
            "initial": "def perfect_function(): return 'perfect'",
            "improved": "def perfect_function(): return 'perfect'"
        },
        "reflection": {
            "self_critique": "This code is already perfect.",
            "improvement_plan": "No improvements needed."
        }
    }),
    
    "always_fails": FailingLLM(failure_rate=1.0),
    
    "improves_quickly": ProgressiveLLM(initial_quality=0.1, improvement_rate=0.3),
    
    "slow_responder": DelayedLLM(delay_seconds=2.0)
}
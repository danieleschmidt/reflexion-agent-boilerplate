"""
Reflexion Agent Boilerplate

Production-ready implementation of Reflexion for language agents.
"""

__version__ = "0.1.0"
__author__ = "Your Organization"

from .core.agent import ReflexionAgent
from .core.optimized_agent import OptimizedReflexionAgent, AutoScalingReflexionAgent
from .core.types import ReflectionType, ReflexionResult
from .prompts import ReflectionPrompts, PromptDomain, CustomReflectionPrompts
from .memory import EpisodicMemory, MemoryStore

__all__ = [
    "ReflexionAgent", 
    "OptimizedReflexionAgent", 
    "AutoScalingReflexionAgent",
    "ReflectionType", 
    "ReflexionResult",
    "ReflectionPrompts",
    "PromptDomain", 
    "CustomReflectionPrompts",
    "EpisodicMemory",
    "MemoryStore"
]
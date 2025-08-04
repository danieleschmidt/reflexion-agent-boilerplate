"""
Reflexion Agent Boilerplate

Production-ready implementation of Reflexion for language agents.
"""

__version__ = "0.1.0"
__author__ = "Your Organization"

from .core.agent import ReflexionAgent
from .core.optimized_agent import OptimizedReflexionAgent, AutoScalingReflexionAgent
from .core.types import ReflectionType, ReflexionResult

__all__ = [
    "ReflexionAgent", 
    "OptimizedReflexionAgent", 
    "AutoScalingReflexionAgent",
    "ReflectionType", 
    "ReflexionResult"
]
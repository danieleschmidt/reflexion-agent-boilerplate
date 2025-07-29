"""Core type definitions for the reflexion system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ReflectionType(Enum):
    """Types of reflection strategies."""
    BINARY = "binary"
    SCALAR = "scalar"
    STRUCTURED = "structured"


@dataclass
class Reflection:
    """Represents a single reflection on task execution."""
    task: str
    output: str
    success: bool
    score: float
    issues: List[str]
    improvements: List[str]
    confidence: float
    timestamp: str


@dataclass
class ReflexionResult:
    """Result of reflexion-enhanced task execution."""
    task: str
    output: str
    success: bool
    iterations: int
    reflections: List[Reflection]
    total_time: float
    metadata: Dict[str, Any]
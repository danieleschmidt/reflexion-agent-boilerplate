# ADR 001: Framework-Agnostic Design

## Status
Accepted

## Context
Need to support multiple AI agent frameworks (AutoGen, CrewAI, LangChain, Claude-Flow) with a single reflexion implementation.

## Decision
Implement an adapter pattern where each framework has a specific adapter that translates framework-specific concepts to our universal reflexion interface.

## Consequences
**Positive:**
- Single codebase for reflexion logic
- Easy to add new framework support
- Framework-specific optimizations possible

**Negative:**
- Additional abstraction layer
- Need to maintain multiple adapters
- Framework updates may break adapters

## Implementation
- Core `ReflexionAgent` class with framework-agnostic interface
- Framework-specific adapters in `reflexion.adapters.*`
- Common interface for task execution, evaluation, and reflection
# API Reference

## Core Classes

### ReflexionAgent

Base agent implementing self-reflection patterns.

```python
class ReflexionAgent:
    def __init__(
        self,
        llm: str,
        max_iterations: int = 3,
        reflection_type: ReflectionType = ReflectionType.BINARY,
        success_threshold: float = 0.8,
        **kwargs
    )
```

**Parameters:**
- `llm`: LLM model identifier (e.g., "gpt-4")
- `max_iterations`: Maximum reflection iterations
- `reflection_type`: Type of reflection (BINARY, SCALAR, STRUCTURED)
- `success_threshold`: Threshold for considering task successful
- `**kwargs`: Additional configuration options

**Methods:**

#### run()
Execute task with reflexion.
```python
def run(
    self, 
    task: str, 
    success_criteria: Optional[str] = None, 
    **kwargs
) -> ReflexionResult
```

### OptimizedReflexionAgent

Enhanced agent with performance optimizations.

```python
class OptimizedReflexionAgent(ReflexionAgent):
    def __init__(
        self,
        llm: str,
        max_iterations: int = 3,
        reflection_type: ReflectionType = ReflectionType.BINARY,
        success_threshold: float = 0.8,
        enable_caching: bool = True,
        enable_parallel_execution: bool = True,
        enable_memoization: bool = True,
        enable_prefetching: bool = False,
        max_concurrent_tasks: int = 4,
        batch_size: int = 10,
        cache_size: int = 1000,
        **kwargs
    )
```

**Additional Parameters:**
- `enable_caching`: Enable smart caching for performance
- `enable_parallel_execution`: Enable parallel task execution
- `enable_memoization`: Enable result memoization
- `enable_prefetching`: Enable intelligent prefetching
- `max_concurrent_tasks`: Maximum concurrent tasks for batch processing
- `batch_size`: Batch size for processing multiple tasks
- `cache_size`: Maximum cache size for optimization

**Additional Methods:**

#### run_batch()
Execute multiple tasks with optimization.
```python
async def run_batch(
    self,
    tasks: List[str],
    success_criteria: Optional[str] = None,
    **kwargs
) -> List[ReflexionResult]
```

#### get_performance_stats()
Get comprehensive performance statistics.
```python
def get_performance_stats() -> Dict[str, Any]
```

#### get_optimization_recommendations()
Get recommendations for optimizing agent performance.
```python
def get_optimization_recommendations() -> Dict[str, Any]
```

#### optimize_for_throughput()
Optimize settings for maximum throughput.
```python
def optimize_for_throughput()
```

#### optimize_for_accuracy()
Optimize settings for maximum accuracy.
```python
def optimize_for_accuracy()
```

#### optimize_for_cost()
Optimize settings for cost efficiency.
```python
def optimize_for_cost()
```

### AutoScalingReflexionAgent

Auto-scaling agent that adapts to load.

```python
class AutoScalingReflexionAgent(OptimizedReflexionAgent):
    def __init__(self, *args, **kwargs)
```

**Additional Methods:**

#### run_with_autoscaling()
Run task with automatic scaling based on load.
```python
async def run_with_autoscaling(
    self,
    task: str,
    success_criteria: Optional[str] = None,
    **kwargs
) -> ReflexionResult
```

#### get_scaling_stats()
Get auto-scaling statistics.
```python
def get_scaling_stats() -> Dict[str, Any]
```

## Memory Systems

### EpisodicMemory

Long-term memory system for storing and recalling experiences.

```python
class EpisodicMemory:
    def __init__(
        self,
        storage_path: str = "./memory.json",
        max_episodes: int = 1000,
        similarity_threshold: float = 0.7
    )
```

**Methods:**

#### store_episode()
Store a new episode in memory.
```python
def store_episode(
    self,
    task: str,
    result: ReflexionResult,
    metadata: Dict[str, Any] = None
)
```

#### recall_similar()
Recall similar episodes based on task similarity.
```python
def recall_similar(
    self,
    task: str,
    k: int = 5,
    min_similarity: float = 0.5
) -> List[Episode]
```

#### get_success_patterns()
Analyze and return success patterns.
```python
def get_success_patterns() -> Dict[str, Any]
```

### MemoryStore

Centralized memory storage with multiple backends.

```python
class MemoryStore:
    def __init__(
        self,
        backend: str = "dict",
        max_size: int = 10000,
        ttl: int = 3600
    )
```

## Framework Adapters

### AutoGenReflexion

AutoGen framework integration with reflexion capabilities.

```python
class AutoGenReflexion:
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Dict[str, Any],
        max_self_iterations: int = 3,
        memory_window: int = 10,
        reflection_strategy: str = "balanced"
    )
```

**Methods:**

#### initiate_chat()
Initiate chat with reflexion capabilities.
```python
def initiate_chat(
    self,
    message: str,
    recipient: Optional[Any] = None,
    clear_history: bool = False
) -> str
```

#### get_reflection_summary()
Get summary of reflexion performance.
```python
def get_reflection_summary() -> Dict[str, Any]
```

### ReflexiveCrewMember

CrewAI integration with collective learning.

```python
class ReflexiveCrewMember:
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str = "",
        reflection_strategy: str = "balanced",
        share_learnings: bool = True,
        learn_from_crew_feedback: bool = True
    )
```

**Methods:**

#### execute_task()
Execute task with crew coordination.
```python
def execute_task(
    self,
    task_description: str,
    context: Optional[str] = None
) -> Dict[str, Any]
```

#### share_learnings()
Share learnings with crew members.
```python
def share_learnings() -> List[Dict[str, Any]]
```

#### receive_crew_feedback()
Receive feedback from crew members.
```python
def receive_crew_feedback(self, feedback: Dict[str, Any])
```

### ReflexionChain

LangChain integration with reflexion wrapper.

```python
class ReflexionChain:
    def __init__(
        self,
        base_chain,
        max_self_iterations: int = 3,
        enable_memory: bool = True,
        memory_window: int = 5
    )
```

## Prompt System

### ReflectionPrompts

Domain-specific prompt management.

```python
class ReflectionPrompts:
    def get_reflection_prompt(
        self,
        domain: PromptDomain,
        context: Optional[str] = None
    ) -> str
```

### PromptDomain

Available prompt domains.

```python
class PromptDomain(Enum):
    GENERAL = "general"
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    RESEARCH = "research"
```

## Optimization & Scaling

### SmartCache

LRU cache with TTL support.

```python
class SmartCache:
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600
    )
```

**Methods:**

#### get()
Retrieve cached value.
```python
def get(self, key: str) -> Optional[Any]
```

#### put()
Store value in cache.
```python
def put(
    self,
    key: str,
    value: Any,
    ttl: Optional[int] = None
)
```

#### get_stats()
Get cache statistics.
```python
def get_stats() -> Dict[str, Any]
```

### ParallelExecutor

Parallel task execution with load balancing.

```python
class ParallelExecutor:
    def __init__(
        self,
        max_workers: int = 4,
        batch_size: int = 10
    )
```

**Methods:**

#### execute_batch()
Execute tasks in parallel batches.
```python
async def execute_batch(
    self,
    tasks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]
```

## Resilience Patterns

### ResilienceManager

Comprehensive resilience management system.

```python
class ResilienceManager:
    def __init__(self, config: Optional[ResilienceConfig] = None)
```

**Methods:**

#### execute_with_resilience()
Execute operation with comprehensive resilience patterns.
```python
async def execute_with_resilience(
    self,
    operation: Callable,
    operation_name: str,
    patterns: Optional[List[ResiliencePattern]] = None,
    fallback: Optional[Callable] = None,
    **kwargs
) -> Any
```

#### batch_execute_resilient()
Execute multiple operations with resilience patterns.
```python
async def batch_execute_resilient(
    self,
    operations: List[Dict[str, Any]],
    max_concurrent: Optional[int] = None
) -> List[Dict[str, Any]]
```

#### get_resilience_metrics()
Get comprehensive resilience metrics.
```python
def get_resilience_metrics() -> Dict[str, Any]
```

### CircuitBreaker

Circuit breaker pattern implementation.

```python
class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    )
```

### RetryManager

Advanced retry mechanisms with multiple backoff strategies.

```python
class RetryManager:
    def __init__(self, default_config: Optional[RetryConfig] = None)
```

**Methods:**

#### execute_with_retry()
Execute operation with retry logic.
```python
async def execute_with_retry(
    self,
    operation: Callable,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any
```

## Health & Monitoring

### HealthChecker

System health monitoring.

```python
class HealthChecker:
    def get_system_metrics(self) -> Dict[str, Any]
    def is_healthy(self) -> bool
    def get_health_status(self) -> Dict[str, Any]
```

### PerformanceMonitor

Performance metrics collection.

```python
class PerformanceMonitor:
    def start_timing(self, operation_name: str) -> str
    def end_timing(self, timer_id: str) -> float
    def get_metrics(self) -> Dict[str, Any]
```

## Data Types

### ReflexionResult

Result object returned by reflexion operations.

```python
@dataclass
class ReflexionResult:
    task: str
    output: str
    success: bool
    iterations: int
    reflections: List[Reflection]
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Reflection

Individual reflection data.

```python
@dataclass
class Reflection:
    iteration: int
    issues: List[str]
    improvements: List[str]
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)
```

### ReflectionType

Available reflection types.

```python
class ReflectionType(Enum):
    BINARY = "binary"
    SCALAR = "scalar"
    STRUCTURED = "structured"
```

## Configuration

### Environment Variables

- `REFLEXION_ENV`: Environment (development/production)
- `REFLEXION_LOG_LEVEL`: Logging level
- `REFLEXION_CACHE_SIZE`: Cache size limit
- `REFLEXION_MAX_WORKERS`: Maximum concurrent workers
- `REFLEXION_HEALTH_CHECK_INTERVAL`: Health check frequency

### Production Configuration Example

```python
config = {
    'llm': 'gpt-4',
    'max_iterations': 3,
    'success_threshold': 0.8,
    'enable_caching': True,
    'enable_parallel_execution': True,
    'max_concurrent_tasks': 8,
    'cache_size': 2000,
    'memory_path': './production_memory.json',
    'max_episodes': 10000
}
```

## Error Handling

### Custom Exceptions

```python
class ReflexionError(Exception):
    """Base exception for reflexion operations."""
    pass

class TimeoutError(ReflexionError):
    """Timeout during operation execution."""
    pass

class ResourceExhaustedError(ReflexionError):
    """System resources exhausted."""
    pass
```

## Examples

### Basic Usage
```python
from reflexion import ReflexionAgent, ReflectionType

agent = ReflexionAgent(
    llm="gpt-4",
    max_iterations=3,
    reflection_type=ReflectionType.STRUCTURED
)

result = agent.run("Write a Python function to calculate fibonacci numbers")
print(f"Success: {result.success}, Iterations: {result.iterations}")
```

### Optimized Usage
```python
from reflexion import OptimizedReflexionAgent

agent = OptimizedReflexionAgent(
    llm="gpt-4",
    enable_caching=True,
    enable_parallel_execution=True,
    max_concurrent_tasks=4
)

# Batch processing
tasks = ["Task 1", "Task 2", "Task 3"]
results = await agent.run_batch(tasks)
```

### Production Usage
```python
from reflexion import AutoScalingReflexionAgent

agent = AutoScalingReflexionAgent(
    llm="gpt-4",
    min_workers=2,
    max_workers=16,
    scale_up_threshold=0.8
)

result = await agent.run_with_autoscaling("Complex production task")
```
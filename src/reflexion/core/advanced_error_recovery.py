"""
Advanced error recovery and self-healing mechanisms for reflexion agents.
"""

import asyncio
import time
import traceback
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque
import random

from .exceptions import ReflectionError, LLMError, ValidationError, TimeoutError, SecurityError
from .logging_config import logger


class ErrorSeverity(Enum):
    """Error severity levels for recovery prioritization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    SELF_HEALING = "self_healing"
    ADAPTIVE_TIMEOUT = "adaptive_timeout"
    LOAD_SHEDDING = "load_shedding"


@dataclass
class ErrorContext:
    """Comprehensive error context for intelligent recovery."""
    error: Exception
    severity: ErrorSeverity
    context: Dict[str, Any]
    timestamp: datetime
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    last_recovery_strategy: Optional[RecoveryStrategy] = None
    correlation_id: Optional[str] = None
    
    def is_recoverable(self) -> bool:
        """Determine if error is recoverable."""
        return (
            self.recovery_attempts < self.max_recovery_attempts and
            self.severity != ErrorSeverity.CRITICAL and
            not isinstance(self.error, SecurityError)
        )
    
    def get_error_signature(self) -> str:
        """Get unique signature for error pattern recognition."""
        error_type = type(self.error).__name__
        error_message = str(self.error)[:100]  # Truncate for signature
        return f"{error_type}:{hash(error_message) % 10000}"


@dataclass 
class RecoveryMetrics:
    """Metrics for tracking error recovery performance."""
    total_errors: int = 0
    recoverable_errors: int = 0
    successful_recoveries: int = 0
    recovery_time_total: float = 0.0
    recovery_strategies_used: Dict[RecoveryStrategy, int] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    
    def add_error(self, error_context: ErrorContext):
        """Record error occurrence."""
        self.total_errors += 1
        if error_context.is_recoverable():
            self.recoverable_errors += 1
        
        signature = error_context.get_error_signature()
        self.error_patterns[signature] = self.error_patterns.get(signature, 0) + 1
    
    def add_recovery(self, strategy: RecoveryStrategy, recovery_time: float, success: bool):
        """Record recovery attempt."""
        self.recovery_strategies_used[strategy] = self.recovery_strategies_used.get(strategy, 0) + 1
        self.recovery_time_total += recovery_time
        
        if success:
            self.successful_recoveries += 1
    
    def get_success_rate(self) -> float:
        """Calculate overall recovery success rate."""
        if self.recoverable_errors == 0:
            return 0.0
        return self.successful_recoveries / self.recoverable_errors
    
    def get_avg_recovery_time(self) -> float:
        """Calculate average recovery time."""
        total_attempts = sum(self.recovery_strategies_used.values())
        if total_attempts == 0:
            return 0.0
        return self.recovery_time_total / total_attempts


class CircuitBreaker:
    """Circuit breaker pattern for error recovery."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        timeout_duration: float = 60.0,
        recovery_timeout: float = 300.0
    ):
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.logger.info("Circuit breaker transitioning to half-open")
            else:
                raise Exception(f"Circuit breaker is open, blocking request")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                self.logger.info("Circuit breaker reset to closed")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout


class AdaptiveTimeout:
    """Adaptive timeout mechanism that learns from execution patterns."""
    
    def __init__(self, initial_timeout: float = 30.0, max_timeout: float = 300.0):
        self.initial_timeout = initial_timeout
        self.max_timeout = max_timeout
        self.current_timeout = initial_timeout
        
        self.execution_times = deque(maxlen=50)
        self.success_times = deque(maxlen=50)
        self.timeout_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive timeout."""
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else 
                asyncio.create_task(asyncio.coroutine(lambda: func(*args, **kwargs))()),
                timeout=self.current_timeout
            )
            
            execution_time = time.time() - start_time
            self.success_times.append(execution_time)
            self.execution_times.append(execution_time)
            
            # Adapt timeout based on recent success patterns
            self._adapt_timeout()
            
            return result
            
        except asyncio.TimeoutError:
            self.timeout_count += 1
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Increase timeout after timeout
            self.current_timeout = min(self.max_timeout, self.current_timeout * 1.5)
            self.logger.warning(f"Adaptive timeout increased to {self.current_timeout:.1f}s after timeout")
            
            raise TimeoutError(f"Operation timed out after {execution_time:.1f}s", "adaptive_timeout", self.current_timeout)
    
    def _adapt_timeout(self):
        """Adapt timeout based on execution history."""
        if len(self.success_times) < 5:
            return
        
        # Calculate statistics
        recent_times = list(self.success_times)[-20:]
        avg_time = sum(recent_times) / len(recent_times)
        max_time = max(recent_times)
        
        # Set timeout to cover 95% of successful executions
        new_timeout = max(self.initial_timeout, avg_time * 2.5, max_time * 1.2)
        new_timeout = min(self.max_timeout, new_timeout)
        
        if abs(new_timeout - self.current_timeout) > 5.0:  # Only update if significant change
            self.current_timeout = new_timeout
            self.logger.info(f"Adaptive timeout updated to {self.current_timeout:.1f}s")


class SelfHealingSystem:
    """Self-healing system that learns from failures and adapts."""
    
    def __init__(self):
        self.healing_strategies: Dict[str, Callable] = {}
        self.healing_history = deque(maxlen=100)
        self.pattern_recognition = defaultdict(list)
        self.healing_effectiveness = defaultdict(float)
        
        self.logger = logging.getLogger(__name__)
        
        # Register default healing strategies
        self._register_default_healers()
    
    def _register_default_healers(self):
        """Register default self-healing strategies."""
        
        self.healing_strategies.update({
            "memory_pressure": self._heal_memory_pressure,
            "connection_failure": self._heal_connection_failure,
            "resource_exhaustion": self._heal_resource_exhaustion,
            "timeout_cascade": self._heal_timeout_cascade,
            "data_corruption": self._heal_data_corruption
        })
    
    def register_healer(self, pattern: str, healer: Callable):
        """Register custom healing strategy."""
        self.healing_strategies[pattern] = healer
        self.logger.info(f"Registered self-healing strategy for pattern: {pattern}")
    
    async def attempt_healing(self, error_context: ErrorContext) -> bool:
        """Attempt to heal the system based on error context."""
        
        error_signature = error_context.get_error_signature()
        self.pattern_recognition[error_signature].append(error_context)
        
        # Identify healing pattern
        healing_pattern = self._identify_healing_pattern(error_context)
        
        if healing_pattern and healing_pattern in self.healing_strategies:
            try:
                start_time = time.time()
                
                healer = self.healing_strategies[healing_pattern]
                success = await healer(error_context) if asyncio.iscoroutinefunction(healer) else healer(error_context)
                
                healing_time = time.time() - start_time
                
                # Record healing attempt
                self.healing_history.append({
                    "pattern": healing_pattern,
                    "error_signature": error_signature,
                    "success": success,
                    "healing_time": healing_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update effectiveness
                if success:
                    self.healing_effectiveness[healing_pattern] = (
                        self.healing_effectiveness[healing_pattern] * 0.9 + 0.1
                    )
                    self.logger.info(f"Self-healing successful for pattern: {healing_pattern}")
                else:
                    self.healing_effectiveness[healing_pattern] *= 0.8
                    self.logger.warning(f"Self-healing failed for pattern: {healing_pattern}")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Self-healing strategy failed: {e}")
                return False
        
        return False
    
    def _identify_healing_pattern(self, error_context: ErrorContext) -> Optional[str]:
        """Identify applicable healing pattern for error."""
        
        error_message = str(error_context.error).lower()
        error_type = type(error_context.error).__name__
        
        # Pattern matching based on error characteristics
        if "memory" in error_message or "ram" in error_message:
            return "memory_pressure"
        elif "connection" in error_message or "network" in error_message:
            return "connection_failure"
        elif "timeout" in error_message or isinstance(error_context.error, TimeoutError):
            return "timeout_cascade"
        elif "resource" in error_message or "limit" in error_message:
            return "resource_exhaustion"
        elif "corrupt" in error_message or "invalid" in error_message:
            return "data_corruption"
        
        return None
    
    async def _heal_memory_pressure(self, error_context: ErrorContext) -> bool:
        """Heal memory pressure issues."""
        try:
            # Simulate memory cleanup
            import gc
            gc.collect()
            
            # Clear caches if available
            if hasattr(error_context.context, 'clear_caches'):
                error_context.context.clear_caches()
            
            self.logger.info("Memory pressure healing applied")
            return True
        except Exception:
            return False
    
    async def _heal_connection_failure(self, error_context: ErrorContext) -> bool:
        """Heal connection failures."""
        try:
            # Wait and retry with exponential backoff
            retry_delay = min(30.0, 1.0 * (2 ** error_context.recovery_attempts))
            await asyncio.sleep(retry_delay)
            
            self.logger.info(f"Connection healing applied with {retry_delay}s delay")
            return True
        except Exception:
            return False
    
    async def _heal_resource_exhaustion(self, error_context: ErrorContext) -> bool:
        """Heal resource exhaustion."""
        try:
            # Implement resource throttling
            await asyncio.sleep(random.uniform(1.0, 5.0))
            
            # Reduce load if possible
            if 'load_reducer' in error_context.context:
                error_context.context['load_reducer']()
            
            self.logger.info("Resource exhaustion healing applied")
            return True
        except Exception:
            return False
    
    async def _heal_timeout_cascade(self, error_context: ErrorContext) -> bool:
        """Heal timeout cascade failures."""
        try:
            # Progressive timeout increase
            if 'timeout_multiplier' in error_context.context:
                error_context.context['timeout_multiplier'] *= 1.5
            
            # Circuit breaker reset
            await asyncio.sleep(5.0)
            
            self.logger.info("Timeout cascade healing applied")
            return True
        except Exception:
            return False
    
    async def _heal_data_corruption(self, error_context: ErrorContext) -> bool:
        """Heal data corruption issues."""
        try:
            # Data validation and cleanup
            if 'data_validator' in error_context.context:
                error_context.context['data_validator']()
            
            # Fallback to safe defaults
            if 'safe_defaults' in error_context.context:
                error_context.context.update(error_context.context['safe_defaults'])
            
            self.logger.info("Data corruption healing applied")
            return True
        except Exception:
            return False
    
    def get_healing_report(self) -> Dict[str, Any]:
        """Get comprehensive self-healing report."""
        
        total_attempts = len(self.healing_history)
        successful_healings = sum(1 for h in self.healing_history if h["success"])
        
        pattern_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
        for healing in self.healing_history:
            pattern = healing["pattern"]
            pattern_stats[pattern]["attempts"] += 1
            if healing["success"]:
                pattern_stats[pattern]["successes"] += 1
        
        return {
            "overall_statistics": {
                "total_healing_attempts": total_attempts,
                "successful_healings": successful_healings,
                "healing_success_rate": successful_healings / max(1, total_attempts),
                "registered_healers": len(self.healing_strategies)
            },
            "pattern_effectiveness": {
                pattern: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": stats["successes"] / max(1, stats["attempts"]),
                    "effectiveness_score": self.healing_effectiveness.get(pattern, 0.0)
                }
                for pattern, stats in pattern_stats.items()
            },
            "recent_healing_activity": list(self.healing_history)[-10:],
            "most_common_patterns": sorted(
                pattern_stats.items(), 
                key=lambda x: x[1]["attempts"], 
                reverse=True
            )[:5]
        }


class AdvancedErrorRecoveryManager:
    """Advanced error recovery manager with multiple strategies and self-healing."""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.adaptive_timeout = AdaptiveTimeout()
        self.self_healing = SelfHealingSystem()
        
        self.recovery_metrics = RecoveryMetrics()
        self.error_history = deque(maxlen=1000)
        
        self.recovery_strategies = {
            RecoveryStrategy.RETRY: self._retry_strategy,
            RecoveryStrategy.FALLBACK: self._fallback_strategy,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation_strategy,
            RecoveryStrategy.CIRCUIT_BREAKER: self._circuit_breaker_strategy,
            RecoveryStrategy.BULKHEAD: self._bulkhead_strategy,
            RecoveryStrategy.SELF_HEALING: self._self_healing_strategy,
            RecoveryStrategy.ADAPTIVE_TIMEOUT: self._adaptive_timeout_strategy,
            RecoveryStrategy.LOAD_SHEDDING: self._load_shedding_strategy
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def handle_error(
        self,
        error: Exception,
        operation: Callable,
        context: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Any:
        """Handle error with intelligent recovery strategies."""
        
        # Create error context
        severity = self._assess_error_severity(error)
        error_context = ErrorContext(
            error=error,
            severity=severity,
            context=context,
            timestamp=datetime.now(),
            correlation_id=correlation_id
        )
        
        # Record error
        self.recovery_metrics.add_error(error_context)
        self.error_history.append(error_context)
        
        self.logger.error(f"Error occurred: {error} (Severity: {severity.value})")
        
        # Attempt recovery
        return await self._attempt_recovery(error_context, operation)
    
    def _assess_error_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity for recovery prioritization."""
        
        if isinstance(error, SecurityError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (LLMError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValidationError, ConnectionError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    async def _attempt_recovery(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Attempt recovery using appropriate strategy."""
        
        if not error_context.is_recoverable():
            self.logger.error("Error is not recoverable, propagating exception")
            raise error_context.error
        
        # Select recovery strategy
        strategy = self._select_recovery_strategy(error_context)
        error_context.recovery_attempts += 1
        error_context.last_recovery_strategy = strategy
        
        self.logger.info(f"Attempting recovery with strategy: {strategy.value}")
        
        start_time = time.time()
        try:
            result = await self.recovery_strategies[strategy](error_context, operation)
            recovery_time = time.time() - start_time
            
            self.recovery_metrics.add_recovery(strategy, recovery_time, True)
            self.logger.info(f"Recovery successful with {strategy.value} in {recovery_time:.2f}s")
            
            return result
            
        except Exception as e:
            recovery_time = time.time() - start_time
            self.recovery_metrics.add_recovery(strategy, recovery_time, False)
            
            self.logger.warning(f"Recovery failed with {strategy.value}: {e}")
            
            # Try next strategy if available
            if error_context.recovery_attempts < error_context.max_recovery_attempts:
                return await self._attempt_recovery(error_context, operation)
            else:
                raise error_context.error
    
    def _select_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on error context."""
        
        error_type = type(error_context.error).__name__
        
        # Strategy selection logic
        if error_context.recovery_attempts == 0:
            # First attempt - try gentle strategies
            if isinstance(error_context.error, TimeoutError):
                return RecoveryStrategy.ADAPTIVE_TIMEOUT
            elif isinstance(error_context.error, ConnectionError):
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.SELF_HEALING
        
        elif error_context.recovery_attempts == 1:
            # Second attempt - more aggressive strategies
            if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                return RecoveryStrategy.CIRCUIT_BREAKER
            else:
                return RecoveryStrategy.FALLBACK
        
        else:
            # Final attempts - degradation strategies
            if error_context.severity == ErrorSeverity.HIGH:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
            else:
                return RecoveryStrategy.LOAD_SHEDDING
    
    async def _retry_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Retry strategy with exponential backoff."""
        
        retry_delay = min(30.0, 1.0 * (2 ** error_context.recovery_attempts))
        await asyncio.sleep(retry_delay)
        
        return await operation()
    
    async def _fallback_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Fallback strategy using alternative implementation."""
        
        # Try to find fallback function in context
        if 'fallback_operation' in error_context.context:
            fallback_op = error_context.context['fallback_operation']
            return await fallback_op() if asyncio.iscoroutinefunction(fallback_op) else fallback_op()
        
        # Default fallback response
        return {
            "error": "Operation failed, fallback response provided",
            "fallback": True,
            "original_error": str(error_context.error)
        }
    
    async def _graceful_degradation_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Graceful degradation with reduced functionality."""
        
        # Provide degraded response
        return {
            "degraded": True,
            "message": "Service operating in degraded mode due to errors",
            "limited_functionality": True,
            "error_context": str(error_context.error)
        }
    
    async def _circuit_breaker_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Circuit breaker strategy."""
        
        return await self.circuit_breaker.call(operation)
    
    async def _bulkhead_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Bulkhead isolation strategy."""
        
        # Isolate operation in separate execution context
        try:
            return await asyncio.wait_for(operation(), timeout=30.0)
        except asyncio.TimeoutError:
            raise TimeoutError("Operation timed out in bulkhead isolation", "bulkhead", 30.0)
    
    async def _self_healing_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Self-healing strategy."""
        
        healing_successful = await self.self_healing.attempt_healing(error_context)
        
        if healing_successful:
            # Retry operation after healing
            return await operation()
        else:
            raise error_context.error
    
    async def _adaptive_timeout_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Adaptive timeout strategy."""
        
        return await self.adaptive_timeout.execute_with_timeout(operation)
    
    async def _load_shedding_strategy(self, error_context: ErrorContext, operation: Callable) -> Any:
        """Load shedding strategy."""
        
        # Simulate load shedding with random acceptance
        if random.random() < 0.5:  # 50% load shedding
            raise Exception("Request shed due to high load")
        
        return await operation()
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """Get comprehensive error recovery report."""
        
        circuit_breaker_stats = {
            "state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None
        }
        
        adaptive_timeout_stats = {
            "current_timeout": self.adaptive_timeout.current_timeout,
            "timeout_count": self.adaptive_timeout.timeout_count,
            "avg_execution_time": sum(self.adaptive_timeout.success_times) / max(1, len(self.adaptive_timeout.success_times))
        }
        
        return {
            "recovery_metrics": {
                "total_errors": self.recovery_metrics.total_errors,
                "recoverable_errors": self.recovery_metrics.recoverable_errors,
                "successful_recoveries": self.recovery_metrics.successful_recoveries,
                "recovery_success_rate": self.recovery_metrics.get_success_rate(),
                "avg_recovery_time": self.recovery_metrics.get_avg_recovery_time()
            },
            "strategy_usage": dict(self.recovery_metrics.recovery_strategies_used),
            "error_patterns": dict(self.recovery_metrics.error_patterns),
            "circuit_breaker": circuit_breaker_stats,
            "adaptive_timeout": adaptive_timeout_stats,
            "self_healing": self.self_healing.get_healing_report(),
            "recent_errors": [
                {
                    "error": str(ctx.error),
                    "severity": ctx.severity.value,
                    "recovery_attempts": ctx.recovery_attempts,
                    "timestamp": ctx.timestamp.isoformat()
                }
                for ctx in list(self.error_history)[-10:]
            ]
        }


# Global error recovery manager
error_recovery_manager = AdvancedErrorRecoveryManager()
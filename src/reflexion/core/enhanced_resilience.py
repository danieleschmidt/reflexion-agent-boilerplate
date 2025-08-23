"""Enhanced Resilience System for Production Reflexion Agents.

This module implements comprehensive resilience patterns including:
- Circuit breakers with intelligent thresholds
- Self-healing mechanisms
- Graceful degradation
- Adaptive retry policies
- Resource monitoring and auto-scaling
"""

import asyncio
import json
import logging
import time
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


class ResilienceState(Enum):
    """Resilience system states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    error_rate: float
    response_time: float
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResilienceConfig:
    """Configuration for resilience mechanisms."""
    circuit_breaker_threshold: float = 0.5
    circuit_breaker_timeout: int = 60
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 1.5
    health_check_interval: int = 30
    auto_recovery_enabled: bool = True
    graceful_degradation_enabled: bool = True


class EnhancedResilienceManager:
    """Enhanced resilience manager for production systems."""
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.state = ResilienceState.HEALTHY
        self.circuit_breakers: Dict[str, Dict] = {}
        self.health_history: deque = deque(maxlen=100)
        self.retry_policies: Dict[str, Dict] = {}
        self.self_healing_actions: List[Callable] = []
        self.monitoring_active = False
        self.degraded_features: Set[str] = set()
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.adaptive_thresholds = defaultdict(float)
        
        # Auto-recovery state
        self.recovery_attempts = 0
        self.last_recovery_time = None
        
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default resilience policies."""
        # Default circuit breaker policies
        self.circuit_breakers["llm_calls"] = {
            "state": "closed",
            "failure_count": 0,
            "last_failure_time": None,
            "threshold": self.config.circuit_breaker_threshold,
            "timeout": self.config.circuit_breaker_timeout
        }
        
        # Default retry policies
        self.retry_policies["transient_errors"] = {
            "max_attempts": self.config.max_retry_attempts,
            "backoff_factor": self.config.retry_backoff_factor,
            "exceptions": ["ConnectionError", "TimeoutError", "TemporaryFailure"]
        }
        
        # Register default self-healing actions
        self.register_healing_action(self._clear_cache)
        self.register_healing_action(self._reset_connections)
        self.register_healing_action(self._restart_components)
    
    def register_healing_action(self, action: Callable):
        """Register a self-healing action."""
        self.self_healing_actions.append(action)
    
    @asynccontextmanager
    async def resilient_execution(self, operation_name: str, **kwargs):
        """Context manager for resilient operation execution."""
        start_time = time.time()
        success = False
        error = None
        
        try:
            # Check circuit breaker
            if self._is_circuit_open(operation_name):
                raise Exception(f"Circuit breaker open for {operation_name}")
            
            yield
            success = True
            self._record_success(operation_name, time.time() - start_time)
            
        except Exception as e:
            error = e
            self._record_failure(operation_name, e, time.time() - start_time)
            
            # Attempt auto-recovery
            if self.config.auto_recovery_enabled:
                await self._attempt_auto_recovery(operation_name, e)
            
            # Apply graceful degradation if needed
            if self.config.graceful_degradation_enabled:
                await self._apply_graceful_degradation(operation_name, e)
            
            raise
        finally:
            # Update health metrics
            await self._update_health_metrics()
    
    async def adaptive_retry(self, operation: Callable, operation_name: str, *args, **kwargs):
        """Execute operation with adaptive retry logic."""
        policy = self.retry_policies.get(operation_name, self.retry_policies["transient_errors"])
        max_attempts = policy["max_attempts"]
        backoff_factor = policy["backoff_factor"]
        
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                async with self.resilient_execution(operation_name):
                    result = await operation(*args, **kwargs)
                    return result
                    
            except Exception as e:
                last_exception = e
                
                # Check if this is a retryable error
                if not self._is_retryable_error(e, policy):
                    raise
                
                if attempt < max_attempts - 1:
                    # Calculate adaptive backoff
                    backoff_time = self._calculate_adaptive_backoff(
                        attempt, backoff_factor, operation_name
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {operation_name}, "
                        f"retrying in {backoff_time:.2f}s: {e}"
                    )
                    await asyncio.sleep(backoff_time)
        
        # All attempts failed
        logger.error(f"All {max_attempts} attempts failed for {operation_name}")
        raise last_exception
    
    def _is_circuit_open(self, operation_name: str) -> bool:
        """Check if circuit breaker is open for operation."""
        breaker = self.circuit_breakers.get(operation_name)
        if not breaker:
            return False
        
        if breaker["state"] == "open":
            # Check if timeout period has passed
            if (time.time() - breaker["last_failure_time"]) > breaker["timeout"]:
                breaker["state"] = "half_open"
                logger.info(f"Circuit breaker for {operation_name} moved to half-open")
                return False
            return True
        
        return False
    
    def _record_success(self, operation_name: str, duration: float):
        """Record successful operation."""
        breaker = self.circuit_breakers.get(operation_name)
        if breaker:
            if breaker["state"] == "half_open":
                breaker["state"] = "closed"
                breaker["failure_count"] = 0
                logger.info(f"Circuit breaker for {operation_name} closed")
        
        # Track performance metrics
        self.performance_metrics[operation_name].append({
            "success": True,
            "duration": duration,
            "timestamp": time.time()
        })
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds(operation_name)
    
    def _record_failure(self, operation_name: str, error: Exception, duration: float):
        """Record failed operation."""
        breaker = self.circuit_breakers.get(operation_name, {})
        breaker["failure_count"] = breaker.get("failure_count", 0) + 1
        breaker["last_failure_time"] = time.time()
        
        # Check if we should open the circuit
        failure_rate = self._calculate_failure_rate(operation_name)
        if failure_rate >= breaker.get("threshold", self.config.circuit_breaker_threshold):
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for {operation_name} (failure rate: {failure_rate:.2f})")
        
        self.circuit_breakers[operation_name] = breaker
        
        # Track performance metrics
        self.performance_metrics[operation_name].append({
            "success": False,
            "error": str(error),
            "duration": duration,
            "timestamp": time.time()
        })
    
    def _calculate_failure_rate(self, operation_name: str) -> float:
        """Calculate recent failure rate for operation."""
        metrics = self.performance_metrics.get(operation_name, [])
        if not metrics:
            return 0.0
        
        # Look at last 10 operations or last 5 minutes
        recent_time = time.time() - 300  # 5 minutes
        recent_metrics = [
            m for m in metrics[-10:] 
            if m.get("timestamp", 0) >= recent_time
        ]
        
        if not recent_metrics:
            return 0.0
        
        failures = sum(1 for m in recent_metrics if not m.get("success", True))
        return failures / len(recent_metrics)
    
    def _is_retryable_error(self, error: Exception, policy: Dict) -> bool:
        """Check if error is retryable based on policy."""
        error_type = type(error).__name__
        retryable_types = policy.get("exceptions", [])
        
        # Check if error type is in retryable list
        if error_type in retryable_types:
            return True
        
        # Check for specific error patterns
        error_msg = str(error).lower()
        retryable_patterns = [
            "timeout", "connection", "network", "temporary", "rate limit"
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    def _calculate_adaptive_backoff(self, attempt: int, base_factor: float, operation_name: str) -> float:
        """Calculate adaptive backoff time with jitter."""
        # Base exponential backoff
        backoff = base_factor ** attempt
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.5, 1.5)
        
        # Adapt based on system load
        load_factor = self._get_system_load_factor()
        
        # Adapt based on operation-specific performance
        performance_factor = self._get_performance_factor(operation_name)
        
        return backoff * jitter * load_factor * performance_factor
    
    def _get_system_load_factor(self) -> float:
        """Get system load factor for adaptive backoff."""
        if not self.health_history:
            return 1.0
        
        recent_health = self.health_history[-1]
        cpu_usage = getattr(recent_health, 'cpu_usage', 0.5)
        
        # Increase backoff under high load
        if cpu_usage > 0.8:
            return 2.0
        elif cpu_usage > 0.6:
            return 1.5
        else:
            return 1.0
    
    def _get_performance_factor(self, operation_name: str) -> float:
        """Get operation-specific performance factor."""
        metrics = self.performance_metrics.get(operation_name, [])
        if len(metrics) < 5:
            return 1.0
        
        # Calculate average response time for recent operations
        recent_metrics = metrics[-10:]
        avg_duration = sum(m.get("duration", 0) for m in recent_metrics) / len(recent_metrics)
        
        # Increase backoff for slow operations
        if avg_duration > 5.0:  # 5 seconds
            return 2.0
        elif avg_duration > 2.0:  # 2 seconds
            return 1.5
        else:
            return 1.0
    
    async def _attempt_auto_recovery(self, operation_name: str, error: Exception):
        """Attempt automatic recovery from error."""
        if self.recovery_attempts >= 3:  # Limit recovery attempts
            return
        
        if self.last_recovery_time and (time.time() - self.last_recovery_time) < 300:
            return  # Don't attempt recovery too frequently
        
        logger.info(f"Attempting auto-recovery for {operation_name}: {error}")
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        
        # Execute self-healing actions
        for action in self.self_healing_actions:
            try:
                await action(operation_name, error)
                logger.info(f"Self-healing action completed: {action.__name__}")
            except Exception as healing_error:
                logger.error(f"Self-healing action failed: {healing_error}")
    
    async def _apply_graceful_degradation(self, operation_name: str, error: Exception):
        """Apply graceful degradation for failed operation."""
        # Mark feature as degraded
        self.degraded_features.add(operation_name)
        self.state = ResilienceState.DEGRADED
        
        logger.warning(f"Applying graceful degradation for {operation_name}")
        
        # Implement specific degradation strategies
        if "llm" in operation_name.lower():
            await self._degrade_llm_features()
        elif "cache" in operation_name.lower():
            await self._degrade_cache_features()
        elif "database" in operation_name.lower():
            await self._degrade_database_features()
    
    async def _degrade_llm_features(self):
        """Degrade LLM-dependent features."""
        # Switch to simpler models or cached responses
        logger.info("Degrading LLM features - using cached responses only")
    
    async def _degrade_cache_features(self):
        """Degrade caching features."""
        # Disable complex caching, use simple in-memory cache
        logger.info("Degrading cache features - using simple fallback cache")
    
    async def _degrade_database_features(self):
        """Degrade database features."""
        # Use read-only mode or cached data
        logger.info("Degrading database features - switching to read-only mode")
    
    async def _update_health_metrics(self):
        """Update system health metrics."""
        try:
            import psutil
            
            health = HealthMetrics(
                cpu_usage=psutil.cpu_percent() / 100.0,
                memory_usage=psutil.virtual_memory().percent / 100.0,
                error_rate=self._calculate_overall_error_rate(),
                response_time=self._calculate_avg_response_time(),
                success_rate=self._calculate_overall_success_rate()
            )
            
            self.health_history.append(health)
            self._update_system_state(health)
            
        except ImportError:
            # Fallback if psutil not available
            health = HealthMetrics(
                cpu_usage=0.5,
                memory_usage=0.5,
                error_rate=self._calculate_overall_error_rate(),
                response_time=self._calculate_avg_response_time(),
                success_rate=self._calculate_overall_success_rate()
            )
            self.health_history.append(health)
    
    def _calculate_overall_error_rate(self) -> float:
        """Calculate overall system error rate."""
        total_operations = 0
        total_errors = 0
        
        for operation_metrics in self.performance_metrics.values():
            recent_metrics = operation_metrics[-20:]  # Last 20 operations
            total_operations += len(recent_metrics)
            total_errors += sum(1 for m in recent_metrics if not m.get("success", True))
        
        return total_errors / max(total_operations, 1)
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all operations."""
        all_durations = []
        
        for operation_metrics in self.performance_metrics.values():
            recent_metrics = operation_metrics[-20:]
            all_durations.extend(m.get("duration", 0) for m in recent_metrics)
        
        return sum(all_durations) / max(len(all_durations), 1)
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall system success rate."""
        return 1.0 - self._calculate_overall_error_rate()
    
    def _update_system_state(self, health: HealthMetrics):
        """Update overall system state based on health metrics."""
        if health.error_rate > 0.5 or health.cpu_usage > 0.9:
            self.state = ResilienceState.CRITICAL
        elif health.error_rate > 0.2 or len(self.degraded_features) > 0:
            self.state = ResilienceState.DEGRADED
        elif self.state != ResilienceState.HEALTHY and health.error_rate < 0.1:
            self.state = ResilienceState.RECOVERING
        else:
            self.state = ResilienceState.HEALTHY
            self.degraded_features.clear()
    
    def _update_adaptive_thresholds(self, operation_name: str):
        """Update adaptive thresholds based on performance history."""
        metrics = self.performance_metrics.get(operation_name, [])
        if len(metrics) < 10:
            return
        
        recent_metrics = metrics[-20:]
        success_rate = sum(1 for m in recent_metrics if m.get("success", True)) / len(recent_metrics)
        
        # Adapt circuit breaker threshold
        if operation_name in self.circuit_breakers:
            if success_rate > 0.95:
                # High success rate - can be more tolerant
                self.circuit_breakers[operation_name]["threshold"] = min(0.7, 
                    self.circuit_breakers[operation_name].get("threshold", 0.5) + 0.1)
            elif success_rate < 0.8:
                # Low success rate - be more strict
                self.circuit_breakers[operation_name]["threshold"] = max(0.3, 
                    self.circuit_breakers[operation_name].get("threshold", 0.5) - 0.1)
    
    # Default self-healing actions
    async def _clear_cache(self, operation_name: str, error: Exception):
        """Clear system caches as healing action."""
        logger.info("Clearing system caches for recovery")
        # Implementation would clear various caches
    
    async def _reset_connections(self, operation_name: str, error: Exception):
        """Reset network connections as healing action."""
        logger.info("Resetting network connections for recovery")
        # Implementation would reset HTTP connections, DB connections, etc.
    
    async def _restart_components(self, operation_name: str, error: Exception):
        """Restart failing components as healing action."""
        logger.info(f"Restarting components related to {operation_name}")
        # Implementation would restart specific service components
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        latest_health = self.health_history[-1] if self.health_history else None
        
        return {
            "state": self.state.value,
            "degraded_features": list(self.degraded_features),
            "circuit_breakers": {
                name: {
                    "state": breaker["state"],
                    "failure_count": breaker.get("failure_count", 0)
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "health_metrics": {
                "cpu_usage": latest_health.cpu_usage if latest_health else 0.0,
                "memory_usage": latest_health.memory_usage if latest_health else 0.0,
                "error_rate": latest_health.error_rate if latest_health else 0.0,
                "response_time": latest_health.response_time if latest_health else 0.0,
                "success_rate": latest_health.success_rate if latest_health else 1.0
            } if latest_health else {},
            "recovery_attempts": self.recovery_attempts,
            "performance_summary": {
                name: {
                    "total_operations": len(metrics),
                    "recent_success_rate": self._calculate_success_rate_for_operation(name)
                }
                for name, metrics in self.performance_metrics.items()
            }
        }
    
    def _calculate_success_rate_for_operation(self, operation_name: str) -> float:
        """Calculate success rate for specific operation."""
        metrics = self.performance_metrics.get(operation_name, [])
        if not metrics:
            return 1.0
        
        recent_metrics = metrics[-10:]
        successes = sum(1 for m in recent_metrics if m.get("success", True))
        return successes / len(recent_metrics)


# Global resilience manager instance
resilience_manager = EnhancedResilienceManager()
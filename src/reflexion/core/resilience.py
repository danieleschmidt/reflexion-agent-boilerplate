"""Resilience patterns and fault tolerance for reflexion systems."""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .exceptions import ReflexionError, TimeoutError, ResourceExhaustedError
from .retry import RetryManager, RetryConfig, RateLimiter
from .health import CircuitBreaker


class ResiliencePattern(Enum):
    """Available resilience patterns."""
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    RATE_LIMITING = "rate_limiting"
    LOAD_BALANCING = "load_balancing"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns."""
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    # Timeout settings
    operation_timeout: float = 30.0
    graceful_shutdown_timeout: float = 10.0
    
    # Rate limiting
    max_requests_per_minute: int = 100
    
    # Bulkhead settings
    max_concurrent_operations: int = 10
    thread_pool_size: int = 5
    
    # Load balancing
    health_check_interval: float = 30.0
    failed_endpoint_recovery_time: float = 300.0


@dataclass
class OperationMetrics:
    """Metrics for resilience monitoring."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    circuit_breaker_opens: int = 0
    timeouts: int = 0
    rate_limit_hits: int = 0
    average_response_time: float = 0.0
    last_updated: float = field(default_factory=time.time)


class ResilienceManager:
    """Comprehensive resilience management system."""
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            recovery_timeout=self.config.circuit_breaker_recovery_timeout
        )
        
        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute,
            time_window=60.0
        )
        
        self.retry_manager = RetryManager()
        
        # Bulkhead - separate thread pools for different operations
        self.thread_pools = {}
        self.semaphores = {}
        
        # Load balancer state
        self.endpoints = []
        self.endpoint_health = {}
        
        # Metrics
        self.metrics = OperationMetrics()
        self.operation_history: List[Dict[str, Any]] = []
        
        # Graceful degradation handlers
        self.degradation_handlers: Dict[str, Callable] = {}
    
    async def execute_with_resilience(
        self,
        operation: Callable,
        operation_name: str,
        patterns: Optional[List[ResiliencePattern]] = None,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """Execute operation with comprehensive resilience patterns."""
        patterns = patterns or [
            ResiliencePattern.CIRCUIT_BREAKER,
            ResiliencePattern.TIMEOUT,
            ResiliencePattern.RATE_LIMITING
        ]
        
        start_time = time.time()
        self.metrics.total_operations += 1
        
        try:
            # Apply rate limiting
            if ResiliencePattern.RATE_LIMITING in patterns:
                if not await self.rate_limiter.acquire(timeout=5.0):
                    self.metrics.rate_limit_hits += 1
                    raise ResourceExhaustedError(
                        "Rate limit exceeded",
                        resource_type="requests",
                        retry_count=0,
                        max_retries=1
                    )
            
            # Apply bulkhead pattern
            if ResiliencePattern.BULKHEAD in patterns:
                semaphore = self._get_semaphore(operation_name)
                async with semaphore:
                    result = await self._execute_with_timeout(operation, patterns, **kwargs)
            else:
                result = await self._execute_with_timeout(operation, patterns, **kwargs)
            
            # Success
            self.metrics.successful_operations += 1
            execution_time = time.time() - start_time
            self._update_response_time(execution_time)
            
            self.operation_history.append({
                "operation": operation_name,
                "success": True,
                "duration": execution_time,
                "patterns_used": [p.value for p in patterns],
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            self.metrics.failed_operations += 1
            execution_time = time.time() - start_time
            
            self.operation_history.append({
                "operation": operation_name,
                "success": False,
                "error": str(e),
                "duration": execution_time,
                "patterns_used": [p.value for p in patterns],
                "timestamp": time.time()
            })
            
            # Apply graceful degradation
            if ResiliencePattern.GRACEFUL_DEGRADATION in patterns and fallback:
                self.logger.warning(f"Operation {operation_name} failed, using fallback: {str(e)}")
                try:
                    return await self._execute_fallback(fallback, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback also failed: {str(fallback_error)}")
                    raise e
            else:
                raise e
    
    async def _execute_with_timeout(
        self,
        operation: Callable,
        patterns: List[ResiliencePattern],
        **kwargs
    ) -> Any:
        """Execute operation with timeout protection."""
        if ResiliencePattern.TIMEOUT in patterns:
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await asyncio.wait_for(
                        operation(**kwargs),
                        timeout=self.config.operation_timeout
                    )
                else:
                    # Run sync operation in thread pool
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: operation(**kwargs)),
                        timeout=self.config.operation_timeout
                    )
                return result
                
            except asyncio.TimeoutError:
                self.metrics.timeouts += 1
                raise TimeoutError(
                    f"Operation timed out after {self.config.operation_timeout}s",
                    "execute_operation",
                    self.config.operation_timeout
                )
        else:
            if asyncio.iscoroutinefunction(operation):
                return await operation(**kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: operation(**kwargs))
    
    async def _execute_fallback(self, fallback: Callable, **kwargs) -> Any:
        """Execute fallback operation."""
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: fallback(**kwargs))
    
    def _get_semaphore(self, operation_name: str) -> asyncio.Semaphore:
        """Get or create semaphore for operation bulkhead."""
        if operation_name not in self.semaphores:
            self.semaphores[operation_name] = asyncio.Semaphore(
                self.config.max_concurrent_operations
            )
        return self.semaphores[operation_name]
    
    def _get_thread_pool(self, pool_name: str) -> ThreadPoolExecutor:
        """Get or create thread pool for operation isolation."""
        if pool_name not in self.thread_pools:
            self.thread_pools[pool_name] = ThreadPoolExecutor(
                max_workers=self.config.thread_pool_size,
                thread_name_prefix=f"resilience-{pool_name}"
            )
        return self.thread_pools[pool_name]
    
    def _update_response_time(self, execution_time: float):
        """Update average response time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = execution_time
        else:
            self.metrics.average_response_time = (
                alpha * execution_time + 
                (1 - alpha) * self.metrics.average_response_time
            )
    
    def register_degradation_handler(self, operation_name: str, handler: Callable):
        """Register a graceful degradation handler for an operation."""
        self.degradation_handlers[operation_name] = handler
        self.logger.info(f"Registered degradation handler for {operation_name}")
    
    async def batch_execute_resilient(
        self,
        operations: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Execute multiple operations with resilience patterns applied."""
        max_concurrent = max_concurrent or self.config.max_concurrent_operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_operation(op_config: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                operation_name = op_config.get("name", "unknown")
                operation_func = op_config.get("function")
                patterns = op_config.get("patterns", [ResiliencePattern.CIRCUIT_BREAKER])
                fallback = op_config.get("fallback")
                kwargs = op_config.get("kwargs", {})
                
                try:
                    result = await self.execute_with_resilience(
                        operation_func,
                        operation_name,
                        patterns,
                        fallback,
                        **kwargs
                    )
                    return {
                        "operation": operation_name,
                        "success": True,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "operation": operation_name,
                        "success": False,
                        "error": str(e)
                    }
        
        # Execute all operations concurrently
        tasks = [execute_single_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in gathering
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "operation": operations[i].get("name", f"operation_{i}"),
                    "success": False,
                    "error": str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    def add_endpoint(self, endpoint_id: str, endpoint_url: str, weight: int = 1):
        """Add an endpoint for load balancing."""
        self.endpoints.append({
            "id": endpoint_id,
            "url": endpoint_url,
            "weight": weight,
            "healthy": True,
            "last_health_check": time.time()
        })
        self.endpoint_health[endpoint_id] = {
            "consecutive_failures": 0,
            "last_failure": None,
            "total_requests": 0,
            "successful_requests": 0
        }
        self.logger.info(f"Added endpoint {endpoint_id}: {endpoint_url}")
    
    def get_healthy_endpoint(self) -> Optional[Dict[str, Any]]:
        """Get a healthy endpoint using weighted round-robin."""
        healthy_endpoints = [ep for ep in self.endpoints if ep["healthy"]]
        
        if not healthy_endpoints:
            # Try to recover failed endpoints
            self._attempt_endpoint_recovery()
            healthy_endpoints = [ep for ep in self.endpoints if ep["healthy"]]
            
            if not healthy_endpoints:
                return None
        
        # Simple weighted selection (can be enhanced with proper round-robin)
        total_weight = sum(ep["weight"] for ep in healthy_endpoints)
        if total_weight == 0:
            return healthy_endpoints[0] if healthy_endpoints else None
        
        # For simplicity, return the endpoint with highest weight
        return max(healthy_endpoints, key=lambda ep: ep["weight"])
    
    def _attempt_endpoint_recovery(self):
        """Attempt to recover failed endpoints after recovery time."""
        current_time = time.time()
        recovery_threshold = self.config.failed_endpoint_recovery_time
        
        for endpoint in self.endpoints:
            if not endpoint["healthy"]:
                health_info = self.endpoint_health[endpoint["id"]]
                if (health_info["last_failure"] and 
                    current_time - health_info["last_failure"] > recovery_threshold):
                    endpoint["healthy"] = True
                    health_info["consecutive_failures"] = 0
                    self.logger.info(f"Attempting recovery for endpoint {endpoint['id']}")
    
    def mark_endpoint_failure(self, endpoint_id: str):
        """Mark an endpoint as failed."""
        health_info = self.endpoint_health.get(endpoint_id)
        if health_info:
            health_info["consecutive_failures"] += 1
            health_info["last_failure"] = time.time()
            
            # Mark as unhealthy after 3 consecutive failures
            if health_info["consecutive_failures"] >= 3:
                for endpoint in self.endpoints:
                    if endpoint["id"] == endpoint_id:
                        endpoint["healthy"] = False
                        self.logger.warning(f"Marked endpoint {endpoint_id} as unhealthy")
                        break
    
    def mark_endpoint_success(self, endpoint_id: str):
        """Mark an endpoint operation as successful."""
        health_info = self.endpoint_health.get(endpoint_id)
        if health_info:
            health_info["consecutive_failures"] = 0
            health_info["successful_requests"] += 1
            health_info["total_requests"] += 1
            
            # Ensure endpoint is marked as healthy
            for endpoint in self.endpoints:
                if endpoint["id"] == endpoint_id:
                    endpoint["healthy"] = True
                    break
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics."""
        total_ops = self.metrics.total_operations
        success_rate = self.metrics.successful_operations / total_ops if total_ops > 0 else 0
        
        # Recent performance (last 100 operations)
        recent_ops = self.operation_history[-100:]
        recent_success_rate = len([op for op in recent_ops if op["success"]]) / len(recent_ops) if recent_ops else 0
        
        # Endpoint health
        endpoint_stats = {}
        for endpoint in self.endpoints:
            health_info = self.endpoint_health[endpoint["id"]]
            total_reqs = health_info["total_requests"]
            success_reqs = health_info["successful_requests"]
            
            endpoint_stats[endpoint["id"]] = {
                "healthy": endpoint["healthy"],
                "success_rate": success_reqs / total_reqs if total_reqs > 0 else 0,
                "total_requests": total_reqs,
                "consecutive_failures": health_info["consecutive_failures"]
            }
        
        return {
            "operations": {
                "total": self.metrics.total_operations,
                "successful": self.metrics.successful_operations,
                "failed": self.metrics.failed_operations,
                "success_rate": success_rate,
                "recent_success_rate": recent_success_rate,
                "average_response_time": self.metrics.average_response_time
            },
            "resilience_patterns": {
                "circuit_breaker_opens": self.metrics.circuit_breaker_opens,
                "timeouts": self.metrics.timeouts,
                "rate_limit_hits": self.metrics.rate_limit_hits
            },
            "endpoints": endpoint_stats,
            "rate_limiter": self.rate_limiter.get_stats(),
            "semaphores": {
                name: sem._value for name, sem in self.semaphores.items()
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown resilience manager."""
        self.logger.info("Initiating graceful shutdown of resilience manager")
        
        # Shutdown thread pools
        for name, pool in self.thread_pools.items():
            self.logger.info(f"Shutting down thread pool: {name}")
            pool.shutdown(wait=True, timeout=self.config.graceful_shutdown_timeout)
        
        self.logger.info("Resilience manager shutdown completed")


# Decorator for applying resilience patterns
def resilient(
    patterns: Optional[List[ResiliencePattern]] = None,
    fallback: Optional[Callable] = None,
    operation_name: Optional[str] = None
):
    """Decorator to apply resilience patterns to a function."""
    
    def decorator(func):
        resilience_manager = ResilienceManager()
        func_name = operation_name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await resilience_manager.execute_with_resilience(
                    func,
                    func_name,
                    patterns,
                    fallback,
                    *args,
                    **kwargs
                )
            return async_wrapper
        else:
            async def sync_wrapper(*args, **kwargs):
                return await resilience_manager.execute_with_resilience(
                    func,
                    func_name,
                    patterns,
                    fallback,
                    *args,
                    **kwargs
                )
            return sync_wrapper
    
    return decorator


# Global resilience manager instance
resilience_manager = ResilienceManager()
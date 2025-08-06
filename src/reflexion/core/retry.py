"""Advanced retry mechanisms with backoff strategies."""

import asyncio
import random
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

from .exceptions import RetryableError, RateLimitError, ResourceExhaustedError


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True
    retryable_exceptions: tuple = (RetryableError, RateLimitError, ResourceExhaustedError)
    retry_on_result: Optional[Callable[[Any], bool]] = None


class RetryableException(Exception):
    """Base exception for operations that should be retried."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class RetryManager:
    """Advanced retry management with multiple backoff strategies."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
        self.attempt_history: List[Dict[str, Any]] = []
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay for retry attempt based on backoff strategy."""
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = base_delay
            
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = base_delay * attempt
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** (attempt - 1))
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            exponential_delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, exponential_delay * 0.1)
            delay = exponential_delay + jitter
            
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = base_delay * self._fibonacci(attempt)
            
        else:
            delay = base_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.backoff_strategy != BackoffStrategy.EXPONENTIAL_JITTER:
            jitter_amount = delay * 0.1 * random.random()
            delay = delay + jitter_amount
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 2:
            return 1
        
        a, b = 1, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b
    
    async def retry_async(
        self,
        func: Callable,
        *args,
        config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with async retry logic."""
        retry_config = config or self.config
        last_exception = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                # Record attempt
                attempt_start = time.time()
                
                result = await func(*args, **kwargs)
                
                # Check if result should trigger retry
                if retry_config.retry_on_result and retry_config.retry_on_result(result):
                    if attempt < retry_config.max_attempts:
                        self.logger.warning(f"Result-based retry attempt {attempt}")
                        await self._delay_retry(attempt, retry_config)
                        continue
                    else:
                        self.logger.error("Max attempts reached with unsatisfactory result")
                        return result
                
                # Success - record and return
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": True,
                    "duration": time.time() - attempt_start,
                    "timestamp": time.time()
                })
                
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt}")
                
                return result
                
            except retry_config.retryable_exceptions as e:
                last_exception = e
                
                # Record failed attempt
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - attempt_start,
                    "timestamp": time.time()
                })
                
                if attempt < retry_config.max_attempts:
                    delay = self.calculate_delay(attempt, retry_config.base_delay)
                    
                    # Handle rate limiting with specific delay
                    if isinstance(e, RateLimitError) and hasattr(e, 'retry_after') and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {retry_config.max_attempts} attempts failed")
                    break
            
            except Exception as e:
                # Non-retryable exception
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "non_retryable": True,
                    "duration": time.time() - attempt_start,
                    "timestamp": time.time()
                })
                
                self.logger.error(f"Non-retryable exception on attempt {attempt}: {str(e)}")
                raise e
        
        # All retries exhausted
        if last_exception:
            raise last_exception
    
    def retry_sync(
        self,
        func: Callable,
        *args,
        config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with synchronous retry logic."""
        retry_config = config or self.config
        last_exception = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                attempt_start = time.time()
                
                result = func(*args, **kwargs)
                
                # Check if result should trigger retry
                if retry_config.retry_on_result and retry_config.retry_on_result(result):
                    if attempt < retry_config.max_attempts:
                        self.logger.warning(f"Result-based retry attempt {attempt}")
                        delay = self.calculate_delay(attempt, retry_config.base_delay)
                        time.sleep(delay)
                        continue
                    else:
                        return result
                
                # Success
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": True,
                    "duration": time.time() - attempt_start,
                    "timestamp": time.time()
                })
                
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt}")
                
                return result
                
            except retry_config.retryable_exceptions as e:
                last_exception = e
                
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - attempt_start,
                    "timestamp": time.time()
                })
                
                if attempt < retry_config.max_attempts:
                    delay = self.calculate_delay(attempt, retry_config.base_delay)
                    
                    if isinstance(e, RateLimitError) and hasattr(e, 'retry_after') and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f}s"
                    )
                    
                    time.sleep(delay)
                else:
                    break
                    
            except Exception as e:
                # Non-retryable exception
                self.attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "non_retryable": True,
                    "duration": time.time() - attempt_start,
                    "timestamp": time.time()
                })
                
                self.logger.error(f"Non-retryable exception: {str(e)}")
                raise e
        
        if last_exception:
            raise last_exception
    
    async def _delay_retry(self, attempt: int, config: RetryConfig):
        """Apply delay before retry."""
        delay = self.calculate_delay(attempt, config.base_delay)
        await asyncio.sleep(delay)
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get statistics about retry attempts."""
        if not self.attempt_history:
            return {"total_attempts": 0}
        
        total_attempts = len(self.attempt_history)
        successful_attempts = len([h for h in self.attempt_history if h["success"]])
        failed_attempts = total_attempts - successful_attempts
        
        # Calculate average duration
        durations = [h["duration"] for h in self.attempt_history]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Recent attempt success rate (last 100)
        recent_attempts = self.attempt_history[-100:]
        recent_success_rate = len([h for h in recent_attempts if h["success"]]) / len(recent_attempts)
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "failed_attempts": failed_attempts,
            "success_rate": successful_attempts / total_attempts,
            "recent_success_rate": recent_success_rate,
            "average_duration": avg_duration,
            "config": {
                "max_attempts": self.config.max_attempts,
                "backoff_strategy": self.config.backoff_strategy.value,
                "base_delay": self.config.base_delay,
                "max_delay": self.config.max_delay
            }
        }
    
    def clear_history(self):
        """Clear retry attempt history."""
        self.attempt_history.clear()


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for automatic retry with configuration."""
    
    def decorator(func):
        retry_manager = RetryManager(config)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await retry_manager.retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return retry_manager.retry_sync(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


class RateLimiter:
    """Rate limiter to prevent overwhelming services."""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire rate limit permission."""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            # Calculate wait time
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request)
            
            if timeout and wait_time > timeout:
                return False
            
            # Wait for rate limit to reset
            await asyncio.sleep(wait_time)
            
            # Try again after waiting
            self.requests = [req_time for req_time in self.requests if time.time() - req_time < self.time_window]
            if len(self.requests) < self.max_requests:
                self.requests.append(time.time())
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        active_requests = len([req for req in self.requests if now - req < self.time_window])
        
        return {
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "active_requests": active_requests,
            "available_slots": self.max_requests - active_requests,
            "utilization": active_requests / self.max_requests
        }


# Global retry manager instance
default_retry_manager = RetryManager()
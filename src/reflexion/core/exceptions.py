"""Custom exceptions for reflexion agents."""

from typing import Any, Dict, List, Optional


class ReflexionError(Exception):
    """Base exception for reflexion-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(ReflexionError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, validation_errors: List[str], warnings: Optional[List[str]] = None):
        super().__init__(message)
        self.validation_errors = validation_errors
        self.warnings = warnings or []


class LLMError(ReflexionError):
    """Raised when LLM operations fail."""
    
    def __init__(self, message: str, llm_model: str, operation: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.llm_model = llm_model
        self.operation = operation


class ReflectionError(ReflexionError):
    """Raised when reflection generation fails."""
    
    def __init__(self, message: str, task: str, iteration: int, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.task = task
        self.iteration = iteration


class MemoryError(ReflexionError):
    """Raised when memory operations fail."""
    
    def __init__(self, message: str, operation: str, backend: str, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.operation = operation
        self.backend = backend


class ConfigurationError(ReflexionError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str, config_value: Any):
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value


class SecurityError(ReflexionError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, security_issue: str, input_data: str):
        super().__init__(message)
        self.security_issue = security_issue
        self.input_data = input_data[:100]  # Truncate for safety


class TimeoutError(ReflexionError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, operation: str, timeout_seconds: float):
        super().__init__(message)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class RetryableError(ReflexionError):
    """Base class for errors that can be retried."""
    
    def __init__(self, message: str, retry_count: int, max_retries: int, details: Optional[Dict] = None):
        super().__init__(message, details)
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.can_retry = retry_count < max_retries


class RateLimitError(RetryableError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ResourceExhaustedError(RetryableError):
    """Raised when resources are exhausted."""
    
    def __init__(self, message: str, resource_type: str, **kwargs):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type


class ComplianceError(ReflexionError):
    """Raised when compliance violations are detected."""
    
    def __init__(self, message: str, compliance_type: str, violation_details: Optional[Dict] = None):
        super().__init__(message, violation_details)
        self.compliance_type = compliance_type
        self.violation_details = violation_details or {}
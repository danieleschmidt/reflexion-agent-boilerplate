"""
Advanced Validation and Security Module for Reflexion Agents.

Provides comprehensive input validation, output sanitization, and security checks.
"""

import re
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .exceptions import SecurityError, ValidationError


class SecurityLevel(Enum):
    """Security levels for validation."""
    LOW = 1
    MEDIUM = 2  
    HIGH = 3
    CRITICAL = 4
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    sanitized_input: str
    errors: List[str]
    warnings: List[str]
    security_score: float
    risk_level: SecurityLevel


class SecurityValidator:
    """Advanced security validation for reflexion inputs and outputs."""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.code_injection_patterns = [
            r'__import__\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\(\s*["\'][/\\]',  # File system access
            r'\.\./',  # Directory traversal
            r'rm\s+-rf',
            r'del\s+/\w+',
        ]
        
        self.sql_injection_patterns = [
            r';\s*drop\s+table',
            r'union\s+select',
            r'or\s+1\s*=\s*1',
            r'\';\s*--',
            r'xp_cmdshell',
        ]
        
        self.prompt_injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'disregard\s+system\s+prompt',
            r'new\s+instructions:',
            r'override\s+security',
        ]
        
        # Suspicious keywords (adjusted for legitimate development tasks)
        self.suspicious_keywords = {
            'critical': ['delete_all_files', 'rm -rf /', 'format_hard_drive'],
            'high': ['sudo rm', 'privilege escalation', 'backdoor'],
            'medium': ['admin_override', 'bypass_security', 'disable_firewall']
        }
    
    def validate_task_input(self, task: str) -> ValidationResult:
        """Comprehensive validation of task input."""
        errors = []
        warnings = []
        security_score = 1.0
        risk_level = SecurityLevel.LOW
        
        # Basic validation
        if not task or not task.strip():
            errors.append("Task cannot be empty")
            return ValidationResult(False, "", errors, warnings, 0.0, SecurityLevel.CRITICAL)
        
        if len(task) > 10000:
            errors.append("Task too long (>10000 characters)")
            security_score -= 0.3
        
        # Security pattern detection
        security_issues = self._detect_security_patterns(task)
        
        if security_issues['critical']:
            errors.extend(security_issues['critical'])
            risk_level = SecurityLevel.CRITICAL
            security_score = 0.0
        
        if security_issues['high']:
            errors.extend(security_issues['high'])
            risk_level = max(risk_level, SecurityLevel.HIGH)
            security_score -= 0.5
        
        if security_issues['medium']:
            warnings.extend(security_issues['medium'])
            risk_level = max(risk_level, SecurityLevel.MEDIUM)
            security_score -= 0.2
        
        # Content analysis
        content_issues = self._analyze_content(task)
        warnings.extend(content_issues)
        
        # Sanitize input
        sanitized = self._sanitize_input(task)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            errors=errors,
            warnings=warnings,
            security_score=max(0.0, security_score),
            risk_level=risk_level
        )
    
    def _detect_security_patterns(self, text: str) -> Dict[str, List[str]]:
        """Detect security-related patterns in text."""
        issues = {'critical': [], 'high': [], 'medium': []}
        text_lower = text.lower()
        
        # Code injection detection
        for pattern in self.code_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues['critical'].append(f"Potential code injection detected: {pattern}")
        
        # SQL injection detection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues['high'].append(f"Potential SQL injection detected: {pattern}")
        
        # Prompt injection detection
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues['high'].append(f"Potential prompt injection detected: {pattern}")
        
        # Suspicious keyword detection
        for level, keywords in self.suspicious_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    issues[level].append(f"Suspicious keyword detected: {keyword}")
        
        return issues
    
    def _analyze_content(self, text: str) -> List[str]:
        """Analyze content for potential issues."""
        warnings = []
        
        # Check for extremely long lines
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 500:
                warnings.append(f"Very long line detected at line {i+1}")
        
        # Check for suspicious file extensions
        suspicious_extensions = ['.exe', '.bat', '.sh', '.ps1', '.dll']
        for ext in suspicious_extensions:
            if ext in text.lower():
                warnings.append(f"Suspicious file extension mentioned: {ext}")
        
        # Check for network-related content
        network_patterns = [
            r'https?://[^\s]+',
            r'ftp://[^\s]+', 
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, text):
                warnings.append("Network-related content detected")
                break
        
        return warnings
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        # Remove null bytes
        sanitized = text.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in sanitized 
                          if ord(char) >= 32 or char in '\n\t')
        
        # Limit length
        if len(sanitized) > 8000:
            sanitized = sanitized[:8000] + "... [truncated]"
        
        return sanitized.strip()


class OutputSanitizer:
    """Sanitize and validate output from LLM responses."""
    
    def __init__(self):
        self.allowed_code_keywords = {
            'python': ['def', 'class', 'import', 'from', 'return', 'if', 'for', 'while', 'try', 'except'],
            'sql': ['select', 'insert', 'update', 'create', 'alter', 'index'],
            'general': ['function', 'method', 'variable', 'parameter']
        }
    
    def sanitize_output(self, output: str) -> str:
        """Sanitize LLM output for safe consumption."""
        if not output:
            return ""
        
        # Remove potential executable content markers
        dangerous_markers = [
            '```bash',
            '```sh', 
            '```powershell',
            '```cmd'
        ]
        
        sanitized = output
        for marker in dangerous_markers:
            if marker in sanitized:
                # Replace with safe alternative
                sanitized = sanitized.replace(marker, '```text')
        
        # Remove or escape HTML/JS if present
        sanitized = self._escape_html(sanitized)
        
        # Validate code blocks
        sanitized = self._validate_code_blocks(sanitized)
        
        return sanitized
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters."""
        html_escapes = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#x27;'
        }
        
        for char, escape in html_escapes.items():
            text = text.replace(char, escape)
        
        return text
    
    def _validate_code_blocks(self, text: str) -> str:
        """Validate and sanitize code blocks."""
        # Pattern to match code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        
        def validate_code_block(match):
            language = match.group(1) or 'text'
            code = match.group(2)
            
            # Check for dangerous patterns in code
            if self._contains_dangerous_code(code):
                return f"```{language}\n# Code block removed for security reasons\n```"
            
            return match.group(0)  # Return original if safe
        
        return re.sub(code_block_pattern, validate_code_block, text, flags=re.DOTALL)
    
    def _contains_dangerous_code(self, code: str) -> bool:
        """Check if code contains dangerous patterns."""
        dangerous_patterns = [
            r'rm\s+-rf',
            r'del\s+/\w+',
            r'format\s+c:',
            r'subprocess\.call',
            r'os\.system',
            r'__import__.*os',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        
        return False


class RateLimiter:
    """Rate limiting for task execution."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}  # user_id -> [(timestamp, hash), ...]
    
    def is_allowed(self, user_id: str, request_data: str) -> Tuple[bool, str]:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        request_hash = hashlib.md5(request_data.encode()).hexdigest()
        
        # Initialize user if not exists
        if user_id not in self.requests:
            self.requests[user_id] = []
        
        # Clean old requests
        self.requests[user_id] = [
            (ts, h) for ts, h in self.requests[user_id]
            if current_time - ts < self.time_window
        ]
        
        # Check rate limit
        if len(self.requests[user_id]) >= self.max_requests:
            return False, f"Rate limit exceeded. Max {self.max_requests} requests per {self.time_window}s"
        
        # Check for duplicate requests (potential spam)
        recent_hashes = [h for ts, h in self.requests[user_id] if current_time - ts < 300]  # 5 minutes
        if recent_hashes.count(request_hash) > 3:
            return False, "Duplicate request detected. Please wait before retrying."
        
        # Record request
        self.requests[user_id].append((current_time, request_hash))
        
        return True, ""


class ComprehensiveValidator:
    """Main validator combining all validation components."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.security_validator = SecurityValidator()
        self.output_sanitizer = OutputSanitizer()
        self.rate_limiter = RateLimiter()
    
    def validate_task(self, task: str, user_id: str = "default") -> ValidationResult:
        """Comprehensive task validation."""
        # Rate limiting check
        allowed, message = self.rate_limiter.is_allowed(user_id, task)
        if not allowed:
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                errors=[f"Rate limit: {message}"],
                warnings=[],
                security_score=0.0,
                risk_level=SecurityLevel.CRITICAL
            )
        
        # Security validation
        result = self.security_validator.validate_task_input(task)
        
        # Apply security level restrictions
        if self.security_level == SecurityLevel.HIGH and result.risk_level != SecurityLevel.LOW:
            result.is_valid = False
            result.errors.append("High security mode: only low-risk tasks allowed")
        
        elif self.security_level == SecurityLevel.CRITICAL and result.security_score < 0.9:
            result.is_valid = False
            result.errors.append("Critical security mode: extremely high standards required")
        
        return result
    
    def sanitize_output(self, output: str) -> str:
        """Sanitize output with security considerations."""
        return self.output_sanitizer.sanitize_output(output)
    
    def validate_llm_config(self, model: str) -> ValidationResult:
        """Validate LLM configuration."""
        errors = []
        warnings = []
        
        if not model:
            errors.append("Model name cannot be empty")
        
        # Check for valid model names
        valid_models = [
            'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo',
            'claude-3-sonnet', 'claude-3-opus', 'claude-3-haiku',
            'llama2', 'llama3', 'mistral', 'codellama'
        ]
        
        if model not in valid_models and not any(vm in model for vm in valid_models):
            warnings.append(f"Unknown model: {model}. Proceeding with caution.")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=model,
            errors=errors,
            warnings=warnings,
            security_score=0.8 if len(errors) == 0 else 0.0,
            risk_level=SecurityLevel.LOW
        )
    
    def validate_reflexion_params(self, task: str, llm: str, max_iterations: int, 
                                 reflection_type: Any, success_threshold: float) -> ValidationResult:
        """Validate reflexion execution parameters."""
        errors = []
        warnings = []
        
        # Validate task
        if not task or not task.strip():
            errors.append("Task cannot be empty")
        elif len(task) > 5000:
            warnings.append("Task is very long, consider breaking it down")
        
        # Validate LLM
        if not llm:
            errors.append("LLM model cannot be empty")
        
        # Validate iterations
        if max_iterations < 1:
            errors.append("Max iterations must be at least 1")
        elif max_iterations > 10:
            warnings.append("High iteration count may increase costs")
        
        # Validate threshold
        if not 0.0 <= success_threshold <= 1.0:
            errors.append("Success threshold must be between 0.0 and 1.0")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=task,
            errors=errors,
            warnings=warnings,
            security_score=0.9 if len(errors) == 0 else 0.0,
            risk_level=SecurityLevel.LOW if len(errors) == 0 else SecurityLevel.MEDIUM
        )


# Global validator instance
validator = ComprehensiveValidator()
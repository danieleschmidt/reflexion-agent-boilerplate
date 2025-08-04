"""Input validation and sanitization for reflexion agents."""

import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[Any] = None


class InputValidator:
    """Comprehensive input validation for reflexion systems."""
    
    def __init__(self):
        self.max_task_length = 10000
        self.max_iterations = 20
        self.min_success_threshold = 0.0
        self.max_success_threshold = 1.0
        
        # Security patterns to detect potentially malicious input
        self.security_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',  # JavaScript URLs
            r'vbscript:',  # VBScript URLs
            r'data:text/html',  # Data URLs with HTML
            r'eval\s*\(',  # eval() calls
            r'exec\s*\(',  # exec() calls
            r'import\s+os',  # OS imports
            r'__import__',  # Dynamic imports
            r'subprocess',  # Subprocess calls
        ]
    
    def validate_task(self, task: str) -> ValidationResult:
        """Validate task input."""
        errors = []
        warnings = []
        
        if not isinstance(task, str):
            errors.append("Task must be a string")
            return ValidationResult(False, errors, warnings)
        
        if not task.strip():
            errors.append("Task cannot be empty")
        
        if len(task) > self.max_task_length:
            errors.append(f"Task length ({len(task)}) exceeds maximum ({self.max_task_length})")
        
        # Security validation
        for pattern in self.security_patterns:
            if re.search(pattern, task, re.IGNORECASE):
                errors.append(f"Task contains potentially unsafe content: {pattern}")
        
        # Content quality warnings
        if len(task) < 10:
            warnings.append("Task is very short - consider providing more detail")
        
        if not any(char.isalpha() for char in task):
            warnings.append("Task contains no alphabetic characters")
        
        # Sanitize input
        sanitized_task = self._sanitize_text(task)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized_task
        )
    
    def validate_llm_config(self, llm: str, config: Optional[Dict] = None) -> ValidationResult:
        """Validate LLM configuration."""
        errors = []
        warnings = []
        
        if not isinstance(llm, str):
            errors.append("LLM model must be a string")
        elif not llm.strip():
            errors.append("LLM model cannot be empty")
        
        # Validate known model patterns
        known_patterns = [
            r'gpt-[34](\.[0-9])?(-turbo)?',
            r'claude-[12]',
            r'text-davinci-[0-9]+',
            r'chat-bison',
            r'llama-[0-9]+[b]?'
        ]
        
        if llm and not any(re.match(pattern, llm, re.IGNORECASE) for pattern in known_patterns):
            warnings.append(f"Unknown LLM model: {llm}")
        
        # Validate config if provided
        if config:
            if not isinstance(config, dict):
                errors.append("LLM config must be a dictionary")
            else:
                # Check for required/recommended fields
                if 'temperature' in config:
                    temp = config['temperature']
                    if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
                        errors.append("Temperature must be between 0 and 2")
                
                if 'max_tokens' in config:
                    max_tokens = config['max_tokens']
                    if not isinstance(max_tokens, int) or max_tokens <= 0:
                        errors.append("max_tokens must be a positive integer")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_reflexion_params(
        self,
        max_iterations: int,
        success_threshold: float,
        reflection_type: str
    ) -> ValidationResult:
        """Validate reflexion parameters."""
        errors = []
        warnings = []
        
        # Validate max_iterations
        if not isinstance(max_iterations, int):
            errors.append("max_iterations must be an integer")
        elif max_iterations <= 0:
            errors.append("max_iterations must be positive")
        elif max_iterations > self.max_iterations:
            errors.append(f"max_iterations ({max_iterations}) exceeds maximum ({self.max_iterations})")
        elif max_iterations > 10:
            warnings.append("High iteration count may lead to increased costs and latency")
        
        # Validate success_threshold
        if not isinstance(success_threshold, (int, float)):
            errors.append("success_threshold must be a number")
        elif not self.min_success_threshold <= success_threshold <= self.max_success_threshold:
            errors.append(f"success_threshold must be between {self.min_success_threshold} and {self.max_success_threshold}")
        
        # Validate reflection_type
        valid_types = ["binary", "scalar", "structured"]
        if reflection_type not in valid_types:
            errors.append(f"reflection_type must be one of: {valid_types}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def sanitize_output(self, output: str) -> str:
        """Sanitize output for safe display."""
        if not isinstance(output, str):
            return str(output)
        
        return self._sanitize_text(output)
    
    def _sanitize_text(self, text: str) -> str:
        """Remove potentially harmful content from text."""
        # Remove script tags and their content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove other potentially harmful tags
        harmful_tags = ['iframe', 'object', 'embed', 'form', 'input']
        for tag in harmful_tags:
            text = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(f'<{tag}[^>]*/?>', '', text, flags=re.IGNORECASE)
        
        # Remove javascript: and vbscript: URLs
        text = re.sub(r'(javascript|vbscript):[^\\s]*', '', text, flags=re.IGNORECASE)
        
        # Limit excessive whitespace
        text = re.sub(r'\\s+', ' ', text)
        text = text.strip()
        
        return text


# Global validator instance
validator = InputValidator()
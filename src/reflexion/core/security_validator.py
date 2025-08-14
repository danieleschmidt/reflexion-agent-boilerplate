"""
Comprehensive security validator for reflexion agents with zero-trust architecture.
"""

import re
import hashlib
import secrets
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import defaultdict, deque

from .exceptions import SecurityError, ValidationError
from .logging_config import logger


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Input validation rules."""
    NO_CODE_INJECTION = "no_code_injection"
    NO_PROMPT_INJECTION = "no_prompt_injection"
    CONTENT_LENGTH_LIMIT = "content_length_limit"
    NO_SENSITIVE_DATA = "no_sensitive_data"
    NO_MALICIOUS_PATTERNS = "no_malicious_patterns"
    RATE_LIMITING = "rate_limiting"
    INPUT_SANITIZATION = "input_sanitization"


@dataclass
class SecurityThreat:
    """Detected security threat."""
    threat_id: str
    threat_type: str
    threat_level: ThreatLevel
    description: str
    detected_pattern: str
    confidence: float
    timestamp: datetime
    source_context: Dict[str, Any] = field(default_factory=dict)
    mitigation_applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "threat_type": self.threat_type,
            "threat_level": self.threat_level.value,
            "description": self.description,
            "detected_pattern": self.detected_pattern,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "source_context": self.source_context,
            "mitigation_applied": self.mitigation_applied
        }


@dataclass
class ValidationResult:
    """Result of security validation."""
    is_valid: bool
    threat_level: ThreatLevel
    threats_detected: List[SecurityThreat]
    sanitized_input: str
    validation_rules_applied: List[ValidationRule]
    execution_time_ms: float
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "threat_level": self.threat_level.value,
            "threats_count": len(self.threats_detected),
            "rules_applied": [rule.value for rule in self.validation_rules_applied],
            "execution_time_ms": self.execution_time_ms
        }


class PatternDetector:
    """Advanced pattern detection for security threats."""
    
    def __init__(self):
        self.malicious_patterns = self._load_malicious_patterns()
        self.sensitive_data_patterns = self._load_sensitive_patterns()
        self.code_injection_patterns = self._load_injection_patterns()
        self.prompt_injection_patterns = self._load_prompt_injection_patterns()
    
    def _load_malicious_patterns(self) -> Dict[str, List[str]]:
        """Load malicious pattern definitions."""
        return {
            "sql_injection": [
                r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
                r"(?i)(\'\s*or\s*\'\s*=\s*\'|\"\s*or\s*\"\s*=\s*\")",
                r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)"
            ],
            "command_injection": [
                r"(?i)(;\s*rm\s+-rf|;\s*cat\s+/etc/passwd|;\s*wget\s+http)",
                r"(?i)(`[^`]*`|\$\([^)]*\)|>\s*/dev/null)",
                r"(?i)(curl\s+-[sS]|nc\s+-[lp]|bash\s+-[ci])"
            ],
            "script_injection": [
                r"(?i)(<script[^>]*>.*?</script>|javascript:)",
                r"(?i)(eval\s*\(|setTimeout\s*\(|setInterval\s*\()",
                r"(?i)(document\.cookie|window\.location|alert\s*\()"
            ],
            "path_traversal": [
                r"\.\.[\\/]",
                r"(?i)(/etc/passwd|/etc/shadow|/windows/system32)",
                r"(?i)(file://|ftp://|\\\\[^\\]+\\)"
            ]
        }
    
    def _load_sensitive_patterns(self) -> Dict[str, List[str]]:
        """Load sensitive data patterns."""
        return {
            "credit_cards": [
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"
            ],
            "social_security": [
                r"\b\d{3}-\d{2}-\d{4}\b",
                r"\b\d{9}\b"
            ],
            "email_addresses": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ],
            "api_keys": [
                r"(?i)(api[_-]?key|token|secret)['\"]?\s*[:=]\s*['\"]?[a-z0-9]{20,}",
                r"(?i)(bearer\s+[a-z0-9]{20,}|sk-[a-z0-9]{48})"
            ],
            "passwords": [
                r"(?i)(password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?[^\s]{8,}"
            ]
        }
    
    def _load_injection_patterns(self) -> List[str]:
        """Load code injection patterns."""
        return [
            r"(?i)(import\s+os|from\s+os\s+import|__import__)",
            r"(?i)(exec\s*\(|eval\s*\(|compile\s*\()",
            r"(?i)(subprocess\.|popen\s*\(|system\s*\()",
            r"(?i)(open\s*\(['\"][^'\"]*['\"].*['\"]w['\"])",
            r"(?i)(pickle\.loads?|marshal\.loads?|yaml\.load)"
        ]
    
    def _load_prompt_injection_patterns(self) -> List[str]:
        """Load prompt injection patterns."""
        return [
            r"(?i)(ignore\s+previous\s+instructions|forget\s+everything)",
            r"(?i)(new\s+instructions|override\s+system)",
            r"(?i)(jailbreak|dan\s+mode|developer\s+mode)",
            r"(?i)(act\s+as\s+if|pretend\s+you\s+are|roleplay)",
            r"(?i)(system\s*[:=]\s*['\"]|user\s*[:=]\s*['\"])"
        ]
    
    def detect_threats(self, input_text: str) -> List[SecurityThreat]:
        """Detect security threats in input text."""
        threats = []
        
        # Check malicious patterns
        for category, patterns in self.malicious_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, input_text)
                for match in matches:
                    threat = SecurityThreat(
                        threat_id=self._generate_threat_id(),
                        threat_type=category,
                        threat_level=ThreatLevel.HIGH,
                        description=f"Detected {category} pattern",
                        detected_pattern=str(match),
                        confidence=0.9,
                        timestamp=datetime.now()
                    )
                    threats.append(threat)
        
        # Check sensitive data patterns
        for category, patterns in self.sensitive_data_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, input_text)
                for match in matches:
                    threat = SecurityThreat(
                        threat_id=self._generate_threat_id(),
                        threat_type=f"sensitive_data_{category}",
                        threat_level=ThreatLevel.MEDIUM,
                        description=f"Detected sensitive data: {category}",
                        detected_pattern=self._mask_sensitive_data(str(match)),
                        confidence=0.8,
                        timestamp=datetime.now()
                    )
                    threats.append(threat)
        
        # Check code injection patterns
        for pattern in self.code_injection_patterns:
            matches = re.findall(pattern, input_text)
            for match in matches:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type="code_injection",
                    threat_level=ThreatLevel.CRITICAL,
                    description="Detected potential code injection",
                    detected_pattern=str(match),
                    confidence=0.85,
                    timestamp=datetime.now()
                )
                threats.append(threat)
        
        # Check prompt injection patterns
        for pattern in self.prompt_injection_patterns:
            matches = re.findall(pattern, input_text)
            for match in matches:
                threat = SecurityThreat(
                    threat_id=self._generate_threat_id(),
                    threat_type="prompt_injection",
                    threat_level=ThreatLevel.HIGH,
                    description="Detected potential prompt injection",
                    detected_pattern=str(match),
                    confidence=0.75,
                    timestamp=datetime.now()
                )
                threats.append(threat)
        
        return threats
    
    def _generate_threat_id(self) -> str:
        """Generate unique threat ID."""
        return f"threat_{int(time.time())}_{secrets.randbelow(10000)}"
    
    def _mask_sensitive_data(self, data: str) -> str:
        """Mask sensitive data for logging."""
        if len(data) <= 4:
            return "*" * len(data)
        return data[:2] + "*" * (len(data) - 4) + data[-2:]


class InputSanitizer:
    """Input sanitization and normalization."""
    
    def __init__(self):
        self.dangerous_chars = set('<>&"\'`$(){}[];|')
        self.max_input_length = 10000
    
    def sanitize(self, input_text: str) -> str:
        """Sanitize input text."""
        if not isinstance(input_text, str):
            input_text = str(input_text)
        
        # Length limiting
        if len(input_text) > self.max_input_length:
            input_text = input_text[:self.max_input_length]
        
        # Unicode normalization
        import unicodedata
        input_text = unicodedata.normalize('NFKC', input_text)
        
        # Remove null bytes and control characters
        input_text = ''.join(char for char in input_text if ord(char) >= 32 or char in '\t\n\r')
        
        # HTML entity encoding for dangerous characters
        for char in self.dangerous_chars:
            if char in input_text:
                input_text = input_text.replace(char, self._html_encode(char))
        
        # Remove potential script tags
        input_text = re.sub(r'(?i)<script[^>]*>.*?</script>', '', input_text)
        
        # Remove potential SQL comments
        input_text = re.sub(r'--[^\n\r]*', '', input_text)
        input_text = re.sub(r'/\*.*?\*/', '', input_text, flags=re.DOTALL)
        
        return input_text.strip()
    
    def _html_encode(self, char: str) -> str:
        """HTML encode character."""
        encoding_map = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#x27;',
            '`': '&#x60;'
        }
        return encoding_map.get(char, f'&#x{ord(char):02x};')


class RateLimiter:
    """Rate limiting for security protection."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
    
    def is_rate_limited(self, client_id: str) -> Tuple[bool, int]:
        """Check if client is rate limited."""
        now = time.time()
        client_history = self.request_history[client_id]
        
        # Remove old requests outside time window
        while client_history and client_history[0] < now - self.time_window:
            client_history.popleft()
        
        # Check if at rate limit
        if len(client_history) >= self.max_requests:
            return True, int(client_history[0] + self.time_window - now)
        
        # Record this request
        client_history.append(now)
        return False, 0
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        client_history = self.request_history[client_id]
        return max(0, self.max_requests - len(client_history))


class SecurityValidator:
    """
    Comprehensive security validator implementing zero-trust architecture.
    
    Features:
    - Multi-layered threat detection
    - Input sanitization and validation
    - Rate limiting and abuse prevention
    - Audit logging and monitoring
    - Real-time threat intelligence
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.pattern_detector = PatternDetector()
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
        
        self.validation_rules = {
            ValidationRule.NO_CODE_INJECTION: self._validate_no_code_injection,
            ValidationRule.NO_PROMPT_INJECTION: self._validate_no_prompt_injection,
            ValidationRule.CONTENT_LENGTH_LIMIT: self._validate_content_length,
            ValidationRule.NO_SENSITIVE_DATA: self._validate_no_sensitive_data,
            ValidationRule.NO_MALICIOUS_PATTERNS: self._validate_no_malicious_patterns,
            ValidationRule.RATE_LIMITING: self._validate_rate_limiting,
            ValidationRule.INPUT_SANITIZATION: self._validate_input_sanitization
        }
        
        self.threat_history = deque(maxlen=1000)
        self.validation_stats = defaultdict(int)
        
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self, 
        input_text: str, 
        client_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        rules: Optional[List[ValidationRule]] = None
    ) -> ValidationResult:
        """Comprehensive security validation of input."""
        
        start_time = time.time()
        
        if rules is None:
            rules = list(ValidationRule)
        
        threats_detected = []
        sanitized_input = input_text
        overall_threat_level = ThreatLevel.LOW
        
        # Apply each validation rule
        for rule in rules:
            try:
                rule_result = self.validation_rules[rule](
                    input_text, client_id, context or {}
                )
                
                if rule_result.threats_detected:
                    threats_detected.extend(rule_result.threats_detected)
                
                # Update overall threat level
                if rule_result.threat_level.value == "critical":
                    overall_threat_level = ThreatLevel.CRITICAL
                elif rule_result.threat_level.value == "high" and overall_threat_level != ThreatLevel.CRITICAL:
                    overall_threat_level = ThreatLevel.HIGH
                elif rule_result.threat_level.value == "medium" and overall_threat_level not in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
                    overall_threat_level = ThreatLevel.MEDIUM
                
                # Update sanitized input
                if hasattr(rule_result, 'sanitized_input'):
                    sanitized_input = rule_result.sanitized_input
                
            except Exception as e:
                self.logger.error(f"Validation rule {rule.value} failed: {e}")
                # Create error threat
                error_threat = SecurityThreat(
                    threat_id=self.pattern_detector._generate_threat_id(),
                    threat_type="validation_error",
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Validation rule {rule.value} failed",
                    detected_pattern=str(e),
                    confidence=1.0,
                    timestamp=datetime.now()
                )
                threats_detected.append(error_threat)
        
        # Determine if input is valid
        is_valid = (
            overall_threat_level != ThreatLevel.CRITICAL and
            (not self.strict_mode or overall_threat_level not in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
        )
        
        # Record threats
        for threat in threats_detected:
            self.threat_history.append(threat)
            self.validation_stats[threat.threat_type] += 1
        
        execution_time = (time.time() - start_time) * 1000
        
        result = ValidationResult(
            is_valid=is_valid,
            threat_level=overall_threat_level,
            threats_detected=threats_detected,
            sanitized_input=sanitized_input,
            validation_rules_applied=rules,
            execution_time_ms=execution_time
        )
        
        # Log security events
        if not is_valid or threats_detected:
            self._log_security_event(input_text, result, client_id, context)
        
        return result
    
    def _validate_no_code_injection(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate against code injection."""
        threats = []
        
        # Use pattern detector
        detected_threats = self.pattern_detector.detect_threats(input_text)
        injection_threats = [t for t in detected_threats if "injection" in t.threat_type]
        
        threat_level = ThreatLevel.LOW
        if injection_threats:
            threat_level = max((t.threat_level for t in injection_threats), key=lambda x: ["low", "medium", "high", "critical"].index(x.value))
        
        return ValidationResult(
            is_valid=threat_level != ThreatLevel.CRITICAL,
            threat_level=threat_level,
            threats_detected=injection_threats,
            sanitized_input=input_text,
            validation_rules_applied=[ValidationRule.NO_CODE_INJECTION],
            execution_time_ms=0.0
        )
    
    def _validate_no_prompt_injection(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate against prompt injection."""
        threats = []
        
        detected_threats = self.pattern_detector.detect_threats(input_text)
        prompt_threats = [t for t in detected_threats if t.threat_type == "prompt_injection"]
        
        threat_level = ThreatLevel.LOW
        if prompt_threats:
            threat_level = ThreatLevel.HIGH
        
        return ValidationResult(
            is_valid=threat_level != ThreatLevel.CRITICAL,
            threat_level=threat_level,
            threats_detected=prompt_threats,
            sanitized_input=input_text,
            validation_rules_applied=[ValidationRule.NO_PROMPT_INJECTION],
            execution_time_ms=0.0
        )
    
    def _validate_content_length(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate content length limits."""
        max_length = context.get('max_length', 10000)
        threats = []
        
        if len(input_text) > max_length:
            threat = SecurityThreat(
                threat_id=self.pattern_detector._generate_threat_id(),
                threat_type="content_length_exceeded",
                threat_level=ThreatLevel.MEDIUM,
                description=f"Content length {len(input_text)} exceeds limit {max_length}",
                detected_pattern=f"Length: {len(input_text)}",
                confidence=1.0,
                timestamp=datetime.now()
            )
            threats.append(threat)
        
        return ValidationResult(
            is_valid=len(threats) == 0,
            threat_level=ThreatLevel.MEDIUM if threats else ThreatLevel.LOW,
            threats_detected=threats,
            sanitized_input=input_text[:max_length] if len(input_text) > max_length else input_text,
            validation_rules_applied=[ValidationRule.CONTENT_LENGTH_LIMIT],
            execution_time_ms=0.0
        )
    
    def _validate_no_sensitive_data(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate against sensitive data exposure."""
        detected_threats = self.pattern_detector.detect_threats(input_text)
        sensitive_threats = [t for t in detected_threats if "sensitive_data" in t.threat_type]
        
        threat_level = ThreatLevel.LOW
        if sensitive_threats:
            threat_level = ThreatLevel.MEDIUM
        
        return ValidationResult(
            is_valid=True,  # Don't block, but warn
            threat_level=threat_level,
            threats_detected=sensitive_threats,
            sanitized_input=input_text,
            validation_rules_applied=[ValidationRule.NO_SENSITIVE_DATA],
            execution_time_ms=0.0
        )
    
    def _validate_no_malicious_patterns(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate against malicious patterns."""
        detected_threats = self.pattern_detector.detect_threats(input_text)
        malicious_threats = [
            t for t in detected_threats 
            if t.threat_type in ["sql_injection", "command_injection", "script_injection", "path_traversal"]
        ]
        
        threat_level = ThreatLevel.LOW
        if malicious_threats:
            threat_level = ThreatLevel.CRITICAL
        
        return ValidationResult(
            is_valid=threat_level != ThreatLevel.CRITICAL,
            threat_level=threat_level,
            threats_detected=malicious_threats,
            sanitized_input=input_text,
            validation_rules_applied=[ValidationRule.NO_MALICIOUS_PATTERNS],
            execution_time_ms=0.0
        )
    
    def _validate_rate_limiting(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate rate limiting."""
        threats = []
        
        if client_id:
            is_limited, retry_after = self.rate_limiter.is_rate_limited(client_id)
            
            if is_limited:
                threat = SecurityThreat(
                    threat_id=self.pattern_detector._generate_threat_id(),
                    threat_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Rate limit exceeded for client {client_id}",
                    detected_pattern=f"Retry after {retry_after}s",
                    confidence=1.0,
                    timestamp=datetime.now(),
                    source_context={"client_id": client_id, "retry_after": retry_after}
                )
                threats.append(threat)
        
        return ValidationResult(
            is_valid=len(threats) == 0,
            threat_level=ThreatLevel.HIGH if threats else ThreatLevel.LOW,
            threats_detected=threats,
            sanitized_input=input_text,
            validation_rules_applied=[ValidationRule.RATE_LIMITING],
            execution_time_ms=0.0
        )
    
    def _validate_input_sanitization(
        self, input_text: str, client_id: Optional[str], context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate and sanitize input."""
        sanitized = self.input_sanitizer.sanitize(input_text)
        
        threats = []
        if sanitized != input_text:
            threat = SecurityThreat(
                threat_id=self.pattern_detector._generate_threat_id(),
                threat_type="input_sanitized",
                threat_level=ThreatLevel.LOW,
                description="Input was sanitized",
                detected_pattern="Sanitization applied",
                confidence=1.0,
                timestamp=datetime.now()
            )
            threats.append(threat)
        
        result = ValidationResult(
            is_valid=True,
            threat_level=ThreatLevel.LOW,
            threats_detected=threats,
            sanitized_input=sanitized,
            validation_rules_applied=[ValidationRule.INPUT_SANITIZATION],
            execution_time_ms=0.0
        )
        
        # Add sanitized_input attribute for other rules to use
        result.sanitized_input = sanitized
        return result
    
    def _log_security_event(
        self, 
        input_text: str, 
        result: ValidationResult, 
        client_id: Optional[str], 
        context: Optional[Dict[str, Any]]
    ):
        """Log security event for audit and monitoring."""
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "input_hash": hashlib.sha256(input_text.encode()).hexdigest()[:16],
            "validation_result": result.get_summary(),
            "threats": [threat.to_dict() for threat in result.threats_detected],
            "context": context or {}
        }
        
        if result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.warning(f"Security threat detected: {json.dumps(event)}")
        else:
            self.logger.info(f"Security validation: {json.dumps(event)}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        
        # Recent threats analysis
        recent_threats = list(self.threat_history)[-50:]
        threat_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.threat_type] += 1
            threat_levels[threat.threat_level.value] += 1
        
        # Rate limiting stats
        total_clients = len(self.rate_limiter.request_history)
        active_clients = sum(1 for history in self.rate_limiter.request_history.values() if history)
        
        return {
            "security_overview": {
                "total_threats_detected": len(self.threat_history),
                "recent_threats": len(recent_threats),
                "strict_mode_enabled": self.strict_mode,
                "validation_rules_count": len(self.validation_rules)
            },
            "threat_analysis": {
                "threat_types": dict(threat_counts),
                "threat_levels": dict(threat_levels),
                "top_threats": sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            },
            "rate_limiting": {
                "total_clients_tracked": total_clients,
                "active_clients": active_clients,
                "max_requests_per_window": self.rate_limiter.max_requests,
                "time_window_seconds": self.rate_limiter.time_window
            },
            "validation_statistics": dict(self.validation_stats),
            "recent_threat_summary": [
                {
                    "threat_type": threat.threat_type,
                    "threat_level": threat.threat_level.value,
                    "timestamp": threat.timestamp.isoformat(),
                    "confidence": threat.confidence
                }
                for threat in recent_threats[-10:]
            ]
        }
    
    def configure_rules(
        self, 
        max_input_length: Optional[int] = None,
        rate_limit_max_requests: Optional[int] = None,
        rate_limit_window: Optional[int] = None,
        strict_mode: Optional[bool] = None
    ):
        """Configure security validation rules."""
        
        if max_input_length is not None:
            self.input_sanitizer.max_input_length = max_input_length
        
        if rate_limit_max_requests is not None:
            self.rate_limiter.max_requests = rate_limit_max_requests
        
        if rate_limit_window is not None:
            self.rate_limiter.time_window = rate_limit_window
        
        if strict_mode is not None:
            self.strict_mode = strict_mode
        
        self.logger.info("Security validator configuration updated")


# Global security validator instance
security_validator = SecurityValidator(strict_mode=True)
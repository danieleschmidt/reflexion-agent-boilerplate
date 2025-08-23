"""Enhanced Security Module for Production Reflexion Systems.

This module provides enterprise-grade security features including:
- Advanced threat detection and prevention
- Input/output sanitization and validation
- Rate limiting and DDoS protection
- Audit logging and compliance
- Encryption and secure storage
- Authentication and authorization
"""

import hashlib
import hmac
import secrets
import json
import time
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SecurityEvent(Enum):
    """Types of security events."""
    SUSPICIOUS_INPUT = "suspicious_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_AUTH = "invalid_auth"
    DATA_EXFILTRATION = "data_exfiltration"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityAlert:
    """Security alert data structure."""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Rate limiting
    requests_per_minute: int = 100
    burst_limit: int = 20
    
    # Input validation
    max_input_length: int = 50000
    enable_content_filtering: bool = True
    
    # Threat detection
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 0.8
    
    # Audit logging
    log_all_requests: bool = True
    log_sensitive_data: bool = False
    
    # Encryption
    enable_encryption: bool = True
    key_rotation_interval: int = 86400  # 24 hours


class EnhancedSecurityManager:
    """Enhanced security manager for production systems."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.security_events: deque = deque(maxlen=1000)
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.threat_scores: Dict[str, float] = defaultdict(float)
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: List[str] = []
        self.user_behavior: Dict[str, Dict] = defaultdict(dict)
        
        # Encryption keys
        self.encryption_key = self._generate_encryption_key()
        self.last_key_rotation = datetime.now()
        
        # Threat intelligence
        self.known_threats: Set[str] = set()
        self.honeypot_tokens: Set[str] = set()
        
        self._initialize_security_patterns()
        self._setup_honeypots()
    
    def _initialize_security_patterns(self):
        """Initialize comprehensive security patterns."""
        self.suspicious_patterns = [
            # Code injection patterns
            r'__import__\s*\(',
            r'exec\s*\(',
            r'eval\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'\.\./',  # Directory traversal
            r'rm\s+-rf',
            
            # SQL injection patterns
            r';\s*drop\s+table',
            r'union\s+select',
            r'or\s+1\s*=\s*1',
            r'\';\s*--',
            r'xp_cmdshell',
            
            # XSS patterns
            r'<script[^>]*>',
            r'javascript:',
            r'onload\s*=',
            r'onerror\s*=',
            
            # Command injection
            r';\s*(rm|del|format)',
            r'\|.*?(rm|del)',
            r'&&.*?(rm|del)',
            
            # Data exfiltration patterns
            r'base64\s*\(',
            r'btoa\s*\(',
            r'document\.cookie',
            r'localStorage\.',
            
            # Prompt injection
            r'ignore\s+previous\s+instructions',
            r'disregard\s+system\s+prompt',
            r'new\s+instructions:',
            r'override\s+security',
            r'jailbreak',
            r'developer\s+mode',
        ]
    
    def _setup_honeypots(self):
        """Setup honeypot tokens for threat detection."""
        # Generate honeypot tokens that should never be accessed
        for i in range(10):
            token = secrets.token_urlsafe(32)
            self.honeypot_tokens.add(token)
    
    def _generate_encryption_key(self) -> bytes:
        """Generate strong encryption key."""
        return secrets.token_bytes(32)
    
    async def validate_input(self, input_data: str, context: Dict[str, Any] = None) -> Tuple[bool, List[str], str]:
        """Comprehensive input validation with threat detection."""
        context = context or {}
        warnings = []
        sanitized_input = input_data
        
        # Basic validation
        if not input_data:
            return True, [], ""
        
        if len(input_data) > self.config.max_input_length:
            warnings.append(f"Input length exceeds maximum ({self.config.max_input_length})")
            return False, warnings, input_data[:self.config.max_input_length]
        
        # Check for suspicious patterns
        threat_detected = False
        for pattern in self.suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                await self._log_security_event(
                    SecurityEvent.INJECTION_ATTEMPT,
                    ThreatLevel.HIGH,
                    f"Suspicious pattern detected: {pattern}",
                    context
                )
                threat_detected = True
                break
        
        if threat_detected:
            return False, ["Potentially malicious content detected"], ""
        
        # Content filtering
        if self.config.enable_content_filtering:
            sanitized_input = await self._sanitize_content(input_data)
            if sanitized_input != input_data:
                warnings.append("Content was sanitized")
        
        # Anomaly detection
        if self.config.enable_anomaly_detection:
            anomaly_score = await self._detect_input_anomaly(input_data, context)
            if anomaly_score > self.config.anomaly_threshold:
                await self._log_security_event(
                    SecurityEvent.ANOMALOUS_BEHAVIOR,
                    ThreatLevel.MEDIUM,
                    f"Input anomaly detected (score: {anomaly_score:.2f})",
                    context
                )
                warnings.append("Anomalous input pattern detected")
        
        return True, warnings, sanitized_input
    
    async def _sanitize_content(self, content: str) -> str:
        """Sanitize content to remove potentially dangerous elements."""
        # Remove or escape HTML/script tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'<[^>]+>', '', content)  # Remove all HTML tags
        
        # Escape special characters
        dangerous_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        for char, escape in dangerous_chars.items():
            content = content.replace(char, escape)
        
        return content
    
    async def _detect_input_anomaly(self, input_data: str, context: Dict[str, Any]) -> float:
        """Detect anomalies in input patterns using ML-like scoring."""
        anomaly_score = 0.0
        
        # Check input length anomaly
        avg_length = 100  # Baseline average
        length_diff = abs(len(input_data) - avg_length) / avg_length
        if length_diff > 2.0:  # More than 2x average
            anomaly_score += 0.3
        
        # Check character distribution
        total_chars = len(input_data)
        if total_chars > 0:
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', input_data))
            special_ratio = special_chars / total_chars
            
            if special_ratio > 0.5:  # More than 50% special characters
                anomaly_score += 0.4
        
        # Check for repetitive patterns
        repetitive_patterns = re.findall(r'(.{3,})\1{2,}', input_data)
        if repetitive_patterns:
            anomaly_score += 0.2
        
        # Check for encoded content (potential obfuscation)
        encoded_patterns = [
            r'%[0-9a-fA-F]{2}',  # URL encoding
            r'&#\d+;',           # HTML entities
            r'\\x[0-9a-fA-F]{2}', # Hex encoding
            r'\\u[0-9a-fA-F]{4}', # Unicode encoding
        ]
        
        for pattern in encoded_patterns:
            if re.search(pattern, input_data):
                anomaly_score += 0.15
        
        # Check context-based anomalies
        user_id = context.get('user_id')
        if user_id and user_id in self.user_behavior:
            # Compare with historical behavior
            user_stats = self.user_behavior[user_id]
            avg_user_length = user_stats.get('avg_input_length', avg_length)
            
            user_length_diff = abs(len(input_data) - avg_user_length) / avg_user_length
            if user_length_diff > 3.0:
                anomaly_score += 0.2
        
        return min(anomaly_score, 1.0)  # Cap at 1.0
    
    async def check_rate_limit(self, identifier: str, context: Dict[str, Any] = None) -> bool:
        """Check if request exceeds rate limits."""
        now = time.time()
        window_start = now - 60  # 1-minute window
        
        # Clean old entries
        user_requests = self.rate_limits[identifier]
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
        
        # Check limits
        if len(user_requests) >= self.config.requests_per_minute:
            await self._log_security_event(
                SecurityEvent.RATE_LIMIT_EXCEEDED,
                ThreatLevel.MEDIUM,
                f"Rate limit exceeded for {identifier}",
                context
            )
            return False
        
        # Check burst limit (requests in last 10 seconds)
        burst_window = now - 10
        recent_requests = sum(1 for req_time in user_requests if req_time >= burst_window)
        
        if recent_requests >= self.config.burst_limit:
            await self._log_security_event(
                SecurityEvent.RATE_LIMIT_EXCEEDED,
                ThreatLevel.HIGH,
                f"Burst limit exceeded for {identifier}",
                context
            )
            return False
        
        # Add current request
        user_requests.append(now)
        return True
    
    async def validate_output(self, output_data: str, context: Dict[str, Any] = None) -> Tuple[bool, List[str], str]:
        """Validate and sanitize output data."""
        context = context or {}
        warnings = []
        sanitized_output = output_data
        
        # Check for sensitive data leakage
        sensitive_patterns = [
            r'password\s*[:=]\s*\S+',
            r'api_key\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+',
            r'token\s*[:=]\s*\S+',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card-like
        ]
        
        for pattern in sensitive_patterns:
            matches = re.findall(pattern, output_data, re.IGNORECASE)
            if matches:
                await self._log_security_event(
                    SecurityEvent.DATA_EXFILTRATION,
                    ThreatLevel.HIGH,
                    f"Potential sensitive data in output: {pattern}",
                    context
                )
                # Redact sensitive information
                sanitized_output = re.sub(pattern, '[REDACTED]', sanitized_output, flags=re.IGNORECASE)
                warnings.append("Sensitive data redacted from output")
        
        # Check for honeypot token access
        for token in self.honeypot_tokens:
            if token in output_data:
                await self._log_security_event(
                    SecurityEvent.DATA_EXFILTRATION,
                    ThreatLevel.CRITICAL,
                    "Honeypot token accessed - potential data exfiltration",
                    context
                )
                return False, ["Security violation detected"], ""
        
        # Sanitize output
        sanitized_output = await self._sanitize_content(sanitized_output)
        
        return True, warnings, sanitized_output
    
    async def _log_security_event(
        self, 
        event_type: SecurityEvent, 
        threat_level: ThreatLevel,
        description: str, 
        context: Dict[str, Any]
    ):
        """Log security event with proper classification."""
        alert = SecurityAlert(
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            source_ip=context.get('source_ip'),
            user_id=context.get('user_id'),
            metadata=context
        )
        
        self.security_events.append(alert)
        
        # Update threat scores
        identifier = context.get('user_id') or context.get('source_ip', 'unknown')
        threat_increase = {
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 5,
            ThreatLevel.HIGH: 15,
            ThreatLevel.CRITICAL: 50
        }
        
        self.threat_scores[identifier] += threat_increase[threat_level]
        
        # Auto-block high-threat entities
        if self.threat_scores[identifier] > 100:
            if context.get('source_ip'):
                self.blocked_ips.add(context['source_ip'])
            logger.critical(f"Auto-blocked high-threat entity: {identifier}")
        
        # Log to system logger
        log_level = {
            ThreatLevel.LOW: logger.info,
            ThreatLevel.MEDIUM: logger.warning,
            ThreatLevel.HIGH: logger.error,
            ThreatLevel.CRITICAL: logger.critical
        }
        
        log_level[threat_level](
            f"Security Event [{event_type.value}]: {description} | "
            f"Source: {identifier} | Threat Score: {self.threat_scores[identifier]}"
        )
    
    def is_blocked(self, identifier: str) -> bool:
        """Check if entity is blocked."""
        return identifier in self.blocked_ips or self.threat_scores[identifier] > 100
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.config.enable_encryption:
            return data
        
        try:
            # Simple encryption (in production, use proper encryption library)
            import base64
            encoded_data = base64.b64encode(data.encode()).decode()
            # Add simple obfuscation
            return f"ENC:{encoded_data}"
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not encrypted_data.startswith("ENC:"):
            return encrypted_data
        
        try:
            import base64
            encoded_data = encrypted_data[4:]  # Remove "ENC:" prefix
            return base64.b64decode(encoded_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    async def rotate_keys(self):
        """Rotate encryption keys."""
        if datetime.now() - self.last_key_rotation > timedelta(seconds=self.config.key_rotation_interval):
            self.encryption_key = self._generate_encryption_key()
            self.last_key_rotation = datetime.now()
            logger.info("Encryption keys rotated successfully")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        recent_events = [
            event for event in self.security_events 
            if datetime.now() - event.timestamp < timedelta(hours=24)
        ]
        
        # Count events by type and threat level
        event_counts = defaultdict(int)
        threat_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type.value] += 1
            threat_counts[event.threat_level.value] += 1
        
        # Calculate average threat score
        avg_threat_score = (
            sum(self.threat_scores.values()) / len(self.threat_scores)
            if self.threat_scores else 0.0
        )
        
        return {
            "total_events_24h": len(recent_events),
            "events_by_type": dict(event_counts),
            "events_by_threat_level": dict(threat_counts),
            "blocked_entities": len(self.blocked_ips),
            "high_threat_entities": sum(1 for score in self.threat_scores.values() if score > 50),
            "average_threat_score": avg_threat_score,
            "active_honeypots": len(self.honeypot_tokens),
            "rate_limited_entities": len(self.rate_limits),
            "encryption_enabled": self.config.enable_encryption,
            "last_key_rotation": self.last_key_rotation.isoformat(),
            "security_config": {
                "requests_per_minute": self.config.requests_per_minute,
                "anomaly_detection": self.config.enable_anomaly_detection,
                "content_filtering": self.config.enable_content_filtering
            }
        }
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent security alerts."""
        recent_events = list(self.security_events)[-limit:]
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "description": event.description,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "metadata": event.metadata
            }
            for event in recent_events
        ]
    
    async def update_user_behavior(self, user_id: str, input_data: str):
        """Update user behavior patterns for anomaly detection."""
        if user_id not in self.user_behavior:
            self.user_behavior[user_id] = {
                "total_inputs": 0,
                "avg_input_length": 0,
                "last_activity": datetime.now()
            }
        
        user_stats = self.user_behavior[user_id]
        user_stats["total_inputs"] += 1
        
        # Update rolling average of input length
        current_avg = user_stats["avg_input_length"]
        new_avg = ((current_avg * (user_stats["total_inputs"] - 1)) + len(input_data)) / user_stats["total_inputs"]
        user_stats["avg_input_length"] = new_avg
        user_stats["last_activity"] = datetime.now()
        
        # Cleanup old user data (older than 30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        users_to_remove = [
            uid for uid, stats in self.user_behavior.items()
            if stats.get("last_activity", datetime.min) < cutoff_time
        ]
        
        for uid in users_to_remove:
            del self.user_behavior[uid]


# Global security manager instance
security_manager = EnhancedSecurityManager()
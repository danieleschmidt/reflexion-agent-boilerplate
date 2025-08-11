"""Security utilities for reflexion agents."""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .exceptions import SecurityError
from .logging_config import logger


class SecurityManager:
    """Centralized security management for reflexion systems with enhanced threat detection."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.blocked_patterns: Set[str] = set()
        self.security_events: List[Dict[str, Any]] = []
        self.threat_scores: Dict[str, float] = {}  # Track threat scores per identifier
        self.honeypot_tokens: Set[str] = set()  # Honeypot detection
        
        # Enhanced security configurations
        self.max_threat_score = 100.0
        self.auto_block_threshold = 75.0
        self.anomaly_detection_enabled = True
        
        # Load default security patterns
        self._load_security_patterns()
        self._initialize_honeypots()
    
    def _load_security_patterns(self):
        """Load comprehensive security patterns to block with advanced threat detection."""
        self.blocked_patterns.update([
            # Code execution patterns - Enhanced
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            r'compile\s*\(',
            r'globals\s*\(\)',
            r'locals\s*\(\)',
            
            # File system access - Enhanced
            r'open\s*\(',
            r'file\s*\(',
            r'\.read\s*\(',
            r'\.write\s*\(',
            r'pathlib\.',
            r'shutil\.',
            r'tempfile\.',
            
            # Network access - Enhanced
            r'urllib\.',
            r'requests\.',
            r'socket\.',
            r'http\.',
            r'ftp\.',
            r'telnet\.',
            r'smtp\.',
            
            # Injection patterns - Enhanced
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'data:text\/html',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'onload\s*=',
            r'onerror\s*=',
            
            # Shell commands - Enhanced
            r'\|\s*sh',
            r'\|\s*bash',
            r'\|\s*zsh',
            r'rm\s+-rf',
            r'sudo\s+',
            r'chmod\s+[0-9]+',
            r'chown\s+',
            r'wget\s+',
            r'curl\s+',
            
            # Database injection patterns
            r'DROP\s+TABLE',
            r'DELETE\s+FROM',
            r'INSERT\s+INTO',
            r'UPDATE\s+.*SET',
            r'UNION\s+SELECT',
            r'OR\s+1\s*=\s*1',
            
            # System information gathering
            r'/etc/passwd',
            r'/proc/version',
            r'whoami',
            r'id\s*;',
            
            # Cryptographic attacks
            r'md5\s*\(',
            r'sha1\s*\(',
            r'hash\s*\(',
            
            # Advanced persistent threats
            r'powershell',
            r'cmd\.exe',
            r'wmic\s+',
            r'reg\s+add',
            
            # AI-specific attacks
            r'ignore\s+previous\s+instructions',
            r'system\s+prompt',
            r'bypass\s+safety',
            r'jailbreak',
        ])
    
    def validate_input_security(self, input_text: str, context: str = "unknown") -> bool:
        """Validate input for security issues."""
        import re
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                self._log_security_event(
                    event_type="blocked_pattern",
                    details={
                        "pattern": pattern,
                        "context": context,
                        "input_sample": input_text[:100]
                    }
                )
                raise SecurityError(
                    f"Input contains blocked security pattern: {pattern}",
                    pattern,
                    input_text
                )
        
        return True
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if request is within rate limits."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Initialize or clean old requests
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old requests outside the window
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if req_time > window_start
        ]
        
        # Check current rate
        current_requests = len(self.rate_limits[identifier])
        
        if current_requests >= max_requests:
            self._log_security_event(
                event_type="rate_limit_exceeded",
                details={
                    "identifier": identifier,
                    "current_requests": current_requests,
                    "max_requests": max_requests,
                    "window_minutes": window_minutes
                }
            )
            return False
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """Generate a secure API key with metadata."""
        key = secrets.token_urlsafe(32)
        
        self.api_keys[key] = {
            "name": name,
            "permissions": permissions or ["read"],
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        logger.info(f"API key generated for: {name}")
        return key
    
    def create_secure_hash(self, data: str, salt: Optional[str] = None) -> tuple:
        """Create secure hash with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hash_obj.hex(), salt
    
    def verify_secure_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify secure hash."""
        computed_hash, _ = self.create_secure_hash(data, salt)
        return hmac.compare_digest(computed_hash, hash_value)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary and statistics."""
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "active_api_keys": len([k for k, v in self.api_keys.items() if v["active"]]),
            "total_api_keys": len(self.api_keys),
            "blocked_patterns": len(self.blocked_patterns),
            "security_events_24h": len(recent_events),
            "rate_limited_identifiers": len(self.rate_limits),
            "recent_event_types": list(set(event["event_type"] for event in recent_events))
        }
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        # Keep only recent events to prevent memory growth
        cutoff_time = datetime.now() - timedelta(days=7)
        self.security_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]
        
        logger.warning(f"Security event: {event_type} - {details}")
        
        # Update threat scoring
        if 'identifier' in details:
            self._update_threat_score(details['identifier'], event_type)
    
    def _initialize_honeypots(self):
        """Initialize honeypot tokens for advanced threat detection."""
        honeypots = [
            "admin_secret_key_do_not_use",
            "development_api_token",
            "test_password_123",
            "backup_access_key",
            "internal_debug_token"
        ]
        self.honeypot_tokens.update(honeypots)
    
    def _update_threat_score(self, identifier: str, event_type: str):
        """Update threat score for an identifier based on security events."""
        if identifier not in self.threat_scores:
            self.threat_scores[identifier] = 0.0
        
        # Scoring based on event severity
        score_increments = {
            "blocked_pattern": 25.0,
            "rate_limit_exceeded": 15.0,
            "honeypot_access": 50.0,
            "injection_attempt": 40.0,
            "privilege_escalation": 60.0,
            "data_exfiltration": 70.0
        }
        
        increment = score_increments.get(event_type, 10.0)
        self.threat_scores[identifier] = min(
            self.max_threat_score,
            self.threat_scores[identifier] + increment
        )
        
        # Auto-block high-threat identifiers
        if (self.threat_scores[identifier] >= self.auto_block_threshold and 
            identifier not in self.blocked_identifiers):
            self.block_identifier(identifier, f"Auto-blocked due to threat score: {self.threat_scores[identifier]}")
            
        logger.info(f"Updated threat score for {identifier}: {self.threat_scores[identifier]:.1f}")
    
    def check_honeypot_access(self, input_text: str, identifier: str) -> bool:
        """Check if input contains honeypot tokens (indicates malicious probing)."""
        for token in self.honeypot_tokens:
            if token.lower() in input_text.lower():
                self._log_security_event(
                    event_type="honeypot_access",
                    details={
                        "identifier": identifier,
                        "honeypot_token": token,
                        "input_sample": input_text[:100]
                    }
                )
                return True
        return False
    
    def analyze_behavioral_anomalies(self, identifier: str, request_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced behavioral analysis for anomaly detection."""
        anomalies = []
        risk_score = 0.0
        
        # Check request frequency patterns
        if request_patterns.get('requests_per_minute', 0) > 60:
            anomalies.append("High request frequency")
            risk_score += 20.0
        
        # Check request size patterns
        avg_request_size = request_patterns.get('average_request_size', 0)
        if avg_request_size > 10000:  # Very large requests
            anomalies.append("Unusually large request sizes")
            risk_score += 15.0
        
        # Check time-based patterns
        request_hours = request_patterns.get('request_hours', [])
        if len(request_hours) > 0:
            # Requests during unusual hours (2-6 AM) might be suspicious
            unusual_hours = sum(1 for hour in request_hours if 2 <= hour <= 6)
            if unusual_hours / len(request_hours) > 0.5:
                anomalies.append("High activity during unusual hours")
                risk_score += 10.0
        
        # Check geographic anomalies (if available)
        ip_locations = request_patterns.get('ip_locations', [])
        if len(set(ip_locations)) > 5:  # Many different locations
            anomalies.append("Requests from many geographic locations")
            risk_score += 25.0
        
        if anomalies:
            self._log_security_event(
                event_type="behavioral_anomaly",
                details={
                    "identifier": identifier,
                    "anomalies": anomalies,
                    "risk_score": risk_score,
                    "patterns": request_patterns
                }
            )
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "risk_score": risk_score,
            "recommended_action": "block" if risk_score > 50 else "monitor"
        }


class HealthChecker:
    """Health check utilities for reflexion systems."""
    
    def __init__(self):
        self.checks = {}
        self.last_results = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("log_directory", self._check_log_directory)
    
    def register_check(self, name: str, check_function):
        """Register a health check function."""
        self.checks[name] = check_function
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "pass" if check_result["healthy"] else "fail",
                    "message": check_result["message"],
                    "details": check_result.get("details", {})
                }
                
                if not check_result["healthy"]:
                    results["overall_status"] = "unhealthy"
                    
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                    "details": {}
                }
                results["overall_status"] = "unhealthy"
        
        self.last_results = results
        return results
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            threshold = 90  # 90% usage threshold
            usage_percent = memory.percent
            
            return {
                "healthy": usage_percent < threshold,
                "message": f"Memory usage: {usage_percent:.1f}%",
                "details": {
                    "usage_percent": usage_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            }
        except ImportError:
            return {
                "healthy": True,
                "message": "Memory check skipped (psutil not available)",
                "details": {}
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            usage_percent = (used / total) * 100
            threshold = 90  # 90% usage threshold
            
            return {
                "healthy": usage_percent < threshold,
                "message": f"Disk usage: {usage_percent:.1f}%",
                "details": {
                    "usage_percent": usage_percent,
                    "free_gb": free / (1024**3),
                    "total_gb": total / (1024**3)
                }
            }
        except Exception:
            return {
                "healthy": True,
                "message": "Disk check skipped (unable to determine usage)",
                "details": {}
            }
    
    def _check_log_directory(self) -> Dict[str, Any]:
        """Check log directory accessibility."""
        from pathlib import Path
        
        log_dir = Path("./logs")
        
        try:
            log_dir.mkdir(exist_ok=True)
            test_file = log_dir / "health_check.tmp"
            
            # Test write access
            test_file.write_text("health check")
            content = test_file.read_text()
            test_file.unlink()
            
            return {
                "healthy": content == "health check",
                "message": "Log directory accessible",
                "details": {
                    "path": str(log_dir.absolute()),
                    "writable": True
                }
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Log directory not accessible: {str(e)}",
                "details": {
                    "path": str(log_dir.absolute()),
                    "error": str(e)
                }
            }


    def block_identifier(self, identifier: str, reason: str):
        """Block an identifier from making further requests."""
        if not hasattr(self, 'blocked_identifiers'):
            self.blocked_identifiers = set()
        
        self.blocked_identifiers.add(identifier)
        
        self._log_security_event(
            event_type="identifier_blocked",
            details={
                "identifier": identifier,
                "reason": reason,
                "threat_score": self.threat_scores.get(identifier, 0.0)
            }
        )
        
        logger.warning(f"Blocked identifier {identifier}: {reason}")
    
    def is_blocked(self, identifier: str) -> bool:
        """Check if an identifier is blocked."""
        if not hasattr(self, 'blocked_identifiers'):
            self.blocked_identifiers = set()
        return identifier in self.blocked_identifiers
    
    def get_threat_assessment(self, identifier: str) -> Dict[str, Any]:
        """Get comprehensive threat assessment for an identifier."""
        return {
            "identifier": identifier,
            "threat_score": self.threat_scores.get(identifier, 0.0),
            "is_blocked": self.is_blocked(identifier),
            "security_events": [
                event for event in self.security_events
                if event.get('details', {}).get('identifier') == identifier
            ][-10:],  # Last 10 events
            "risk_level": self._calculate_risk_level(identifier),
            "recommendations": self._get_security_recommendations(identifier)
        }
    
    def _calculate_risk_level(self, identifier: str) -> str:
        """Calculate risk level based on threat score and patterns."""
        score = self.threat_scores.get(identifier, 0.0)
        
        if score >= 75:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 25:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_security_recommendations(self, identifier: str) -> List[str]:
        """Get security recommendations for an identifier."""
        recommendations = []
        score = self.threat_scores.get(identifier, 0.0)
        
        if score >= 50:
            recommendations.append("Consider immediate blocking or strict rate limiting")
        elif score >= 25:
            recommendations.append("Increase monitoring and implement additional validation")
        
        if self.is_blocked(identifier):
            recommendations.append("Currently blocked - review for potential unblocking")
        
        return recommendations


# Global instances with enhanced security
security_manager = SecurityManager()
health_checker = HealthChecker()
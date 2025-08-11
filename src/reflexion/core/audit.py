"""Comprehensive audit and logging system for reflexion agents."""

import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from .logging_config import logger


@dataclass
class AuditEvent:
    """Structured audit event."""
    event_id: str
    timestamp: str
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    resource: str
    result: str
    risk_level: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    
    
@dataclass 
class SystemMetrics:
    """System performance and security metrics."""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_sessions: int
    failed_authentications: int
    security_events: int
    api_requests: int
    error_rate: float
    response_time_p95: float


class AuditLogger:
    """Enterprise-grade audit logging system."""
    
    def __init__(self, audit_dir: str = "./audit_logs"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)
        
        # Audit configuration
        self.max_log_size_mb = 100
        self.retention_days = 2555  # 7 years for compliance
        self.encrypt_logs = True
        self.integrity_check = True
        
        # Audit categories for risk assessment
        self.high_risk_actions = {
            "data_deletion", "user_creation", "privilege_escalation",
            "system_configuration", "security_bypass", "bulk_data_access"
        }
        
        self.medium_risk_actions = {
            "data_access", "user_modification", "configuration_change",
            "failed_authentication", "unusual_activity"
        }
        
        # Initialize audit streams
        self.current_log_file = self._get_current_log_file()
        self.events_buffer = []
        self.buffer_size = 100
        
        # Metrics tracking
        self.metrics_history: List[SystemMetrics] = []
        
    def log_event(
        self,
        event_type: str,
        action: str,
        resource: str,
        result: str = "success",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Log audit event with comprehensive context."""
        
        event_id = self._generate_event_id()
        risk_level = self._calculate_risk_level(action, result, details or {})
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            result=result,
            risk_level=risk_level,
            details=details or {},
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            correlation_id=kwargs.get('correlation_id')
        )
        
        # Add to buffer
        self.events_buffer.append(event)
        
        # Flush buffer if full
        if len(self.events_buffer) >= self.buffer_size:
            self._flush_buffer()
        
        # Log high-risk events immediately
        if risk_level in ["HIGH", "CRITICAL"]:
            logger.warning(f"High-risk audit event: {action} by {user_id} - {result}")
            self._flush_buffer()  # Immediate flush for critical events
        
        return event_id
    
    def log_security_incident(
        self,
        incident_type: str,
        description: str,
        severity: str,
        affected_resources: List[str],
        user_id: Optional[str] = None,
        mitigation_actions: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Log security incident with detailed tracking."""
        
        details = {
            "incident_type": incident_type,
            "description": description,
            "severity": severity,
            "affected_resources": affected_resources,
            "mitigation_actions": mitigation_actions or [],
            "detection_time": datetime.now().isoformat(),
            "status": "open"
        }
        
        return self.log_event(
            event_type="security_incident",
            action=f"security_incident_{incident_type}",
            resource="security_system",
            result="incident_detected",
            user_id=user_id,
            details=details,
            **kwargs
        )
    
    def log_data_access(
        self,
        data_type: str,
        access_type: str,
        user_id: str,
        resource_id: str,
        success: bool = True,
        data_classification: str = "internal",
        **kwargs
    ) -> str:
        """Log data access events for compliance tracking."""
        
        details = {
            "data_type": data_type,
            "access_type": access_type,
            "data_classification": data_classification,
            "resource_id": resource_id,
            "compliance_relevant": True
        }
        
        return self.log_event(
            event_type="data_access",
            action=f"data_{access_type}",
            resource=resource_id,
            result="success" if success else "failure",
            user_id=user_id,
            details=details,
            **kwargs
        )
    
    def log_system_metrics(self, metrics: SystemMetrics):
        """Log system performance and security metrics."""
        self.metrics_history.append(metrics)
        
        # Keep only last 30 days of metrics
        cutoff_time = datetime.now() - timedelta(days=30)
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        # Log high resource usage as audit event
        if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
            self.log_event(
                event_type="system_alert",
                action="high_resource_usage",
                resource="system",
                result="alert",
                details={
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "disk_usage": metrics.disk_usage
                }
            )
    
    @contextmanager
    def audit_session(self, user_id: str, session_type: str = "user_session"):
        """Context manager for auditing complete user sessions."""
        session_id = self._generate_session_id()
        start_time = datetime.now()
        
        self.log_event(
            event_type="session",
            action="session_start",
            resource="authentication_system",
            result="success",
            user_id=user_id,
            session_id=session_id,
            details={"session_type": session_type}
        )
        
        try:
            yield session_id
        except Exception as e:
            self.log_event(
                event_type="session",
                action="session_error",
                resource="authentication_system", 
                result="error",
                user_id=user_id,
                session_id=session_id,
                details={"error": str(e), "session_type": session_type}
            )
            raise
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_event(
                event_type="session",
                action="session_end",
                resource="authentication_system",
                result="success",
                user_id=user_id,
                session_id=session_id,
                details={
                    "session_type": session_type,
                    "duration_seconds": duration
                }
            )
    
    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report for specified period."""
        
        # Load events from the time period
        events = self._load_events_for_period(start_date, end_date)
        
        # Analyze events
        total_events = len(events)
        
        # Event type distribution
        event_types = {}
        risk_levels = {}
        user_activity = {}
        failed_events = 0
        
        for event in events:
            # Event types
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            # Risk levels
            risk_levels[event.risk_level] = risk_levels.get(event.risk_level, 0) + 1
            
            # User activity
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            # Failed events
            if event.result in ["failure", "error", "denied"]:
                failed_events += 1
        
        # Security incidents
        security_incidents = [
            event for event in events
            if event.event_type == "security_incident"
        ]
        
        # High-risk activities
        high_risk_events = [
            event for event in events
            if event.risk_level in ["HIGH", "CRITICAL"]
        ]
        
        # Generate report
        report = {
            "report_id": self._generate_event_id(),
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "summary": {
                "total_events": total_events,
                "unique_users": len(user_activity),
                "security_incidents": len(security_incidents),
                "high_risk_events": len(high_risk_events),
                "failure_rate": (failed_events / max(total_events, 1)) * 100
            },
            "distributions": {
                "event_types": event_types,
                "risk_levels": risk_levels,
                "top_users": dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            "security_analysis": {
                "incidents_by_type": self._analyze_incidents_by_type(security_incidents),
                "high_risk_patterns": self._analyze_risk_patterns(high_risk_events),
                "unusual_activities": self._detect_unusual_activities(events)
            }
        }
        
        # Add metrics analysis if requested
        if include_metrics:
            relevant_metrics = [
                m for m in self.metrics_history
                if start_date <= datetime.fromisoformat(m.timestamp) <= end_date
            ]
            
            report["system_metrics"] = self._analyze_system_metrics(relevant_metrics)
        
        return report
    
    def search_audit_logs(
        self,
        query: Dict[str, Any],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Search audit logs with flexible criteria."""
        
        # Set default date range if not provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Load events for the period
        events = self._load_events_for_period(start_date, end_date)
        
        # Apply filters
        filtered_events = []
        for event in events:
            if self._matches_query(event, query):
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
        
        return filtered_events
    
    def _calculate_risk_level(self, action: str, result: str, details: Dict[str, Any]) -> str:
        """Calculate risk level based on action, result, and context."""
        
        # Critical failures
        if result in ["security_violation", "unauthorized_access", "data_breach"]:
            return "CRITICAL"
        
        # High-risk actions
        if action in self.high_risk_actions:
            return "HIGH" if result == "success" else "CRITICAL"
        
        # Medium-risk actions
        if action in self.medium_risk_actions:
            return "MEDIUM" if result == "success" else "HIGH"
        
        # Failed operations
        if result in ["failure", "error", "denied"]:
            return "MEDIUM"
        
        # Check for suspicious patterns in details
        if self._contains_suspicious_patterns(details):
            return "HIGH"
        
        return "LOW"
    
    def _contains_suspicious_patterns(self, details: Dict[str, Any]) -> bool:
        """Check for suspicious patterns in event details."""
        suspicious_patterns = [
            "brute_force", "injection", "escalation", "anomalous",
            "unauthorized", "suspicious", "malicious"
        ]
        
        details_str = json.dumps(details).lower()
        return any(pattern in details_str for pattern in suspicious_patterns)
    
    def _matches_query(self, event: AuditEvent, query: Dict[str, Any]) -> bool:
        """Check if event matches search query."""
        
        for key, value in query.items():
            if key == "event_type" and event.event_type != value:
                return False
            elif key == "user_id" and event.user_id != value:
                return False
            elif key == "action" and value not in event.action:
                return False
            elif key == "risk_level" and event.risk_level != value:
                return False
            elif key == "result" and event.result != value:
                return False
        
        return True
    
    def _analyze_incidents_by_type(self, incidents: List[AuditEvent]) -> Dict[str, int]:
        """Analyze security incidents by type."""
        incident_types = {}
        for incident in incidents:
            incident_type = incident.details.get("incident_type", "unknown")
            incident_types[incident_type] = incident_types.get(incident_type, 0) + 1
        
        return incident_types
    
    def _analyze_risk_patterns(self, high_risk_events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Analyze patterns in high-risk events."""
        patterns = []
        
        # Group by action
        action_groups = {}
        for event in high_risk_events:
            action = event.action
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append(event)
        
        # Analyze each action group
        for action, events in action_groups.items():
            if len(events) >= 3:  # Pattern threshold
                user_ids = [e.user_id for e in events if e.user_id]
                patterns.append({
                    "action": action,
                    "occurrences": len(events),
                    "unique_users": len(set(user_ids)),
                    "time_span_hours": self._calculate_time_span(events),
                    "risk_assessment": "investigate" if len(events) > 5 else "monitor"
                })
        
        return sorted(patterns, key=lambda x: x["occurrences"], reverse=True)
    
    def _detect_unusual_activities(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect unusual activity patterns."""
        unusual_activities = []
        
        # Analyze user behavior patterns
        user_actions = {}
        for event in events:
            if event.user_id:
                if event.user_id not in user_actions:
                    user_actions[event.user_id] = []
                user_actions[event.user_id].append(event)
        
        # Look for unusual patterns
        for user_id, user_events in user_actions.items():
            # High frequency of failed actions
            failed_events = [e for e in user_events if e.result in ["failure", "error", "denied"]]
            if len(failed_events) > len(user_events) * 0.5 and len(failed_events) > 5:
                unusual_activities.append({
                    "type": "high_failure_rate",
                    "user_id": user_id,
                    "failed_events": len(failed_events),
                    "total_events": len(user_events),
                    "failure_rate": len(failed_events) / len(user_events)
                })
            
            # Unusual time patterns
            event_hours = [datetime.fromisoformat(e.timestamp).hour for e in user_events]
            night_events = [h for h in event_hours if 0 <= h <= 6 or 22 <= h <= 23]
            if len(night_events) > len(event_hours) * 0.7 and len(user_events) > 10:
                unusual_activities.append({
                    "type": "unusual_time_pattern",
                    "user_id": user_id,
                    "night_activity_rate": len(night_events) / len(event_hours),
                    "total_events": len(user_events)
                })
        
        return unusual_activities
    
    def _analyze_system_metrics(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Analyze system metrics for the reporting period."""
        if not metrics:
            return {"status": "no_metrics_available"}
        
        # Calculate averages and peaks
        avg_cpu = sum(m.cpu_usage for m in metrics) / len(metrics)
        avg_memory = sum(m.memory_usage for m in metrics) / len(metrics)
        avg_response_time = sum(m.response_time_p95 for m in metrics) / len(metrics)
        
        max_cpu = max(m.cpu_usage for m in metrics)
        max_memory = max(m.memory_usage for m in metrics)
        max_response_time = max(m.response_time_p95 for m in metrics)
        
        total_security_events = sum(m.security_events for m in metrics)
        total_api_requests = sum(m.api_requests for m in metrics)
        
        return {
            "period_summary": {
                "total_measurements": len(metrics),
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_response_time_p95": avg_response_time
            },
            "peak_usage": {
                "max_cpu_usage": max_cpu,
                "max_memory_usage": max_memory,
                "max_response_time": max_response_time
            },
            "security_summary": {
                "total_security_events": total_security_events,
                "security_events_per_day": total_security_events / max(len(metrics) / 24, 1)
            },
            "performance_summary": {
                "total_api_requests": total_api_requests,
                "avg_requests_per_hour": total_api_requests / max(len(metrics), 1)
            }
        }
    
    def _calculate_time_span(self, events: List[AuditEvent]) -> float:
        """Calculate time span of events in hours."""
        if len(events) < 2:
            return 0.0
        
        timestamps = [datetime.fromisoformat(e.timestamp) for e in events]
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = str(int(time.time() * 1000000))
        return f"audit_{timestamp}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = str(int(time.time() * 1000000))
        return f"session_{timestamp}"
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.audit_dir / f"audit_{today}.jsonl"
    
    def _flush_buffer(self):
        """Flush events buffer to log file."""
        if not self.events_buffer:
            return
        
        try:
            with open(self.current_log_file, 'a') as f:
                for event in self.events_buffer:
                    log_entry = asdict(event)
                    f.write(json.dumps(log_entry) + '\n')
            
            self.events_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}")
    
    def _load_events_for_period(self, start_date: datetime, end_date: datetime) -> List[AuditEvent]:
        """Load audit events for specified time period."""
        events = []
        
        # Determine which log files to read
        current_date = start_date.date()
        end_date_date = end_date.date()
        
        while current_date <= end_date_date:
            log_file = self.audit_dir / f"audit_{current_date}.jsonl"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            event_data = json.loads(line.strip())
                            event = AuditEvent(**event_data)
                            
                            event_time = datetime.fromisoformat(event.timestamp)
                            if start_date <= event_time <= end_date:
                                events.append(event)
                                
                except Exception as e:
                    logger.error(f"Failed to load audit log {log_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return events


# Global audit logger instance
audit_logger = AuditLogger()
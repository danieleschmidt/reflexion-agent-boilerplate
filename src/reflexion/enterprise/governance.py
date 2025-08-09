"""Governance and compliance framework for enterprise reflexion deployments."""

import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging

from ..core.types import ReflexionResult


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    SOC2 = "soc2"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GovernancePolicy:
    """Defines governance policies for reflexion operations."""
    name: str
    description: str
    compliance_standards: List[ComplianceStandard]
    rules: Dict[str, Any]
    enforcement_level: str  # "warning", "blocking", "audit_only"
    created_at: str
    updated_at: str
    version: str


@dataclass
class AuditEvent:
    """Represents an auditable event."""
    event_id: str
    timestamp: str
    event_type: str
    actor: str
    resource: str
    action: str
    outcome: str
    risk_level: RiskLevel
    metadata: Dict[str, Any]
    compliance_tags: List[str]


@dataclass
class ComplianceAssessment:
    """Results of compliance assessment."""
    standard: ComplianceStandard
    assessment_id: str
    timestamp: str
    status: str  # "compliant", "non_compliant", "needs_review"
    findings: List[str]
    recommendations: List[str]
    score: float  # 0.0-1.0


class AuditTrail:
    """Maintains comprehensive audit trail for reflexion operations."""
    
    def __init__(self, storage_path: str = "./audit_trail.json"):
        """Initialize audit trail.
        
        Args:
            storage_path: Path to store audit events
        """
        self.storage_path = storage_path
        self.events: List[AuditEvent] = []
        self.logger = logging.getLogger(__name__)
        
        # Load existing events
        self._load_events()
    
    def record_event(
        self,
        event_type: str,
        actor: str,
        resource: str,
        action: str,
        outcome: str,
        metadata: Dict[str, Any] = None,
        risk_level: RiskLevel = RiskLevel.LOW
    ) -> str:
        """Record an auditable event."""
        event_id = self._generate_event_id(event_type, actor, resource, action)
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            outcome=outcome,
            risk_level=risk_level,
            metadata=metadata or {},
            compliance_tags=self._determine_compliance_tags(event_type, metadata or {})
        )
        
        self.events.append(event)
        self._persist_event(event)
        
        self.logger.info(
            f"Audit event recorded: {event_type} by {actor} on {resource}"
        )
        
        return event_id
    
    def _generate_event_id(self, event_type: str, actor: str, resource: str, action: str) -> str:
        """Generate unique event ID."""
        timestamp = datetime.now().isoformat()
        content = f"{event_type}:{actor}:{resource}:{action}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _determine_compliance_tags(self, event_type: str, metadata: Dict[str, Any]) -> List[str]:
        """Determine compliance tags for event."""
        tags = []
        
        # Data processing events
        if "data" in event_type.lower() or "processing" in event_type.lower():
            tags.extend(["gdpr", "data_processing"])
        
        # Security events
        if "security" in event_type.lower() or "auth" in event_type.lower():
            tags.extend(["security", "access_control"])
        
        # Financial data
        if metadata.get("contains_financial_data", False):
            tags.extend(["sox", "pci_dss"])
        
        # Healthcare data
        if metadata.get("contains_health_data", False):
            tags.append("hipaa")
        
        return tags
    
    def _persist_event(self, event: AuditEvent):
        """Persist event to storage."""
        try:
            # In production, this would use a secure, immutable storage
            with open(self.storage_path, 'a') as f:
                f.write(json.dumps(asdict(event)) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to persist audit event: {e}")
    
    def _load_events(self):
        """Load existing events from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                for line in f:
                    event_data = json.loads(line.strip())
                    event = AuditEvent(**event_data)
                    self.events.append(event)
        except FileNotFoundError:
            pass  # No existing events
        except Exception as e:
            self.logger.error(f"Failed to load audit events: {e}")
    
    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        filtered_events = self.events
        
        if start_time:
            filtered_events = [
                e for e in filtered_events 
                if datetime.fromisoformat(e.timestamp) >= start_time
            ]
        
        if end_time:
            filtered_events = [
                e for e in filtered_events
                if datetime.fromisoformat(e.timestamp) <= end_time
            ]
        
        if event_type:
            filtered_events = [
                e for e in filtered_events
                if e.event_type == event_type
            ]
        
        if actor:
            filtered_events = [
                e for e in filtered_events
                if e.actor == actor
            ]
        
        if risk_level:
            filtered_events = [
                e for e in filtered_events
                if e.risk_level == risk_level
            ]
        
        return filtered_events
    
    def generate_audit_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        events = self.query_events(start_time, end_time)
        
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_events": len(events),
                "event_types": len(set(e.event_type for e in events)),
                "unique_actors": len(set(e.actor for e in events)),
                "risk_distribution": {
                    risk.value: len([e for e in events if e.risk_level == risk])
                    for risk in RiskLevel
                }
            },
            "events_by_type": {},
            "high_risk_events": [
                asdict(e) for e in events 
                if e.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ]
        }
        
        # Group events by type
        for event in events:
            if event.event_type not in report["events_by_type"]:
                report["events_by_type"][event.event_type] = 0
            report["events_by_type"][event.event_type] += 1
        
        return report


class ComplianceMonitor:
    """Monitors reflexion operations for compliance violations."""
    
    def __init__(self, audit_trail: AuditTrail):
        """Initialize compliance monitor.
        
        Args:
            audit_trail: Audit trail for recording compliance events
        """
        self.audit_trail = audit_trail
        self.policies: Dict[str, GovernancePolicy] = {}
        self.assessments: List[ComplianceAssessment] = []
        self.logger = logging.getLogger(__name__)
        
        # Load default policies
        self._load_default_policies()
    
    def _load_default_policies(self):
        """Load default compliance policies."""
        
        # GDPR Data Processing Policy
        gdpr_policy = GovernancePolicy(
            name="gdpr_data_processing",
            description="GDPR compliance for data processing activities",
            compliance_standards=[ComplianceStandard.GDPR],
            rules={
                "data_minimization": True,
                "purpose_limitation": True,
                "consent_required": True,
                "data_retention_days": 730,
                "pseudonymization_required": True
            },
            enforcement_level="blocking",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            version="1.0"
        )
        
        # Security Policy
        security_policy = GovernancePolicy(
            name="security_controls",
            description="Security controls for reflexion operations",
            compliance_standards=[ComplianceStandard.ISO27001, ComplianceStandard.SOC2],
            rules={
                "authentication_required": True,
                "encryption_in_transit": True,
                "encryption_at_rest": True,
                "access_logging": True,
                "session_timeout_minutes": 30
            },
            enforcement_level="blocking",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            version="1.0"
        )
        
        self.policies["gdpr_data_processing"] = gdpr_policy
        self.policies["security_controls"] = security_policy
    
    def assess_compliance(
        self,
        standard: ComplianceStandard,
        context: Dict[str, Any]
    ) -> ComplianceAssessment:
        """Assess compliance for specific standard."""
        assessment_id = hashlib.sha256(
            f"{standard.value}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        findings = []
        recommendations = []
        score = 1.0
        
        # Get relevant policies
        relevant_policies = [
            policy for policy in self.policies.values()
            if standard in policy.compliance_standards
        ]
        
        if not relevant_policies:
            findings.append(f"No policies defined for {standard.value}")
            score = 0.0
        else:
            # Check each policy
            for policy in relevant_policies:
                policy_violations = self._check_policy_compliance(policy, context)
                if policy_violations:
                    findings.extend(policy_violations)
                    score *= 0.8  # Reduce score for each violation
        
        # Generate recommendations
        if findings:
            recommendations.extend([
                "Review and update compliance policies",
                "Implement additional security controls",
                "Conduct regular compliance training"
            ])
        
        # Determine status
        if score >= 0.9:
            status = "compliant"
        elif score >= 0.7:
            status = "needs_review"
        else:
            status = "non_compliant"
        
        assessment = ComplianceAssessment(
            standard=standard,
            assessment_id=assessment_id,
            timestamp=datetime.now().isoformat(),
            status=status,
            findings=findings,
            recommendations=recommendations,
            score=score
        )
        
        self.assessments.append(assessment)
        
        # Record audit event
        self.audit_trail.record_event(
            event_type="compliance_assessment",
            actor="compliance_monitor",
            resource=f"standard_{standard.value}",
            action="assess",
            outcome=status,
            metadata={"assessment_id": assessment_id, "score": score},
            risk_level=RiskLevel.HIGH if status == "non_compliant" else RiskLevel.LOW
        )
        
        return assessment
    
    def _check_policy_compliance(
        self,
        policy: GovernancePolicy,
        context: Dict[str, Any]
    ) -> List[str]:
        """Check compliance with specific policy."""
        violations = []
        
        # Example compliance checks
        if policy.name == "gdpr_data_processing":
            if context.get("processes_personal_data", False):
                if not context.get("consent_obtained", False):
                    violations.append("GDPR: Consent not obtained for personal data processing")
                
                if not context.get("data_pseudonymized", False):
                    violations.append("GDPR: Personal data not pseudonymized")
        
        elif policy.name == "security_controls":
            if not context.get("authenticated", False):
                violations.append("Security: Authentication required")
            
            if not context.get("encrypted", False):
                violations.append("Security: Data encryption required")
        
        return violations
    
    def monitor_reflexion_execution(self, result: ReflexionResult) -> List[str]:
        """Monitor reflexion execution for compliance issues."""
        issues = []
        
        # Check for sensitive data in output
        if self._contains_sensitive_data(result.output):
            issues.append("Output may contain sensitive data")
            
            # Record compliance event
            self.audit_trail.record_event(
                event_type="data_exposure_risk",
                actor="reflexion_agent",
                resource="execution_output",
                action="generate",
                outcome="sensitive_data_detected",
                metadata={"task": result.task},
                risk_level=RiskLevel.HIGH
            )
        
        # Check execution time compliance
        if result.total_time > 300:  # 5 minutes
            issues.append("Execution time exceeds policy limits")
        
        return issues
    
    def _contains_sensitive_data(self, text: str) -> bool:
        """Check if text contains sensitive data."""
        sensitive_patterns = [
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card pattern
            r'\b\d{3}-\d{2}-\d{4}\b',        # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                return True
        
        return False


class GovernanceFramework:
    """Comprehensive governance framework for enterprise reflexion deployments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize governance framework.
        
        Args:
            config: Governance configuration
        """
        self.config = config or {}
        self.audit_trail = AuditTrail()
        self.compliance_monitor = ComplianceMonitor(self.audit_trail)
        self.logger = logging.getLogger(__name__)
        
        # Initialize governance
        self._initialize_governance()
    
    def _initialize_governance(self):
        """Initialize governance components."""
        self.logger.info("Initializing governance framework")
        
        # Record initialization
        self.audit_trail.record_event(
            event_type="governance_init",
            actor="system",
            resource="governance_framework",
            action="initialize",
            outcome="success",
            metadata={"config": self.config}
        )
    
    def enforce_governance(
        self,
        operation: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enforce governance policies for an operation."""
        
        # Record governance enforcement
        self.audit_trail.record_event(
            event_type="governance_enforcement",
            actor="governance_framework",
            resource=operation,
            action="enforce",
            outcome="started",
            metadata=context
        )
        
        # Run compliance assessments
        assessments = {}
        for standard in ComplianceStandard:
            try:
                assessment = self.compliance_monitor.assess_compliance(standard, context)
                assessments[standard.value] = assessment
            except Exception as e:
                self.logger.error(f"Compliance assessment failed for {standard}: {e}")
        
        # Determine enforcement decision
        blocking_violations = [
            a for a in assessments.values()
            if a.status == "non_compliant"
        ]
        
        enforcement_result = {
            "allowed": len(blocking_violations) == 0,
            "assessments": assessments,
            "violations": blocking_violations,
            "recommendations": []
        }
        
        if blocking_violations:
            for violation in blocking_violations:
                enforcement_result["recommendations"].extend(violation.recommendations)
        
        # Record enforcement outcome
        self.audit_trail.record_event(
            event_type="governance_enforcement",
            actor="governance_framework", 
            resource=operation,
            action="enforce",
            outcome="allowed" if enforcement_result["allowed"] else "blocked",
            metadata={"violations": len(blocking_violations)},
            risk_level=RiskLevel.HIGH if blocking_violations else RiskLevel.LOW
        )
        
        return enforcement_result
    
    def generate_governance_report(self) -> Dict[str, Any]:
        """Generate comprehensive governance report."""
        
        # Get recent audit events
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        audit_report = self.audit_trail.generate_audit_report(start_time, end_time)
        
        # Get compliance assessments
        recent_assessments = [
            a for a in self.compliance_monitor.assessments
            if datetime.fromisoformat(a.timestamp) >= start_time
        ]
        
        # Calculate compliance scores
        compliance_scores = {}
        for standard in ComplianceStandard:
            standard_assessments = [
                a for a in recent_assessments
                if a.standard == standard
            ]
            if standard_assessments:
                avg_score = sum(a.score for a in standard_assessments) / len(standard_assessments)
                compliance_scores[standard.value] = avg_score
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "period": audit_report["period"],
            "audit_summary": audit_report["summary"],
            "compliance_scores": compliance_scores,
            "recent_violations": [
                asdict(a) for a in recent_assessments
                if a.status == "non_compliant"
            ],
            "recommendations": [
                "Regular compliance training for all users",
                "Automated policy enforcement",
                "Continuous monitoring of reflexion operations",
                "Regular security assessments"
            ]
        }
        
        return report
"""Tests for enterprise features."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from reflexion.enterprise.governance import (
    GovernanceFramework, ComplianceMonitor, AuditTrail,
    ComplianceStandard, RiskLevel, GovernancePolicy
)
from reflexion.enterprise.multi_tenant import (
    MultiTenantManager, TenantConfig, TenantTier,
    ResourceType, ResourceQuota, ResourceError
)


class TestAuditTrail:
    """Test audit trail functionality."""
    
    @pytest.fixture
    def audit_trail(self, tmp_path):
        """Create audit trail with temporary storage."""
        return AuditTrail(str(tmp_path / "audit.json"))
    
    def test_record_event(self, audit_trail):
        """Test event recording."""
        event_id = audit_trail.record_event(
            event_type="test_event",
            actor="test_user",
            resource="test_resource",
            action="test_action",
            outcome="success"
        )
        
        assert event_id
        assert len(audit_trail.events) == 1
        
        event = audit_trail.events[0]
        assert event.event_type == "test_event"
        assert event.actor == "test_user"
        assert event.resource == "test_resource"
        assert event.action == "test_action"
        assert event.outcome == "success"
    
    def test_query_events(self, audit_trail):
        """Test event querying."""
        # Record multiple events
        audit_trail.record_event("type1", "user1", "resource1", "action1", "success")
        audit_trail.record_event("type2", "user2", "resource2", "action2", "failure")
        audit_trail.record_event("type1", "user1", "resource3", "action3", "success")
        
        # Query by event type
        type1_events = audit_trail.query_events(event_type="type1")
        assert len(type1_events) == 2
        
        # Query by actor
        user1_events = audit_trail.query_events(actor="user1")
        assert len(user1_events) == 2
        
        # Query by risk level
        low_risk_events = audit_trail.query_events(risk_level=RiskLevel.LOW)
        assert len(low_risk_events) == 3  # Default risk level
    
    def test_generate_audit_report(self, audit_trail):
        """Test audit report generation."""
        # Record some events
        audit_trail.record_event("event1", "user1", "resource1", "action1", "success")
        audit_trail.record_event("event2", "user2", "resource2", "action2", "failure", 
                                risk_level=RiskLevel.HIGH)
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=1)
        
        report = audit_trail.generate_audit_report(start_time, end_time)
        
        assert report["summary"]["total_events"] == 2
        assert report["summary"]["risk_distribution"][RiskLevel.HIGH.value] == 1
        assert len(report["high_risk_events"]) == 1


class TestComplianceMonitor:
    """Test compliance monitoring functionality."""
    
    @pytest.fixture
    def compliance_monitor(self, tmp_path):
        """Create compliance monitor with temporary audit trail."""
        audit_trail = AuditTrail(str(tmp_path / "audit.json"))
        return ComplianceMonitor(audit_trail)
    
    def test_initialization(self, compliance_monitor):
        """Test compliance monitor initialization."""
        assert len(compliance_monitor.policies) > 0
        assert "gdpr_data_processing" in compliance_monitor.policies
        assert "security_controls" in compliance_monitor.policies
    
    def test_assess_compliance_gdpr_compliant(self, compliance_monitor):
        """Test GDPR compliance assessment - compliant case."""
        context = {
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_pseudonymized": True,
            "authenticated": True,
            "encrypted": True
        }
        
        assessment = compliance_monitor.assess_compliance(
            ComplianceStandard.GDPR, context
        )
        
        assert assessment.standard == ComplianceStandard.GDPR
        assert assessment.status in ["compliant", "needs_review"]
        assert assessment.score > 0
        assert len(compliance_monitor.assessments) == 1
    
    def test_assess_compliance_gdpr_violation(self, compliance_monitor):
        """Test GDPR compliance assessment - violation case."""
        context = {
            "processes_personal_data": True,
            "consent_obtained": False,  # Violation
            "data_pseudonymized": False,  # Violation
            "authenticated": True,
            "encrypted": True
        }
        
        assessment = compliance_monitor.assess_compliance(
            ComplianceStandard.GDPR, context
        )
        
        assert assessment.status == "non_compliant"
        assert len(assessment.findings) > 0
        assert any("consent" in finding.lower() for finding in assessment.findings)
    
    def test_monitor_reflexion_execution(self, compliance_monitor):
        """Test reflexion execution monitoring."""
        from reflexion.core.types import ReflexionResult
        
        result = ReflexionResult(
            task="test task",
            output="This contains user@example.com email",  # Contains sensitive data
            success=True,
            iterations=1,
            reflections=[],
            total_time=30.0,
            metadata={}
        )
        
        issues = compliance_monitor.monitor_reflexion_execution(result)
        
        assert len(issues) > 0
        assert any("sensitive data" in issue.lower() for issue in issues)


class TestGovernanceFramework:
    """Test governance framework functionality."""
    
    @pytest.fixture
    def governance_framework(self):
        """Create governance framework."""
        return GovernanceFramework({
            "audit_enabled": True,
            "compliance_monitoring": True
        })
    
    def test_initialization(self, governance_framework):
        """Test governance framework initialization."""
        assert governance_framework.audit_trail
        assert governance_framework.compliance_monitor
        assert len(governance_framework.audit_trail.events) > 0  # Initialization event
    
    def test_enforce_governance_allowed(self, governance_framework):
        """Test governance enforcement - allowed case."""
        context = {
            "processes_personal_data": False,
            "authenticated": True,
            "encrypted": True
        }
        
        result = governance_framework.enforce_governance(
            "test_operation", context
        )
        
        assert result["allowed"] == True
        assert "assessments" in result
    
    def test_enforce_governance_blocked(self, governance_framework):
        """Test governance enforcement - blocked case."""
        context = {
            "processes_personal_data": True,
            "consent_obtained": False,
            "data_pseudonymized": False,
            "authenticated": False,
            "encrypted": False
        }
        
        result = governance_framework.enforce_governance(
            "sensitive_operation", context
        )
        
        # Should be blocked due to multiple violations
        assert result["allowed"] == False
        assert len(result["violations"]) > 0
    
    def test_generate_governance_report(self, governance_framework):
        """Test governance report generation."""
        # Perform some operations to generate data
        governance_framework.enforce_governance("op1", {"authenticated": True})
        governance_framework.enforce_governance("op2", {"authenticated": False})
        
        report = governance_framework.generate_governance_report()
        
        assert "report_generated" in report
        assert "compliance_scores" in report
        assert "recommendations" in report


class TestMultiTenantManager:
    """Test multi-tenant management functionality."""
    
    @pytest.fixture
    def tenant_manager(self):
        """Create multi-tenant manager."""
        return MultiTenantManager()
    
    def test_create_tenant(self, tenant_manager):
        """Test tenant creation."""
        tenant = tenant_manager.create_tenant(
            tenant_id="test_tenant",
            name="Test Tenant", 
            tier=TenantTier.PROFESSIONAL,
            settings={"custom_setting": "value"}
        )
        
        assert tenant.tenant_id == "test_tenant"
        assert tenant.name == "Test Tenant"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.active == True
        assert ResourceType.API_CALLS in tenant.quotas
        
        # Check tenant is registered
        assert "test_tenant" in tenant_manager.tenants
    
    def test_create_execution_context(self, tenant_manager):
        """Test execution context creation."""
        # First create a tenant
        tenant_manager.create_tenant("test_tenant", "Test", TenantTier.BASIC)
        
        context = tenant_manager.create_execution_context(
            tenant_id="test_tenant",
            user_id="user123",
            session_id="session456",
            permissions={"read", "write"}
        )
        
        assert context.tenant_id == "test_tenant"
        assert context.user_id == "user123"
        assert context.session_id == "session456"
        assert context.permissions == {"read", "write"}
        assert context.request_id
    
    def test_create_execution_context_invalid_tenant(self, tenant_manager):
        """Test execution context creation with invalid tenant."""
        with pytest.raises(ValueError, match="not found"):
            tenant_manager.create_execution_context(
                tenant_id="nonexistent",
                user_id="user123",
                session_id="session456"
            )
    
    def test_get_tenant_status(self, tenant_manager):
        """Test tenant status retrieval."""
        tenant_manager.create_tenant("test_tenant", "Test", TenantTier.BASIC)
        
        status = tenant_manager.get_tenant_status("test_tenant")
        
        assert status["tenant_config"]["tenant_id"] == "test_tenant"
        assert status["resource_usage"]["tenant_id"] == "test_tenant"
        assert status["active_sessions"] == 0
        assert status["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_execute_with_tenant_context(self, tenant_manager):
        """Test execution with tenant context."""
        from reflexion import ReflexionAgent
        
        # Create tenant and context
        tenant_manager.create_tenant("test_tenant", "Test", TenantTier.BASIC)
        context = tenant_manager.create_execution_context(
            "test_tenant", "user123", "session456"
        )
        
        # Mock agent
        mock_agent = Mock(spec=ReflexionAgent)
        mock_result = Mock()
        mock_result.metadata = {}
        mock_agent.run.return_value = mock_result
        
        # Execute with context
        result = await tenant_manager.execute_with_tenant_context(
            context, mock_agent, "test task"
        )
        
        # Verify agent was called
        mock_agent.run.assert_called_once()
        
        # Verify tenant metadata was added
        assert "tenant_id" in result.metadata
        assert result.metadata["tenant_id"] == "test_tenant"


class TestResourceQuotas:
    """Test resource quota management."""
    
    @pytest.fixture
    def resource_quotas(self, tmp_path):
        """Create resource quotas manager."""
        return MultiTenantManager().quotas
    
    def test_check_quota_within_limit(self, resource_quotas):
        """Test quota check within limits."""
        quota = ResourceQuota(
            ResourceType.API_CALLS,
            limit=100,
            unit="calls",
            period="monthly"
        )
        
        check = resource_quotas.check_quota(
            "test_tenant",
            ResourceType.API_CALLS,
            10,  # Request 10 calls
            quota
        )
        
        assert check["allowed"] == True
        assert check["current_usage"] == 0
        assert check["new_usage"] == 10
        assert check["limit"] == 100
    
    def test_check_quota_exceeds_limit(self, resource_quotas):
        """Test quota check exceeding limits."""
        quota = ResourceQuota(
            ResourceType.API_CALLS,
            limit=5,
            unit="calls", 
            period="monthly"
        )
        
        check = resource_quotas.check_quota(
            "test_tenant",
            ResourceType.API_CALLS,
            10,  # Request 10 calls (exceeds limit of 5)
            quota
        )
        
        assert check["allowed"] == False
        assert "exceeded" in check["message"].lower()
    
    def test_record_usage(self, resource_quotas):
        """Test usage recording."""
        quota = ResourceQuota(
            ResourceType.API_CALLS,
            limit=100,
            unit="calls",
            period="monthly"
        )
        
        # Record usage
        resource_quotas.record_usage(
            "test_tenant",
            ResourceType.API_CALLS,
            5,
            quota
        )
        
        # Check that usage was recorded
        usage = resource_quotas._get_current_usage(
            "test_tenant",
            ResourceType.API_CALLS,
            "monthly"
        )
        
        assert usage == 5.0
    
    def test_get_tenant_usage_summary(self, resource_quotas):
        """Test usage summary generation."""
        quota = ResourceQuota(
            ResourceType.API_CALLS,
            limit=100,
            unit="calls",
            period="monthly"
        )
        
        # Record some usage
        resource_quotas.record_usage("test_tenant", ResourceType.API_CALLS, 25, quota)
        
        summary = resource_quotas.get_tenant_usage_summary("test_tenant")
        
        assert summary["tenant_id"] == "test_tenant"
        assert "resources" in summary
        
        if ResourceType.API_CALLS.value in summary["resources"]:
            api_usage = summary["resources"][ResourceType.API_CALLS.value]
            assert api_usage["usage"] == 25
            assert api_usage["limit"] == 100
            assert api_usage["usage_percentage"] == 25.0


@pytest.mark.integration 
class TestEnterpriseIntegration:
    """Integration tests for enterprise features."""
    
    def test_governance_with_multi_tenant(self):
        """Test governance integration with multi-tenant features."""
        # Create governance framework
        governance = GovernanceFramework()
        
        # Create multi-tenant manager  
        tenant_manager = MultiTenantManager()
        
        # Create tenant
        tenant_manager.create_tenant(
            "enterprise_tenant",
            "Enterprise Client",
            TenantTier.ENTERPRISE
        )
        
        # Test governance enforcement for tenant operation
        tenant_context = {
            "tenant_id": "enterprise_tenant",
            "processes_personal_data": True,
            "consent_obtained": True,
            "data_pseudonymized": True,
            "authenticated": True,
            "encrypted": True
        }
        
        enforcement_result = governance.enforce_governance(
            "tenant_operation",
            tenant_context
        )
        
        assert enforcement_result["allowed"] == True
        
        # Verify audit event was recorded
        audit_events = governance.audit_trail.query_events(
            event_type="governance_enforcement"
        )
        assert len(audit_events) > 0
    
    def test_compliance_monitoring_integration(self):
        """Test compliance monitoring across enterprise features."""
        from reflexion.core.types import ReflexionResult
        
        # Create governance framework with monitoring
        governance = GovernanceFramework()
        
        # Create a reflexion result that might have compliance issues
        result = ReflexionResult(
            task="Process customer data",
            output="Customer john.doe@company.com has been processed",  # Contains email
            success=True,
            iterations=1,
            reflections=[],
            total_time=45.0,
            metadata={"contains_personal_data": True}
        )
        
        # Monitor for compliance
        issues = governance.compliance_monitor.monitor_reflexion_execution(result)
        
        # Should detect sensitive data
        assert len(issues) > 0
        
        # Check that audit events were recorded
        sensitive_data_events = governance.audit_trail.query_events(
            event_type="data_exposure_risk"
        )
        assert len(sensitive_data_events) > 0
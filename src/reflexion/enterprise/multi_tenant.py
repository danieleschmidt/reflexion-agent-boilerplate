"""Multi-tenant support for enterprise reflexion deployments."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging

from ..core.types import ReflexionResult
from ..core.agent import ReflexionAgent


class TenantTier(Enum):
    """Tenant service tiers."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    PREMIUM = "premium"


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU_TIME = "cpu_time"
    MEMORY = "memory"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    CONCURRENT_EXECUTIONS = "concurrent_executions"
    MONTHLY_EXECUTIONS = "monthly_executions"


@dataclass
class ResourceQuota:
    """Resource quota configuration."""
    resource_type: ResourceType
    limit: float
    unit: str
    period: str  # "daily", "monthly", "concurrent"
    soft_limit: Optional[float] = None  # Warning threshold


@dataclass
class TenantConfig:
    """Configuration for a tenant."""
    tenant_id: str
    name: str
    tier: TenantTier
    created_at: str
    quotas: Dict[ResourceType, ResourceQuota]
    settings: Dict[str, Any]
    metadata: Dict[str, Any]
    active: bool = True


@dataclass
class ResourceUsage:
    """Tracks resource usage for a tenant."""
    tenant_id: str
    resource_type: ResourceType
    usage: float
    limit: float
    period_start: str
    period_end: str
    last_updated: str


@dataclass
class TenantExecutionContext:
    """Execution context for tenant operations."""
    tenant_id: str
    user_id: str
    session_id: str
    request_id: str
    permissions: Set[str]
    quotas: Dict[ResourceType, ResourceQuota]
    current_usage: Dict[ResourceType, float]


class TenantIsolation:
    """Ensures proper isolation between tenants."""
    
    def __init__(self):
        """Initialize tenant isolation."""
        self.tenant_namespaces: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_tenant_namespace(self, tenant_id: str) -> str:
        """Create isolated namespace for tenant."""
        namespace = f"tenant_{hashlib.sha256(tenant_id.encode()).hexdigest()[:8]}"
        self.tenant_namespaces[tenant_id] = namespace
        
        self.logger.info(f"Created namespace {namespace} for tenant {tenant_id}")
        return namespace
    
    def get_tenant_namespace(self, tenant_id: str) -> str:
        """Get namespace for tenant."""
        if tenant_id not in self.tenant_namespaces:
            return self.create_tenant_namespace(tenant_id)
        return self.tenant_namespaces[tenant_id]
    
    def isolate_data_access(self, tenant_id: str, data_path: str) -> str:
        """Create tenant-specific data path."""
        namespace = self.get_tenant_namespace(tenant_id)
        return f"{namespace}/{data_path}"
    
    def validate_cross_tenant_access(
        self, 
        requesting_tenant: str, 
        target_resource: str,
        resource_tenant: str
    ) -> bool:
        """Validate if cross-tenant access is allowed."""
        
        # Basic rule: no cross-tenant access unless explicitly allowed
        if requesting_tenant != resource_tenant:
            self.logger.warning(
                f"Cross-tenant access denied: {requesting_tenant} -> {resource_tenant}"
            )
            return False
        
        return True


class ResourceQuotas:
    """Manages resource quotas for tenants."""
    
    def __init__(self, storage_path: str = "./tenant_quotas.json"):
        """Initialize resource quota management.
        
        Args:
            storage_path: Path to store quota data
        """
        self.storage_path = storage_path
        self.usage_data: Dict[str, Dict[ResourceType, ResourceUsage]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing usage data
        self._load_usage_data()
    
    def _load_usage_data(self):
        """Load usage data from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            for tenant_id, resources in data.items():
                self.usage_data[tenant_id] = {}
                for resource_type_str, usage_data in resources.items():
                    resource_type = ResourceType(resource_type_str)
                    self.usage_data[tenant_id][resource_type] = ResourceUsage(**usage_data)
                    
        except FileNotFoundError:
            pass  # No existing data
        except Exception as e:
            self.logger.error(f"Failed to load usage data: {e}")
    
    def _save_usage_data(self):
        """Save usage data to storage."""
        try:
            data = {}
            for tenant_id, resources in self.usage_data.items():
                data[tenant_id] = {}
                for resource_type, usage in resources.items():
                    data[tenant_id][resource_type.value] = asdict(usage)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save usage data: {e}")
    
    def check_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        requested_amount: float,
        quota: ResourceQuota
    ) -> Dict[str, Any]:
        """Check if resource usage is within quota."""
        
        # Get current usage
        current_usage = self._get_current_usage(tenant_id, resource_type, quota.period)
        
        # Calculate new usage
        new_usage = current_usage + requested_amount
        
        # Check against limits
        quota_check = {
            "allowed": True,
            "current_usage": current_usage,
            "requested": requested_amount,
            "new_usage": new_usage,
            "limit": quota.limit,
            "usage_percentage": (new_usage / quota.limit) * 100 if quota.limit > 0 else 0,
            "warning": False,
            "message": ""
        }
        
        # Check hard limit
        if new_usage > quota.limit:
            quota_check["allowed"] = False
            quota_check["message"] = f"Quota exceeded for {resource_type.value}"
        
        # Check soft limit
        elif quota.soft_limit and new_usage > quota.soft_limit:
            quota_check["warning"] = True
            quota_check["message"] = f"Approaching quota limit for {resource_type.value}"
        
        return quota_check
    
    def _get_current_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        period: str
    ) -> float:
        """Get current usage for tenant and resource type."""
        
        if tenant_id not in self.usage_data:
            self.usage_data[tenant_id] = {}
        
        if resource_type not in self.usage_data[tenant_id]:
            return 0.0
        
        usage = self.usage_data[tenant_id][resource_type]
        
        # Check if usage data is current for the period
        if not self._is_usage_current(usage, period):
            # Reset usage for new period
            self._reset_usage_for_period(tenant_id, resource_type, period)
            return 0.0
        
        return usage.usage
    
    def _is_usage_current(self, usage: ResourceUsage, period: str) -> bool:
        """Check if usage data is current for the given period."""
        now = datetime.now()
        period_start = datetime.fromisoformat(usage.period_start)
        
        if period == "daily":
            return period_start.date() == now.date()
        elif period == "monthly":
            return (period_start.year == now.year and 
                   period_start.month == now.month)
        elif period == "concurrent":
            return True  # Always current for concurrent usage
        
        return False
    
    def _reset_usage_for_period(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        period: str
    ):
        """Reset usage data for new period."""
        now = datetime.now()
        
        if period == "daily":
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
        elif period == "monthly":
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Next month
            if period_start.month == 12:
                period_end = period_start.replace(year=period_start.year + 1, month=1)
            else:
                period_end = period_start.replace(month=period_start.month + 1)
        else:
            period_start = now
            period_end = now + timedelta(days=30)  # Default
        
        self.usage_data[tenant_id][resource_type] = ResourceUsage(
            tenant_id=tenant_id,
            resource_type=resource_type,
            usage=0.0,
            limit=0.0,  # Will be set when quota is checked
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat(),
            last_updated=now.isoformat()
        )
    
    def record_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: float,
        quota: ResourceQuota
    ):
        """Record resource usage for tenant."""
        
        current_usage = self._get_current_usage(tenant_id, resource_type, quota.period)
        new_usage = current_usage + amount
        
        # Update usage
        now = datetime.now()
        if resource_type == ResourceType.CONCURRENT_EXECUTIONS:
            # For concurrent resources, set absolute value
            new_usage = amount
        
        self.usage_data[tenant_id][resource_type] = ResourceUsage(
            tenant_id=tenant_id,
            resource_type=resource_type,
            usage=new_usage,
            limit=quota.limit,
            period_start=self.usage_data[tenant_id][resource_type].period_start,
            period_end=self.usage_data[tenant_id][resource_type].period_end,
            last_updated=now.isoformat()
        )
        
        # Persist changes
        self._save_usage_data()
        
        self.logger.debug(
            f"Recorded {amount} {resource_type.value} usage for tenant {tenant_id}"
        )
    
    def get_tenant_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage summary for tenant."""
        if tenant_id not in self.usage_data:
            return {"tenant_id": tenant_id, "resources": {}}
        
        summary = {
            "tenant_id": tenant_id,
            "resources": {},
            "last_updated": datetime.now().isoformat()
        }
        
        for resource_type, usage in self.usage_data[tenant_id].items():
            summary["resources"][resource_type.value] = {
                "usage": usage.usage,
                "limit": usage.limit,
                "usage_percentage": (usage.usage / usage.limit * 100) if usage.limit > 0 else 0,
                "period_start": usage.period_start,
                "period_end": usage.period_end
            }
        
        return summary


class MultiTenantManager:
    """Manages multi-tenant reflexion deployments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize multi-tenant manager.
        
        Args:
            config: Multi-tenant configuration
        """
        self.config = config or {}
        self.tenants: Dict[str, TenantConfig] = {}
        self.isolation = TenantIsolation()
        self.quotas = ResourceQuotas()
        self.active_sessions: Dict[str, TenantExecutionContext] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load default tier configurations
        self._load_tier_configurations()
    
    def _load_tier_configurations(self):
        """Load default tier configurations."""
        self.tier_configs = {
            TenantTier.BASIC: {
                ResourceType.API_CALLS: ResourceQuota(
                    ResourceType.API_CALLS, 1000, "calls", "monthly"
                ),
                ResourceType.CONCURRENT_EXECUTIONS: ResourceQuota(
                    ResourceType.CONCURRENT_EXECUTIONS, 2, "executions", "concurrent"
                ),
                ResourceType.CPU_TIME: ResourceQuota(
                    ResourceType.CPU_TIME, 3600, "seconds", "monthly"
                )
            },
            TenantTier.PROFESSIONAL: {
                ResourceType.API_CALLS: ResourceQuota(
                    ResourceType.API_CALLS, 10000, "calls", "monthly"
                ),
                ResourceType.CONCURRENT_EXECUTIONS: ResourceQuota(
                    ResourceType.CONCURRENT_EXECUTIONS, 10, "executions", "concurrent"
                ),
                ResourceType.CPU_TIME: ResourceQuota(
                    ResourceType.CPU_TIME, 36000, "seconds", "monthly"
                )
            },
            TenantTier.ENTERPRISE: {
                ResourceType.API_CALLS: ResourceQuota(
                    ResourceType.API_CALLS, 100000, "calls", "monthly"
                ),
                ResourceType.CONCURRENT_EXECUTIONS: ResourceQuota(
                    ResourceType.CONCURRENT_EXECUTIONS, 50, "executions", "concurrent"
                ),
                ResourceType.CPU_TIME: ResourceQuota(
                    ResourceType.CPU_TIME, 360000, "seconds", "monthly"
                )
            }
        }
    
    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier,
        settings: Dict[str, Any] = None
    ) -> TenantConfig:
        """Create new tenant."""
        
        if tenant_id in self.tenants:
            raise ValueError(f"Tenant {tenant_id} already exists")
        
        # Get quotas for tier
        quotas = self.tier_configs.get(tier, {})
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            created_at=datetime.now().isoformat(),
            quotas=quotas,
            settings=settings or {},
            metadata={},
            active=True
        )
        
        self.tenants[tenant_id] = tenant_config
        
        # Create isolated namespace
        self.isolation.create_tenant_namespace(tenant_id)
        
        self.logger.info(f"Created tenant {tenant_id} with tier {tier.value}")
        
        return tenant_config
    
    def create_execution_context(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        permissions: Set[str] = None
    ) -> TenantExecutionContext:
        """Create execution context for tenant operation."""
        
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        tenant = self.tenants[tenant_id]
        if not tenant.active:
            raise ValueError(f"Tenant {tenant_id} is not active")
        
        request_id = hashlib.sha256(
            f"{tenant_id}:{user_id}:{session_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Get current resource usage
        current_usage = {}
        for resource_type in tenant.quotas:
            current_usage[resource_type] = self.quotas._get_current_usage(
                tenant_id, resource_type, tenant.quotas[resource_type].period
            )
        
        context = TenantExecutionContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            permissions=permissions or set(),
            quotas=tenant.quotas,
            current_usage=current_usage
        )
        
        self.active_sessions[request_id] = context
        
        return context
    
    async def execute_with_tenant_context(
        self,
        context: TenantExecutionContext,
        agent: ReflexionAgent,
        task: str,
        **kwargs
    ) -> ReflexionResult:
        """Execute reflexion with tenant context and quota enforcement."""
        
        # Pre-execution quota checks
        quota_checks = await self._check_execution_quotas(context)
        
        for check in quota_checks.values():
            if not check["allowed"]:
                raise ResourceError(f"Quota exceeded: {check['message']}")
        
        # Record concurrent execution start
        self.quotas.record_usage(
            context.tenant_id,
            ResourceType.CONCURRENT_EXECUTIONS,
            1,  # Add one concurrent execution
            context.quotas[ResourceType.CONCURRENT_EXECUTIONS]
        )
        
        try:
            # Execute with isolation
            isolated_kwargs = self._apply_tenant_isolation(context, kwargs)
            
            # Execute task
            start_time = datetime.now()
            result = agent.run(task, **isolated_kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record resource usage
            await self._record_execution_usage(context, execution_time, result)
            
            # Add tenant metadata to result
            result.metadata.update({
                "tenant_id": context.tenant_id,
                "request_id": context.request_id,
                "execution_time": execution_time
            })
            
            return result
            
        finally:
            # Record concurrent execution end
            current_concurrent = self.quotas._get_current_usage(
                context.tenant_id,
                ResourceType.CONCURRENT_EXECUTIONS,
                "concurrent"
            )
            
            self.quotas.record_usage(
                context.tenant_id,
                ResourceType.CONCURRENT_EXECUTIONS,
                max(0, current_concurrent - 1),  # Remove one concurrent execution
                context.quotas[ResourceType.CONCURRENT_EXECUTIONS]
            )
            
            # Clean up session
            if context.request_id in self.active_sessions:
                del self.active_sessions[context.request_id]
    
    async def _check_execution_quotas(
        self,
        context: TenantExecutionContext
    ) -> Dict[ResourceType, Dict[str, Any]]:
        """Check quotas before execution."""
        checks = {}
        
        # Check API call quota
        if ResourceType.API_CALLS in context.quotas:
            checks[ResourceType.API_CALLS] = self.quotas.check_quota(
                context.tenant_id,
                ResourceType.API_CALLS,
                1,  # One API call
                context.quotas[ResourceType.API_CALLS]
            )
        
        # Check concurrent execution quota
        if ResourceType.CONCURRENT_EXECUTIONS in context.quotas:
            checks[ResourceType.CONCURRENT_EXECUTIONS] = self.quotas.check_quota(
                context.tenant_id,
                ResourceType.CONCURRENT_EXECUTIONS,
                1,  # One more concurrent execution
                context.quotas[ResourceType.CONCURRENT_EXECUTIONS]
            )
        
        return checks
    
    def _apply_tenant_isolation(
        self,
        context: TenantExecutionContext,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tenant isolation to execution parameters."""
        isolated_kwargs = kwargs.copy()
        
        # Isolate file paths
        if "memory_path" in isolated_kwargs:
            isolated_kwargs["memory_path"] = self.isolation.isolate_data_access(
                context.tenant_id, isolated_kwargs["memory_path"]
            )
        
        # Add tenant context to metadata
        if "metadata" not in isolated_kwargs:
            isolated_kwargs["metadata"] = {}
        
        isolated_kwargs["metadata"]["tenant_id"] = context.tenant_id
        
        return isolated_kwargs
    
    async def _record_execution_usage(
        self,
        context: TenantExecutionContext,
        execution_time: float,
        result: ReflexionResult
    ):
        """Record resource usage after execution."""
        
        # Record API call
        if ResourceType.API_CALLS in context.quotas:
            self.quotas.record_usage(
                context.tenant_id,
                ResourceType.API_CALLS,
                1,
                context.quotas[ResourceType.API_CALLS]
            )
        
        # Record CPU time
        if ResourceType.CPU_TIME in context.quotas:
            self.quotas.record_usage(
                context.tenant_id,
                ResourceType.CPU_TIME,
                execution_time,
                context.quotas[ResourceType.CPU_TIME]
            )
        
        self.logger.debug(
            f"Recorded execution usage for tenant {context.tenant_id}: "
            f"{execution_time:.2f}s CPU time"
        )
    
    def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive status for tenant."""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")
        
        tenant = self.tenants[tenant_id]
        usage_summary = self.quotas.get_tenant_usage_summary(tenant_id)
        
        # Count active sessions
        active_sessions = len([
            ctx for ctx in self.active_sessions.values()
            if ctx.tenant_id == tenant_id
        ])
        
        return {
            "tenant_config": asdict(tenant),
            "resource_usage": usage_summary,
            "active_sessions": active_sessions,
            "namespace": self.isolation.get_tenant_namespace(tenant_id),
            "status": "active" if tenant.active else "inactive"
        }


class ResourceError(Exception):
    """Exception raised when resource limits are exceeded."""
    pass
"""Global Configuration System for Autonomous SDLC.

This module provides comprehensive configuration management for global deployments,
including region-specific settings, compliance requirements, and cultural adaptations.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reflexion.i18n import SupportedLanguage, translation_manager


class DeploymentRegion(Enum):
    """Supported deployment regions with specific requirements."""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe" 
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    CHINA = "china"  # Special compliance requirements
    
    
class ComplianceStandard(Enum):
    """Compliance standards for different regions."""
    GDPR = "gdpr"  # European General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # Information security management
    SOC2 = "soc2"  # Service Organization Control 2


@dataclass
class RegionalSettings:
    """Regional configuration settings."""
    region: DeploymentRegion
    primary_language: SupportedLanguage
    fallback_languages: List[SupportedLanguage] = field(default_factory=list)
    timezone: str = "UTC"
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    data_residency_required: bool = False
    encryption_required: bool = True
    audit_logging_required: bool = False
    
    # Cultural settings
    work_week_start: int = 1  # Monday = 1, Sunday = 7
    business_hours_start: str = "09:00"
    business_hours_end: str = "17:00"
    holidays: List[str] = field(default_factory=list)
    
    # Technical settings
    preferred_cloud_regions: List[str] = field(default_factory=list)
    network_restrictions: Dict[str, Any] = field(default_factory=dict)
    storage_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration for global deployments."""
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_rotation_days: int = 90
    password_policy: Dict[str, Any] = field(default_factory=dict)
    session_timeout_minutes: int = 30
    max_failed_login_attempts: int = 5
    require_mfa: bool = False
    allowed_auth_methods: List[str] = field(default_factory=list)
    ip_whitelist: List[str] = field(default_factory=list)
    audit_all_actions: bool = False
    data_retention_days: int = 365
    anonymization_required: bool = False


@dataclass
class PerformanceConfig:
    """Performance configuration for different regions."""
    max_concurrent_tasks: int = 100
    task_timeout_seconds: int = 300
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    cache_ttl_seconds: int = 3600
    batch_size: int = 50
    connection_pool_size: int = 10
    rate_limit_per_minute: int = 1000
    memory_limit_mb: int = 2048
    cpu_limit_cores: float = 2.0


@dataclass
class MonitoringConfig:
    """Monitoring configuration for global deployments."""
    metrics_retention_days: int = 30
    log_level: str = "INFO"
    enable_distributed_tracing: bool = True
    alert_notification_channels: List[str] = field(default_factory=list)
    health_check_interval_seconds: int = 60
    performance_sampling_rate: float = 0.1
    error_reporting_enabled: bool = True
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class GlobalConfigManager:
    """Manages global configuration for autonomous SDLC deployments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config/global_config.json")
        self.regional_configs: Dict[str, RegionalSettings] = {}
        self.security_config = SecurityConfig()
        self.performance_config = PerformanceConfig()
        self.monitoring_config = MonitoringConfig()
        
        # Load configuration
        self._load_default_regional_configs()
        self._load_config_file()
    
    def _load_default_regional_configs(self):
        """Load default regional configurations."""
        # North America
        self.regional_configs[DeploymentRegion.NORTH_AMERICA.value] = RegionalSettings(
            region=DeploymentRegion.NORTH_AMERICA,
            primary_language=SupportedLanguage.ENGLISH,
            timezone="America/New_York",
            compliance_standards=[ComplianceStandard.SOX, ComplianceStandard.SOC2],
            preferred_cloud_regions=["us-east-1", "us-west-2", "ca-central-1"],
            holidays=["2025-01-01", "2025-07-04", "2025-12-25"]
        )
        
        # Europe
        self.regional_configs[DeploymentRegion.EUROPE.value] = RegionalSettings(
            region=DeploymentRegion.EUROPE,
            primary_language=SupportedLanguage.ENGLISH,
            fallback_languages=[SupportedLanguage.GERMAN, SupportedLanguage.FRENCH],
            timezone="Europe/London",
            compliance_standards=[ComplianceStandard.GDPR, ComplianceStandard.ISO27001],
            data_residency_required=True,
            audit_logging_required=True,
            preferred_cloud_regions=["eu-west-1", "eu-central-1"],
            holidays=["2025-01-01", "2025-12-25"]
        )
        
        # Asia Pacific
        self.regional_configs[DeploymentRegion.ASIA_PACIFIC.value] = RegionalSettings(
            region=DeploymentRegion.ASIA_PACIFIC,
            primary_language=SupportedLanguage.ENGLISH,
            fallback_languages=[SupportedLanguage.JAPANESE, SupportedLanguage.KOREAN],
            timezone="Asia/Tokyo",
            work_week_start=1,  # Monday
            business_hours_start="10:00",
            business_hours_end="18:00",
            preferred_cloud_regions=["ap-northeast-1", "ap-southeast-1"],
            holidays=["2025-01-01", "2025-05-01"]
        )
        
        # Latin America
        self.regional_configs[DeploymentRegion.LATIN_AMERICA.value] = RegionalSettings(
            region=DeploymentRegion.LATIN_AMERICA,
            primary_language=SupportedLanguage.SPANISH,
            fallback_languages=[SupportedLanguage.PORTUGUESE, SupportedLanguage.ENGLISH],
            timezone="America/Sao_Paulo",
            currency="BRL",
            preferred_cloud_regions=["sa-east-1"],
            holidays=["2025-01-01", "2025-09-07", "2025-12-25"]
        )
        
        # China
        self.regional_configs[DeploymentRegion.CHINA.value] = RegionalSettings(
            region=DeploymentRegion.CHINA,
            primary_language=SupportedLanguage.CHINESE_SIMPLIFIED,
            fallback_languages=[SupportedLanguage.ENGLISH],
            timezone="Asia/Shanghai",
            currency="CNY",
            data_residency_required=True,
            audit_logging_required=True,
            network_restrictions={
                "require_local_certification": True,
                "blocked_external_services": ["googleapis.com", "github.com"]
            },
            preferred_cloud_regions=["cn-north-1", "cn-northwest-1"],
            holidays=["2025-01-01", "2025-02-12", "2025-10-01"]
        )
    
    def _load_config_file(self):
        """Load configuration from file if it exists."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load regional configurations
                if "regional_configs" in config_data:
                    for region_key, config in config_data["regional_configs"].items():
                        self._load_regional_config(region_key, config)
                
                # Load security configuration
                if "security" in config_data:
                    self._load_security_config(config_data["security"])
                
                # Load performance configuration
                if "performance" in config_data:
                    self._load_performance_config(config_data["performance"])
                
                # Load monitoring configuration
                if "monitoring" in config_data:
                    self._load_monitoring_config(config_data["monitoring"])
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load config file {self.config_path}: {e}")
    
    def _load_regional_config(self, region_key: str, config_data: dict):
        """Load regional configuration from data."""
        if region_key in self.regional_configs:
            existing = self.regional_configs[region_key]
            
            # Update fields from config data
            for key, value in config_data.items():
                if hasattr(existing, key):
                    # Handle enum conversions
                    if key == "primary_language" and isinstance(value, str):
                        try:
                            value = SupportedLanguage(value)
                        except ValueError:
                            continue
                    elif key == "compliance_standards" and isinstance(value, list):
                        try:
                            value = [ComplianceStandard(std) for std in value]
                        except ValueError:
                            continue
                    
                    setattr(existing, key, value)
    
    def _load_security_config(self, config_data: dict):
        """Load security configuration from data."""
        for key, value in config_data.items():
            if hasattr(self.security_config, key):
                setattr(self.security_config, key, value)
    
    def _load_performance_config(self, config_data: dict):
        """Load performance configuration from data."""
        for key, value in config_data.items():
            if hasattr(self.performance_config, key):
                setattr(self.performance_config, key, value)
    
    def _load_monitoring_config(self, config_data: dict):
        """Load monitoring configuration from data."""
        for key, value in config_data.items():
            if hasattr(self.monitoring_config, key):
                setattr(self.monitoring_config, key, value)
    
    def get_regional_config(self, region: Union[DeploymentRegion, str]) -> Optional[RegionalSettings]:
        """Get regional configuration."""
        if isinstance(region, DeploymentRegion):
            region = region.value
        
        return self.regional_configs.get(region)
    
    def set_active_region(self, region: Union[DeploymentRegion, str]):
        """Set the active region and update translation manager."""
        regional_config = self.get_regional_config(region)
        
        if regional_config:
            # Update translation manager language
            translation_manager.set_language(regional_config.primary_language)
            
            # Store active region
            os.environ["AUTONOMOUS_SDLC_REGION"] = regional_config.region.value
            
            print(f"Active region set to: {regional_config.region.value}")
            print(f"Primary language: {regional_config.primary_language.value}")
        else:
            raise ValueError(f"Unknown region: {region}")
    
    def validate_compliance(self, region: Union[DeploymentRegion, str]) -> Dict[str, bool]:
        """Validate compliance requirements for a region."""
        regional_config = self.get_regional_config(region)
        
        if not regional_config:
            return {"error": "Unknown region"}
        
        compliance_checks = {}
        
        for standard in regional_config.compliance_standards:
            if standard == ComplianceStandard.GDPR:
                compliance_checks["gdpr"] = (
                    regional_config.data_residency_required and
                    self.security_config.encryption_at_rest and
                    self.security_config.audit_all_actions
                )
            elif standard == ComplianceStandard.HIPAA:
                compliance_checks["hipaa"] = (
                    self.security_config.encryption_at_rest and
                    self.security_config.encryption_in_transit and
                    self.security_config.audit_all_actions
                )
            elif standard == ComplianceStandard.SOX:
                compliance_checks["sox"] = (
                    self.security_config.audit_all_actions and
                    self.monitoring_config.error_reporting_enabled
                )
            # Add more compliance checks as needed
        
        return compliance_checks
    
    def generate_deployment_config(self, region: Union[DeploymentRegion, str]) -> Dict[str, Any]:
        """Generate deployment configuration for a region."""
        regional_config = self.get_regional_config(region)
        
        if not regional_config:
            raise ValueError(f"Unknown region: {region}")
        
        deployment_config = {
            "region": regional_config.region.value,
            "language": regional_config.primary_language.value,
            "timezone": regional_config.timezone,
            "compliance_standards": [std.value for std in regional_config.compliance_standards],
            "security": {
                "encryption_at_rest": self.security_config.encryption_at_rest,
                "encryption_in_transit": self.security_config.encryption_in_transit,
                "audit_logging": regional_config.audit_logging_required,
                "data_residency": regional_config.data_residency_required,
            },
            "performance": {
                "max_concurrent_tasks": self.performance_config.max_concurrent_tasks,
                "task_timeout_seconds": self.performance_config.task_timeout_seconds,
                "memory_limit_mb": self.performance_config.memory_limit_mb,
            },
            "monitoring": {
                "log_level": self.monitoring_config.log_level,
                "metrics_retention_days": self.monitoring_config.metrics_retention_days,
                "health_check_interval": self.monitoring_config.health_check_interval_seconds,
            },
            "cloud": {
                "preferred_regions": regional_config.preferred_cloud_regions,
                "network_restrictions": regional_config.network_restrictions,
            }
        }
        
        return deployment_config
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        output_path = Path(output_path or self.config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "regional_configs": {},
            "security": self._serialize_dataclass(self.security_config),
            "performance": self._serialize_dataclass(self.performance_config),
            "monitoring": self._serialize_dataclass(self.monitoring_config),
        }
        
        # Serialize regional configurations
        for region_key, config in self.regional_configs.items():
            config_data["regional_configs"][region_key] = self._serialize_regional_config(config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {output_path}")
    
    def _serialize_dataclass(self, obj) -> Dict[str, Any]:
        """Serialize a dataclass to dictionary."""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.value for item in value]
            else:
                result[key] = value
        return result
    
    def _serialize_regional_config(self, config: RegionalSettings) -> Dict[str, Any]:
        """Serialize regional configuration to dictionary."""
        result = {}
        for key, value in config.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.value for item in value]
            else:
                result[key] = value
        return result


# Global configuration manager instance
global_config = GlobalConfigManager()


def get_current_region_config() -> Optional[RegionalSettings]:
    """Get configuration for the current active region."""
    active_region = os.environ.get("AUTONOMOUS_SDLC_REGION")
    if active_region:
        return global_config.get_regional_config(active_region)
    return None


def setup_global_deployment(region: Union[DeploymentRegion, str]):
    """Setup global deployment for a specific region."""
    global_config.set_active_region(region)
    
    # Validate compliance
    compliance_status = global_config.validate_compliance(region)
    print(f"Compliance validation: {compliance_status}")
    
    # Generate deployment configuration
    deployment_config = global_config.generate_deployment_config(region)
    print(f"Deployment configuration generated for {region}")
    
    return deployment_config


__all__ = [
    'DeploymentRegion',
    'ComplianceStandard',
    'RegionalSettings',
    'SecurityConfig',
    'PerformanceConfig',
    'MonitoringConfig',
    'GlobalConfigManager',
    'global_config',
    'get_current_region_config',
    'setup_global_deployment'
]
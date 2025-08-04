"""Production deployment configuration for reflexion agents."""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Region(Enum):
    """Supported global regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    api_rate_limit: int = 1000  # requests per minute
    max_request_size: int = 1024 * 1024  # 1MB
    cors_origins: List[str] = None
    require_api_key: bool = True
    jwt_secret_key: Optional[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["https://*.your-domain.com"]


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics_retention_days: int = 30
    traces_sample_rate: float = 0.1
    health_check_interval: int = 30  # seconds
    alert_email: Optional[str] = None


@dataclass
class I18nConfig:
    """Internationalization configuration."""
    default_language: str = "en"
    supported_languages: List[str] = None
    timezone: str = "UTC"
    date_format: str = "ISO"
    number_format: str = "US"
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "ja", "zh-CN"]


@dataclass
class ComplianceConfig:
    """Compliance and regulatory configuration."""
    gdpr_enabled: bool = True
    ccpa_enabled: bool = True
    pdpa_enabled: bool = True
    data_retention_days: int = 365
    audit_logging: bool = True
    anonymize_pii: bool = True
    consent_required: bool = True


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    environment: DeploymentEnvironment
    region: Region
    scaling: ScalingConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    i18n: I18nConfig
    compliance: ComplianceConfig
    
    # Application settings
    app_name: str = "reflexion-agent-service"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database settings
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # LLM provider settings
    llm_providers: Dict[str, Dict[str, str]] = None
    
    # Custom settings
    custom: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.llm_providers is None:
            self.llm_providers = {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "base_url": "https://api.openai.com/v1"
                },
                "anthropic": {
                    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                    "base_url": "https://api.anthropic.com"
                }
            }
        
        if self.custom is None:
            self.custom = {}
        
        # Environment-specific adjustments
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            self.debug = True
            self.monitoring.log_level = "DEBUG"
            self.security.api_rate_limit = 10000  # Higher for dev
        
        elif self.environment == DeploymentEnvironment.PRODUCTION:
            self.debug = False
            self.monitoring.log_level = "INFO"
            self.scaling.min_instances = 2  # Minimum for HA


def create_config(
    environment: str = "development",
    region: str = "us-east-1",
    **overrides
) -> DeploymentConfig:
    """Create deployment configuration with defaults."""
    
    config = DeploymentConfig(
        environment=DeploymentEnvironment(environment),
        region=Region(region),
        scaling=ScalingConfig(),
        security=SecurityConfig(),
        monitoring=MonitoringConfig(),
        i18n=I18nConfig(),
        compliance=ComplianceConfig()
    )
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.custom[key] = value
    
    return config


def load_config_from_env() -> DeploymentConfig:
    """Load configuration from environment variables."""
    
    environment = os.getenv("REFLEXION_ENV", "development")
    region = os.getenv("REFLEXION_REGION", "us-east-1")
    
    config = create_config(environment, region)
    
    # Override with environment variables
    if os.getenv("REFLEXION_DEBUG"):
        config.debug = os.getenv("REFLEXION_DEBUG").lower() == "true"
    
    if os.getenv("REFLEXION_LOG_LEVEL"):
        config.monitoring.log_level = os.getenv("REFLEXION_LOG_LEVEL")
    
    if os.getenv("REFLEXION_MIN_INSTANCES"):
        config.scaling.min_instances = int(os.getenv("REFLEXION_MIN_INSTANCES"))
    
    if os.getenv("REFLEXION_MAX_INSTANCES"):
        config.scaling.max_instances = int(os.getenv("REFLEXION_MAX_INSTANCES"))
    
    if os.getenv("DATABASE_URL"):
        config.database_url = os.getenv("DATABASE_URL")
    
    if os.getenv("REDIS_URL"):
        config.redis_url = os.getenv("REDIS_URL")
    
    return config


# Regional configurations
REGIONAL_CONFIGS = {
    Region.US_EAST_1: {
        "timezone": "America/New_York",
        "compliance": {"gdpr_enabled": False, "ccpa_enabled": True},
        "i18n": {"default_language": "en", "supported_languages": ["en", "es"]}
    },
    Region.EU_WEST_1: {
        "timezone": "Europe/London", 
        "compliance": {"gdpr_enabled": True, "ccpa_enabled": False},
        "i18n": {"default_language": "en", "supported_languages": ["en", "fr", "de"]}
    },
    Region.AP_SOUTHEAST_1: {
        "timezone": "Asia/Singapore",
        "compliance": {"pdpa_enabled": True, "gdpr_enabled": False},
        "i18n": {"default_language": "en", "supported_languages": ["en", "zh-CN", "ja"]}
    }
}


def apply_regional_config(config: DeploymentConfig) -> DeploymentConfig:
    """Apply region-specific configuration."""
    
    regional_settings = REGIONAL_CONFIGS.get(config.region, {})
    
    for section, settings in regional_settings.items():
        if section == "compliance":
            for key, value in settings.items():
                setattr(config.compliance, key, value)
        elif section == "i18n":
            for key, value in settings.items():
                setattr(config.i18n, key, value)
        else:
            config.custom[section] = settings
    
    return config
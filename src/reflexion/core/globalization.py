"""Global-first implementation with multi-region, i18n, and compliance support."""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
import locale
import gettext

from .logging_config import logger
from .exceptions import ReflexionError


class Region(Enum):
    """Supported global regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"  
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    SOUTH_AMERICA = "sa-east-1"


class Language(Enum):
    """Supported languages (ISO 639-1 codes)."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"


class ComplianceRegime(Enum):
    """Data protection and compliance regimes."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore, Thailand)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    DPA = "dpa"  # Data Protection Act (UK)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)
    POPIA = "popia"  # Protection of Personal Information Act (South Africa)


@dataclass
class RegionalConfig:
    """Regional configuration settings."""
    region: Region
    data_residency_required: bool
    compliance_regimes: List[ComplianceRegime]
    primary_language: Language
    supported_languages: List[Language]
    currency: str
    timezone: str
    business_hours: Tuple[int, int]  # 24-hour format (start, end)
    emergency_contacts: List[str]
    regulatory_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalizationContext:
    """Context for localization and internationalization."""
    language: Language
    region: Region
    timezone: str
    currency: str
    date_format: str
    number_format: str
    rtl_script: bool = False  # Right-to-left script languages
    cultural_adaptations: Dict[str, Any] = field(default_factory=dict)


class LocalizationManager:
    """Comprehensive localization and internationalization management."""
    
    def __init__(self, default_language: Language = Language.ENGLISH):
        self.default_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.contexts: Dict[str, LocalizationContext] = {}
        
        # Load default translations
        self._initialize_translations()
        self._initialize_contexts()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_translations(self):
        """Initialize default translations for key system messages."""
        self.translations = {
            Language.ENGLISH.value: {
                "task_started": "Task execution started",
                "task_completed": "Task completed successfully",
                "task_failed": "Task execution failed",
                "reflection_generated": "Reflection generated for improvement",
                "security_violation": "Security violation detected",
                "system_healthy": "System is operating normally",
                "system_warning": "System warning detected",
                "system_critical": "Critical system issue detected",
                "rate_limit_exceeded": "Rate limit exceeded, please try again later",
                "unauthorized_access": "Unauthorized access attempt detected",
                "data_processed": "Data processing completed",
                "compliance_check": "Compliance validation successful",
                "error_generic": "An error occurred while processing your request",
                "retry_suggestion": "Please try again in a few moments"
            },
            
            Language.SPANISH.value: {
                "task_started": "Ejecución de tarea iniciada",
                "task_completed": "Tarea completada exitosamente",
                "task_failed": "Ejecución de tarea falló",
                "reflection_generated": "Reflexión generada para mejora",
                "security_violation": "Violación de seguridad detectada",
                "system_healthy": "El sistema opera normalmente",
                "system_warning": "Advertencia del sistema detectada",
                "system_critical": "Problema crítico del sistema detectado",
                "rate_limit_exceeded": "Límite de velocidad excedido, intente más tarde",
                "unauthorized_access": "Intento de acceso no autorizado detectado",
                "data_processed": "Procesamiento de datos completado",
                "compliance_check": "Validación de cumplimiento exitosa",
                "error_generic": "Ocurrió un error al procesar su solicitud",
                "retry_suggestion": "Por favor intente nuevamente en unos momentos"
            },
            
            Language.FRENCH.value: {
                "task_started": "Exécution de tâche démarrée",
                "task_completed": "Tâche complétée avec succès",
                "task_failed": "Échec de l'exécution de la tâche",
                "reflection_generated": "Réflexion générée pour amélioration",
                "security_violation": "Violation de sécurité détectée",
                "system_healthy": "Le système fonctionne normalement",
                "system_warning": "Avertissement système détecté",
                "system_critical": "Problème système critique détecté",
                "rate_limit_exceeded": "Limite de débit dépassée, veuillez réessayer plus tard",
                "unauthorized_access": "Tentative d'accès non autorisé détectée",
                "data_processed": "Traitement des données terminé",
                "compliance_check": "Validation de conformité réussie",
                "error_generic": "Une erreur s'est produite lors du traitement de votre demande",
                "retry_suggestion": "Veuillez réessayer dans quelques instants"
            },
            
            Language.GERMAN.value: {
                "task_started": "Aufgabenausführung gestartet",
                "task_completed": "Aufgabe erfolgreich abgeschlossen",
                "task_failed": "Aufgabenausführung fehlgeschlagen",
                "reflection_generated": "Reflexion zur Verbesserung generiert",
                "security_violation": "Sicherheitsverletzung erkannt",
                "system_healthy": "System arbeitet normal",
                "system_warning": "Systemwarnung erkannt",
                "system_critical": "Kritisches Systemproblem erkannt",
                "rate_limit_exceeded": "Rate-Limit überschritten, bitte später versuchen",
                "unauthorized_access": "Unbefugter Zugangsversuch erkannt",
                "data_processed": "Datenverarbeitung abgeschlossen",
                "compliance_check": "Compliance-Validierung erfolgreich",
                "error_generic": "Ein Fehler ist bei der Verarbeitung Ihrer Anfrage aufgetreten",
                "retry_suggestion": "Bitte versuchen Sie es in wenigen Augenblicken erneut"
            },
            
            Language.JAPANESE.value: {
                "task_started": "タスク実行が開始されました",
                "task_completed": "タスクが正常に完了しました",
                "task_failed": "タスク実行が失敗しました",
                "reflection_generated": "改善のためのリフレクションが生成されました",
                "security_violation": "セキュリティ違反が検出されました",
                "system_healthy": "システムは正常に動作しています",
                "system_warning": "システム警告が検出されました",
                "system_critical": "重要なシステム問題が検出されました",
                "rate_limit_exceeded": "レート制限を超過しました。しばらく後に再試行してください",
                "unauthorized_access": "不正アクセス試行が検出されました",
                "data_processed": "データ処理が完了しました",
                "compliance_check": "コンプライアンス検証が成功しました",
                "error_generic": "リクエストの処理中にエラーが発生しました",
                "retry_suggestion": "少し後にもう一度お試しください"
            },
            
            Language.CHINESE_SIMPLIFIED.value: {
                "task_started": "任务执行已开始",
                "task_completed": "任务已成功完成",
                "task_failed": "任务执行失败",
                "reflection_generated": "已生成改进反思",
                "security_violation": "检测到安全违规",
                "system_healthy": "系统运行正常",
                "system_warning": "检测到系统警告",
                "system_critical": "检测到严重系统问题",
                "rate_limit_exceeded": "超过速率限制，请稍后重试",
                "unauthorized_access": "检测到未授权访问尝试",
                "data_processed": "数据处理完成",
                "compliance_check": "合规验证成功",
                "error_generic": "处理您的请求时发生错误",
                "retry_suggestion": "请稍后重试"
            }
        }
    
    def _initialize_contexts(self):
        """Initialize localization contexts for different regions."""
        self.contexts = {
            "us": LocalizationContext(
                language=Language.ENGLISH,
                region=Region.US_EAST,
                timezone="America/New_York",
                currency="USD",
                date_format="%m/%d/%Y",
                number_format="1,234.56"
            ),
            "eu": LocalizationContext(
                language=Language.ENGLISH,
                region=Region.EU_WEST,
                timezone="Europe/London",
                currency="EUR",
                date_format="%d/%m/%Y",
                number_format="1.234,56"
            ),
            "de": LocalizationContext(
                language=Language.GERMAN,
                region=Region.EU_CENTRAL,
                timezone="Europe/Berlin",
                currency="EUR",
                date_format="%d.%m.%Y",
                number_format="1.234,56"
            ),
            "fr": LocalizationContext(
                language=Language.FRENCH,
                region=Region.EU_WEST,
                timezone="Europe/Paris",
                currency="EUR",
                date_format="%d/%m/%Y",
                number_format="1 234,56"
            ),
            "jp": LocalizationContext(
                language=Language.JAPANESE,
                region=Region.ASIA_NORTHEAST,
                timezone="Asia/Tokyo",
                currency="JPY",
                date_format="%Y/%m/%d",
                number_format="1,234"
            ),
            "cn": LocalizationContext(
                language=Language.CHINESE_SIMPLIFIED,
                region=Region.ASIA_PACIFIC,
                timezone="Asia/Shanghai",
                currency="CNY",
                date_format="%Y-%m-%d",
                number_format="1,234.56"
            ),
            "ar": LocalizationContext(
                language=Language.ARABIC,
                region=Region.EU_WEST,  # Many Arabic users in EU
                timezone="Europe/London",
                currency="USD",
                date_format="%d/%m/%Y",
                number_format="1,234.56",
                rtl_script=True
            )
        }
    
    def get_localized_message(self, key: str, language: Language = None, **kwargs) -> str:
        """Get localized message for the given key and language."""
        language = language or self.default_language
        lang_code = language.value
        
        if lang_code in self.translations and key in self.translations[lang_code]:
            message = self.translations[lang_code][key]
        else:
            # Fallback to English
            message = self.translations[Language.ENGLISH.value].get(key, f"Missing translation: {key}")
        
        # Apply any formatting arguments
        try:
            return message.format(**kwargs) if kwargs else message
        except (KeyError, ValueError):
            return message
    
    def get_context(self, region_code: str) -> Optional[LocalizationContext]:
        """Get localization context for a region."""
        return self.contexts.get(region_code)
    
    def format_datetime(self, dt: datetime, context: LocalizationContext) -> str:
        """Format datetime according to regional preferences."""
        return dt.strftime(context.date_format)
    
    def format_number(self, number: float, context: LocalizationContext) -> str:
        """Format number according to regional preferences."""
        if context.number_format == "1,234.56":
            return f"{number:,.2f}"
        elif context.number_format == "1.234,56":
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif context.number_format == "1 234,56":
            return f"{number:,.2f}".replace(",", " ").replace(".", ",")
        elif context.number_format == "1,234":
            return f"{int(number):,}"
        else:
            return str(number)
    
    def format_currency(self, amount: float, context: LocalizationContext) -> str:
        """Format currency according to regional preferences."""
        formatted_number = self.format_number(amount, context)
        
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CNY": "¥",
            "CAD": "C$",
            "AUD": "A$"
        }
        
        symbol = currency_symbols.get(context.currency, context.currency)
        
        if context.currency in ["USD", "CAD", "AUD"]:
            return f"{symbol}{formatted_number}"
        elif context.currency in ["EUR", "GBP"]:
            return f"{formatted_number} {symbol}"
        else:
            return f"{formatted_number} {context.currency}"


class ComplianceFramework:
    """Multi-jurisdictional compliance framework."""
    
    def __init__(self):
        self.regional_requirements = self._initialize_regional_requirements()
        self.data_processing_records: Dict[str, List[Dict]] = {}
        self.consent_records: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
    
    def _initialize_regional_requirements(self) -> Dict[ComplianceRegime, Dict[str, Any]]:
        """Initialize compliance requirements for different regimes."""
        return {
            ComplianceRegime.GDPR: {
                "data_subject_rights": [
                    "right_to_access", "right_to_rectification", "right_to_erasure",
                    "right_to_restrict_processing", "right_to_data_portability",
                    "right_to_object", "right_not_to_be_subject_to_automated_decision_making"
                ],
                "lawful_bases": [
                    "consent", "contract", "legal_obligation", "vital_interests",
                    "public_task", "legitimate_interests"
                ],
                "retention_periods": {
                    "personal_data": 2555,  # 7 years in days
                    "consent_records": 2555,
                    "processing_records": 2555
                },
                "breach_notification_hours": 72,
                "dpo_required": True,
                "impact_assessment_required": True
            },
            
            ComplianceRegime.CCPA: {
                "consumer_rights": [
                    "right_to_know", "right_to_delete", "right_to_opt_out",
                    "right_to_non_discrimination"
                ],
                "categories_of_personal_info": [
                    "identifiers", "personal_info_categories", "commercial_info",
                    "biometric_info", "internet_activity", "geolocation_data",
                    "sensory_data", "professional_info", "education_info",
                    "inferences"
                ],
                "retention_periods": {
                    "personal_data": 1825,  # 5 years in days
                    "deletion_requests": 730   # 2 years in days
                },
                "opt_out_required": True,
                "privacy_policy_required": True
            },
            
            ComplianceRegime.PDPA: {
                "individual_rights": [
                    "right_to_access", "right_to_correction", "right_to_data_portability"
                ],
                "consent_requirements": {
                    "explicit_consent": True,
                    "granular_consent": True,
                    "withdraw_consent": True
                },
                "retention_periods": {
                    "personal_data": 2555,  # 7 years
                    "consent_records": 1095  # 3 years
                },
                "dpo_required": True,
                "breach_notification_hours": 72
            },
            
            ComplianceRegime.LGPD: {
                "data_subject_rights": [
                    "confirmation_and_access", "correction", "anonymization_blocking_deletion",
                    "portability", "information_about_sharing", "revocation_of_consent"
                ],
                "legal_bases": [
                    "consent", "legal_obligation", "public_administration",
                    "research", "contract_execution", "judicial_process",
                    "life_protection", "health_protection", "legitimate_interest",
                    "credit_protection"
                ],
                "retention_periods": {
                    "personal_data": 1825,  # 5 years
                    "consent_records": 1825
                },
                "dpo_required": True,
                "impact_assessment_required": True
            }
        }
    
    def get_applicable_regimes(self, region: Region) -> List[ComplianceRegime]:
        """Get applicable compliance regimes for a region."""
        regime_mapping = {
            Region.US_EAST: [ComplianceRegime.CCPA],
            Region.US_WEST: [ComplianceRegime.CCPA],
            Region.EU_WEST: [ComplianceRegime.GDPR],
            Region.EU_CENTRAL: [ComplianceRegime.GDPR],
            Region.ASIA_PACIFIC: [ComplianceRegime.PDPA],
            Region.ASIA_NORTHEAST: [ComplianceRegime.PDPA],
            Region.CANADA: [ComplianceRegime.PIPEDA, ComplianceRegime.GDPR],  # Quebec has GDPR-like laws
            Region.AUSTRALIA: [ComplianceRegime.PRIVACY_ACT],
            Region.SOUTH_AMERICA: [ComplianceRegime.LGPD]
        }
        
        return regime_mapping.get(region, [])
    
    def validate_data_processing(
        self,
        region: Region,
        data_type: str,
        purpose: str,
        legal_basis: str,
        user_consent: bool = False
    ) -> Dict[str, Any]:
        """Validate data processing against regional compliance requirements."""
        applicable_regimes = self.get_applicable_regimes(region)
        validation_results = {}
        
        for regime in applicable_regimes:
            requirements = self.regional_requirements[regime]
            
            # Check legal basis
            valid_bases = requirements.get("lawful_bases", requirements.get("legal_bases", []))
            legal_basis_valid = legal_basis in valid_bases
            
            # Check consent requirements
            consent_required = False
            if regime in [ComplianceRegime.GDPR, ComplianceRegime.PDPA]:
                if legal_basis == "consent":
                    consent_required = True
            elif regime == ComplianceRegime.CCPA:
                # CCPA generally requires opt-out rather than opt-in
                consent_required = False
            
            validation_results[regime.value] = {
                "legal_basis_valid": legal_basis_valid,
                "consent_required": consent_required,
                "consent_provided": user_consent,
                "compliant": legal_basis_valid and (not consent_required or user_consent)
            }
        
        overall_compliant = all(result["compliant"] for result in validation_results.values())
        
        return {
            "overall_compliant": overall_compliant,
            "regime_results": validation_results,
            "applicable_regimes": [regime.value for regime in applicable_regimes]
        }
    
    def record_data_processing(
        self,
        region: Region,
        user_id: str,
        data_type: str,
        purpose: str,
        legal_basis: str,
        retention_period: int,
        user_consent: bool = False
    ):
        """Record data processing activity for compliance."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "region": region.value,
            "user_id": user_id,
            "data_type": data_type,
            "purpose": purpose,
            "legal_basis": legal_basis,
            "retention_period": retention_period,
            "user_consent": user_consent
        }
        
        if region.value not in self.data_processing_records:
            self.data_processing_records[region.value] = []
        
        self.data_processing_records[region.value].append(record)
        
        self.logger.info(f"Recorded data processing: {data_type} in {region.value} for {purpose}")
    
    def handle_data_subject_request(
        self,
        region: Region,
        request_type: str,
        user_id: str,
        additional_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        applicable_regimes = self.get_applicable_regimes(region)
        
        # Check if request type is supported in this region
        supported_rights = set()
        for regime in applicable_regimes:
            requirements = self.regional_requirements[regime]
            
            # Get rights from different naming conventions
            rights_keys = ["data_subject_rights", "consumer_rights", "individual_rights"]
            for key in rights_keys:
                if key in requirements:
                    supported_rights.update(requirements[key])
        
        if request_type not in supported_rights:
            return {
                "status": "unsupported",
                "message": f"Request type '{request_type}' not supported in region {region.value}",
                "supported_rights": list(supported_rights)
            }
        
        # Process the request
        response = {"status": "processed", "request_type": request_type, "user_id": user_id}
        
        if request_type in ["right_to_access", "right_to_know", "confirmation_and_access"]:
            # Return user's data processing records
            user_records = [
                record for records in self.data_processing_records.values()
                for record in records
                if record["user_id"] == user_id
            ]
            response["data"] = user_records
        
        elif request_type in ["right_to_delete", "right_to_erasure"]:
            # Mark for deletion (actual deletion would be handled by data management systems)
            response["action"] = "marked_for_deletion"
            response["retention_override"] = "immediate_deletion_requested"
        
        elif request_type == "right_to_opt_out":
            # Record opt-out preference
            self.consent_records[user_id] = {
                "opted_out": True,
                "timestamp": datetime.now().isoformat(),
                "region": region.value
            }
            response["action"] = "opted_out"
        
        return response


class MultiRegionManager:
    """Comprehensive multi-region deployment and data residency management."""
    
    def __init__(self):
        self.regional_configs = self._initialize_regional_configs()
        self.localization_manager = LocalizationManager()
        self.compliance_framework = ComplianceFramework()
        
        # Active deployments per region
        self.regional_deployments: Dict[Region, Dict[str, Any]] = {}
        
        # Data routing and residency
        self.data_residency_rules: Dict[Region, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(__name__)
        
        self._initialize_data_residency_rules()
    
    def _initialize_regional_configs(self) -> Dict[Region, RegionalConfig]:
        """Initialize regional configurations."""
        return {
            Region.US_EAST: RegionalConfig(
                region=Region.US_EAST,
                data_residency_required=False,
                compliance_regimes=[ComplianceRegime.CCPA],
                primary_language=Language.ENGLISH,
                supported_languages=[Language.ENGLISH, Language.SPANISH],
                currency="USD",
                timezone="America/New_York",
                business_hours=(9, 17),  # 9 AM to 5 PM
                emergency_contacts=["us-support@company.com"]
            ),
            
            Region.EU_WEST: RegionalConfig(
                region=Region.EU_WEST,
                data_residency_required=True,
                compliance_regimes=[ComplianceRegime.GDPR],
                primary_language=Language.ENGLISH,
                supported_languages=[Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH],
                currency="EUR",
                timezone="Europe/London",
                business_hours=(9, 17),
                emergency_contacts=["eu-support@company.com"]
            ),
            
            Region.EU_CENTRAL: RegionalConfig(
                region=Region.EU_CENTRAL,
                data_residency_required=True,
                compliance_regimes=[ComplianceRegime.GDPR],
                primary_language=Language.GERMAN,
                supported_languages=[Language.GERMAN, Language.ENGLISH, Language.FRENCH],
                currency="EUR",
                timezone="Europe/Berlin",
                business_hours=(8, 16),  # Earlier business hours in Germany
                emergency_contacts=["de-support@company.com"]
            ),
            
            Region.ASIA_PACIFIC: RegionalConfig(
                region=Region.ASIA_PACIFIC,
                data_residency_required=True,
                compliance_regimes=[ComplianceRegime.PDPA],
                primary_language=Language.ENGLISH,
                supported_languages=[Language.ENGLISH, Language.CHINESE_SIMPLIFIED, Language.JAPANESE],
                currency="USD",  # USD commonly used in Singapore
                timezone="Asia/Singapore",
                business_hours=(9, 18),  # Longer business hours in Asia
                emergency_contacts=["apac-support@company.com"]
            ),
            
            Region.ASIA_NORTHEAST: RegionalConfig(
                region=Region.ASIA_NORTHEAST,
                data_residency_required=True,
                compliance_regimes=[ComplianceRegime.PDPA],
                primary_language=Language.JAPANESE,
                supported_languages=[Language.JAPANESE, Language.ENGLISH, Language.KOREAN],
                currency="JPY",
                timezone="Asia/Tokyo",
                business_hours=(9, 17),
                emergency_contacts=["japan-support@company.com"]
            ),
            
            Region.SOUTH_AMERICA: RegionalConfig(
                region=Region.SOUTH_AMERICA,
                data_residency_required=True,
                compliance_regimes=[ComplianceRegime.LGPD],
                primary_language=Language.PORTUGUESE,
                supported_languages=[Language.PORTUGUESE, Language.SPANISH, Language.ENGLISH],
                currency="BRL",
                timezone="America/Sao_Paulo",
                business_hours=(8, 17),
                emergency_contacts=["br-support@company.com"]
            )
        }
    
    def _initialize_data_residency_rules(self):
        """Initialize data residency and routing rules."""
        self.data_residency_rules = {
            Region.EU_WEST: {
                "data_must_stay_in_region": True,
                "allowed_transfer_regions": [Region.EU_CENTRAL],  # Within EU
                "prohibited_regions": [Region.US_EAST, Region.US_WEST, Region.ASIA_PACIFIC],
                "special_categories": {
                    "personal_data": "strict_residency",
                    "financial_data": "strict_residency",
                    "health_data": "no_transfer"
                }
            },
            
            Region.EU_CENTRAL: {
                "data_must_stay_in_region": True,
                "allowed_transfer_regions": [Region.EU_WEST],
                "prohibited_regions": [Region.US_EAST, Region.US_WEST, Region.ASIA_PACIFIC],
                "special_categories": {
                    "personal_data": "strict_residency",
                    "financial_data": "strict_residency",
                    "health_data": "no_transfer"
                }
            },
            
            Region.ASIA_PACIFIC: {
                "data_must_stay_in_region": True,
                "allowed_transfer_regions": [Region.ASIA_NORTHEAST],
                "prohibited_regions": [],
                "special_categories": {
                    "personal_data": "strict_residency",
                    "government_data": "no_transfer"
                }
            },
            
            Region.US_EAST: {
                "data_must_stay_in_region": False,
                "allowed_transfer_regions": [Region.US_WEST, Region.CANADA],
                "prohibited_regions": [],
                "special_categories": {
                    "healthcare_data": "strict_residency",  # HIPAA requirements
                    "financial_data": "regulated_transfer"
                }
            }
        }
    
    def get_optimal_region(
        self,
        user_location: Optional[str] = None,
        data_type: Optional[str] = None,
        compliance_requirements: Optional[List[ComplianceRegime]] = None
    ) -> Region:
        """Determine optimal region for a user request."""
        
        # Geographic proximity mapping
        location_to_region = {
            "US": Region.US_EAST,
            "CA": Region.CANADA,
            "GB": Region.EU_WEST,
            "FR": Region.EU_WEST,
            "DE": Region.EU_CENTRAL,
            "IT": Region.EU_WEST,
            "ES": Region.EU_WEST,
            "JP": Region.ASIA_NORTHEAST,
            "SG": Region.ASIA_PACIFIC,
            "AU": Region.AUSTRALIA,
            "BR": Region.SOUTH_AMERICA
        }
        
        # Start with location-based preference
        preferred_region = location_to_region.get(user_location, Region.US_EAST)
        
        # Check data residency requirements
        if data_type and data_type in ["personal_data", "financial_data", "health_data"]:
            residency_rules = self.data_residency_rules.get(preferred_region, {})
            if not residency_rules.get("data_must_stay_in_region", False):
                # Region doesn't require data residency, can use preferred region
                pass
            else:
                # Must use this region due to residency requirements
                pass
        
        # Check compliance requirements
        if compliance_requirements:
            for region, config in self.regional_configs.items():
                if any(req in config.compliance_regimes for req in compliance_requirements):
                    preferred_region = region
                    break
        
        return preferred_region
    
    def validate_cross_region_transfer(
        self,
        source_region: Region,
        target_region: Region,
        data_type: str
    ) -> Dict[str, Any]:
        """Validate if cross-region data transfer is allowed."""
        
        source_rules = self.data_residency_rules.get(source_region, {})
        
        # Check if target region is allowed
        allowed_regions = source_rules.get("allowed_transfer_regions", [])
        prohibited_regions = source_rules.get("prohibited_regions", [])
        
        if target_region in prohibited_regions:
            return {
                "allowed": False,
                "reason": f"Transfer to {target_region.value} is prohibited from {source_region.value}",
                "alternative_regions": allowed_regions
            }
        
        # Check special categories
        special_categories = source_rules.get("special_categories", {})
        if data_type in special_categories:
            restriction = special_categories[data_type]
            
            if restriction == "no_transfer":
                return {
                    "allowed": False,
                    "reason": f"Data type '{data_type}' cannot be transferred from {source_region.value}",
                    "must_process_in_region": source_region.value
                }
            elif restriction == "strict_residency" and target_region not in allowed_regions:
                return {
                    "allowed": False,
                    "reason": f"Strict residency required for '{data_type}' in {source_region.value}",
                    "allowed_regions": allowed_regions
                }
        
        return {
            "allowed": True,
            "source_region": source_region.value,
            "target_region": target_region.value,
            "data_type": data_type
        }
    
    def get_localized_response(
        self,
        message_key: str,
        region: Region,
        user_language: Optional[Language] = None,
        **kwargs
    ) -> str:
        """Get localized response for a region and language."""
        
        config = self.regional_configs.get(region)
        if not config:
            config = self.regional_configs[Region.US_EAST]  # Fallback
        
        # Determine language
        if user_language and user_language in config.supported_languages:
            language = user_language
        else:
            language = config.primary_language
        
        return self.localization_manager.get_localized_message(message_key, language, **kwargs)
    
    def process_regional_request(
        self,
        request_data: Dict[str, Any],
        user_region: Optional[Region] = None,
        user_language: Optional[Language] = None
    ) -> Dict[str, Any]:
        """Process a request with full regional compliance and localization."""
        
        # Determine optimal region
        optimal_region = user_region or self.get_optimal_region(
            user_location=request_data.get("user_location"),
            data_type=request_data.get("data_type"),
            compliance_requirements=request_data.get("compliance_requirements")
        )
        
        config = self.regional_configs[optimal_region]
        
        # Validate compliance
        compliance_validation = self.compliance_framework.validate_data_processing(
            region=optimal_region,
            data_type=request_data.get("data_type", "general"),
            purpose=request_data.get("purpose", "service_provision"),
            legal_basis=request_data.get("legal_basis", "legitimate_interests"),
            user_consent=request_data.get("user_consent", False)
        )
        
        if not compliance_validation["overall_compliant"]:
            return {
                "status": "compliance_error",
                "message": self.get_localized_response(
                    "compliance_check", optimal_region, user_language
                ),
                "region": optimal_region.value,
                "compliance_issues": compliance_validation["regime_results"]
            }
        
        # Record processing activity
        if request_data.get("user_id"):
            self.compliance_framework.record_data_processing(
                region=optimal_region,
                user_id=request_data["user_id"],
                data_type=request_data.get("data_type", "general"),
                purpose=request_data.get("purpose", "service_provision"),
                legal_basis=request_data.get("legal_basis", "legitimate_interests"),
                retention_period=config.regulatory_requirements.get("default_retention", 365),
                user_consent=request_data.get("user_consent", False)
            )
        
        # Get localization context
        region_code = optimal_region.value.split("-")[0]  # us-east-1 -> us
        context = self.localization_manager.get_context(region_code)
        
        return {
            "status": "success",
            "message": self.get_localized_response(
                "data_processed", optimal_region, user_language
            ),
            "region": optimal_region.value,
            "language": (user_language or config.primary_language).value,
            "compliance_status": compliance_validation,
            "localization_context": context,
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def get_regional_status(self) -> Dict[str, Any]:
        """Get comprehensive status across all regions."""
        status = {
            "total_regions": len(self.regional_configs),
            "regions": {},
            "global_compliance": {},
            "supported_languages": set(),
            "supported_currencies": set()
        }
        
        for region, config in self.regional_configs.items():
            status["regions"][region.value] = {
                "primary_language": config.primary_language.value,
                "supported_languages": [lang.value for lang in config.supported_languages],
                "currency": config.currency,
                "timezone": config.timezone,
                "data_residency_required": config.data_residency_required,
                "compliance_regimes": [regime.value for regime in config.compliance_regimes],
                "business_hours": f"{config.business_hours[0]:02d}:00-{config.business_hours[1]:02d}:00"
            }
            
            # Aggregate global capabilities
            status["supported_languages"].update([lang.value for lang in config.supported_languages])
            status["supported_currencies"].add(config.currency)
        
        # Convert sets to lists for JSON serialization
        status["supported_languages"] = list(status["supported_languages"])
        status["supported_currencies"] = list(status["supported_currencies"])
        
        # Global compliance overview
        all_regimes = set()
        for config in self.regional_configs.values():
            all_regimes.update(config.compliance_regimes)
        
        status["global_compliance"] = {
            "supported_regimes": [regime.value for regime in all_regimes],
            "data_residency_regions": [
                region.value for region, config in self.regional_configs.items()
                if config.data_residency_required
            ]
        }
        
        return status


# Global instances
localization_manager = LocalizationManager()
compliance_framework = ComplianceFramework()
multi_region_manager = MultiRegionManager()
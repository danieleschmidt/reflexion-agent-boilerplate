"""Global-First Internationalization (i18n) Support for Autonomous SDLC.

This module provides comprehensive multilingual support for the autonomous SDLC system,
enabling global deployment with localized messages, errors, and documentation.
"""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for the autonomous SDLC system."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"
    DUTCH = "nl"
    ARABIC = "ar"
    HINDI = "hi"


class TranslationManager:
    """Manages translations for the autonomous SDLC system."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_translations: Dict[str, str] = {}
        
        # Initialize translations
        self._load_translations()
        self._setup_fallback_translations()
    
    def _load_translations(self):
        """Load translation files for all supported languages."""
        translations_dir = Path(__file__).parent / "locales"
        
        # Create translations directory if it doesn't exist
        translations_dir.mkdir(exist_ok=True)
        
        for language in SupportedLanguage:
            lang_file = translations_dir / f"{language.value}.json"
            
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[language.value] = json.load(f)
                    logger.debug(f"Loaded translations for {language.value}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load translations for {language.value}: {e}")
                    self.translations[language.value] = {}
            else:
                self.translations[language.value] = {}
                logger.debug(f"No translation file found for {language.value}, using empty dict")
    
    def _setup_fallback_translations(self):
        """Setup fallback English translations."""
        self.fallback_translations = {
            # System Messages
            "system.startup": "Autonomous SDLC system starting up",
            "system.shutdown": "Autonomous SDLC system shutting down",
            "system.ready": "System ready for operation",
            "system.error": "System error occurred",
            
            # Research Orchestrator
            "research.hypothesis.generated": "Research hypothesis generated: {hypothesis}",
            "research.experiment.started": "Experiment started: {experiment_id}",
            "research.experiment.completed": "Experiment completed with results: {results}",
            "research.validation.success": "Hypothesis validation successful",
            "research.validation.failure": "Hypothesis validation failed",
            "research.cycle.started": "Autonomous research cycle started",
            "research.cycle.completed": "Research cycle completed in {duration}s",
            
            # Error Recovery
            "recovery.circuit.opened": "Circuit breaker opened for component: {component}",
            "recovery.circuit.closed": "Circuit breaker closed for component: {component}",
            "recovery.fallback.activated": "Fallback mechanism activated for {operation}",
            "recovery.retry.attempted": "Retry attempt {attempt} of {max_attempts} for {operation}",
            "recovery.recovery.successful": "Recovery successful for {component}",
            "recovery.recovery.failed": "Recovery failed for {component}: {error}",
            "recovery.self_healing.started": "Self-healing process started",
            "recovery.self_healing.completed": "Self-healing process completed",
            
            # Monitoring System
            "monitoring.alert.triggered": "Alert triggered: {alert_type} - {message}",
            "monitoring.alert.resolved": "Alert resolved: {alert_id}",
            "monitoring.threshold.violated": "Threshold violated: {metric} {operator} {threshold}",
            "monitoring.system.healthy": "System health check passed",
            "monitoring.system.warning": "System health warning: {details}",
            "monitoring.system.critical": "System health critical: {details}",
            "monitoring.metrics.collected": "Metrics collected: {count} data points",
            "monitoring.performance.degraded": "Performance degradation detected",
            
            # Distributed Processing
            "distributed.node.added": "Processing node added: {node_id}",
            "distributed.node.removed": "Processing node removed: {node_id}",
            "distributed.task.submitted": "Task submitted: {task_id}",
            "distributed.task.completed": "Task completed: {task_id} in {duration}s",
            "distributed.task.failed": "Task failed: {task_id} - {error}",
            "distributed.scaling.up": "Scaling up: Added {count} nodes",
            "distributed.scaling.down": "Scaling down: Removed {count} nodes",
            "distributed.load.balanced": "Load balanced across {node_count} nodes",
            
            # Quality Gates
            "quality.test.started": "Quality gate test started: {test_name}",
            "quality.test.passed": "Quality gate passed: {test_name}",
            "quality.test.failed": "Quality gate failed: {test_name} - {reason}",
            "quality.coverage.threshold": "Test coverage: {coverage}% (threshold: {threshold}%)",
            "quality.security.scan": "Security scan completed: {findings} findings",
            "quality.performance.benchmark": "Performance benchmark: {metric} = {value}",
            
            # Configuration
            "config.loaded": "Configuration loaded from {source}",
            "config.invalid": "Invalid configuration: {error}",
            "config.updated": "Configuration updated: {key} = {value}",
            "config.default": "Using default configuration",
            
            # Errors
            "error.connection": "Connection error: {details}",
            "error.timeout": "Operation timeout: {operation}",
            "error.authentication": "Authentication failed: {reason}",
            "error.authorization": "Authorization denied: {resource}",
            "error.validation": "Validation error: {field} - {message}",
            "error.processing": "Processing error: {details}",
            "error.unknown": "Unknown error occurred: {details}",
            
            # Success Messages
            "success.operation.completed": "Operation completed successfully",
            "success.task.finished": "Task finished: {task_name}",
            "success.deployment.ready": "Deployment ready for production",
            "success.optimization.applied": "Optimization applied: {improvement}",
            
            # Warnings
            "warning.resource.low": "Low resource warning: {resource} at {level}%",
            "warning.deprecated": "Deprecated feature used: {feature}",
            "warning.configuration.missing": "Missing configuration: {key}",
            
            # Information
            "info.status.update": "Status update: {status}",
            "info.progress.update": "Progress: {current}/{total} ({percentage}%)",
            "info.resource.usage": "Resource usage: CPU {cpu}%, Memory {memory}%",
        }
    
    def set_language(self, language: Union[SupportedLanguage, str]):
        """Set the current language for translations."""
        if isinstance(language, str):
            try:
                language = SupportedLanguage(language)
            except ValueError:
                logger.warning(f"Unsupported language: {language}, using default")
                language = self.default_language
        
        self.current_language = language
        logger.info(f"Language set to: {language.value}")
    
    def get_translation(self, key: str, **kwargs) -> str:
        """Get a translated message for the given key."""
        lang_code = self.current_language.value
        
        # Try to get translation from current language
        translation = self.translations.get(lang_code, {}).get(key)
        
        # Fall back to English if not found
        if not translation:
            translation = self.translations.get("en", {}).get(key)
        
        # Fall back to hardcoded fallback if still not found
        if not translation:
            translation = self.fallback_translations.get(key, f"[MISSING: {key}]")
        
        # Format the translation with provided arguments
        try:
            return translation.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Translation formatting error for key '{key}': {e}")
            return translation
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand method for getting translations."""
        return self.get_translation(key, **kwargs)
    
    def get_available_languages(self) -> list:
        """Get list of available languages."""
        return [lang.value for lang in SupportedLanguage]
    
    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        try:
            SupportedLanguage(language)
            return True
        except ValueError:
            return False
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a language."""
        language_info = {
            "en": {"name": "English", "native": "English", "rtl": False},
            "es": {"name": "Spanish", "native": "Español", "rtl": False},
            "fr": {"name": "French", "native": "Français", "rtl": False},
            "de": {"name": "German", "native": "Deutsch", "rtl": False},
            "ja": {"name": "Japanese", "native": "日本語", "rtl": False},
            "zh-CN": {"name": "Chinese Simplified", "native": "简体中文", "rtl": False},
            "zh-TW": {"name": "Chinese Traditional", "native": "繁體中文", "rtl": False},
            "ko": {"name": "Korean", "native": "한국어", "rtl": False},
            "pt": {"name": "Portuguese", "native": "Português", "rtl": False},
            "ru": {"name": "Russian", "native": "Русский", "rtl": False},
            "it": {"name": "Italian", "native": "Italiano", "rtl": False},
            "nl": {"name": "Dutch", "native": "Nederlands", "rtl": False},
            "ar": {"name": "Arabic", "native": "العربية", "rtl": True},
            "hi": {"name": "Hindi", "native": "हिन्दी", "rtl": False},
        }
        
        return language_info.get(language, {
            "name": language,
            "native": language,
            "rtl": False
        })
    
    def create_translation_template(self, output_path: Optional[str] = None) -> str:
        """Create a translation template with all available keys."""
        template = {key: "" for key in self.fallback_translations.keys()}
        template_json = json.dumps(template, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template_json)
            logger.info(f"Translation template created: {output_path}")
        
        return template_json
    
    def validate_translations(self) -> Dict[str, list]:
        """Validate all translation files for missing keys."""
        validation_results = {}
        base_keys = set(self.fallback_translations.keys())
        
        for lang_code, translations in self.translations.items():
            missing_keys = base_keys - set(translations.keys())
            extra_keys = set(translations.keys()) - base_keys
            
            validation_results[lang_code] = {
                "missing_keys": list(missing_keys),
                "extra_keys": list(extra_keys),
                "completion_rate": (len(translations) / len(base_keys)) * 100 if base_keys else 0
            }
        
        return validation_results


# Global translation manager instance
translation_manager = TranslationManager()

# Convenience function for quick access
def _(key: str, **kwargs) -> str:
    """Global function for getting translations."""
    return translation_manager.get_translation(key, **kwargs)


class LocalizedLogger:
    """Logger wrapper with i18n support."""
    
    def __init__(self, logger: logging.Logger, translation_manager: TranslationManager):
        self.logger = logger
        self.tm = translation_manager
    
    def info(self, key: str, **kwargs):
        """Log info message with translation."""
        message = self.tm.get_translation(key, **kwargs)
        self.logger.info(message)
    
    def warning(self, key: str, **kwargs):
        """Log warning message with translation."""
        message = self.tm.get_translation(key, **kwargs)
        self.logger.warning(message)
    
    def error(self, key: str, **kwargs):
        """Log error message with translation."""
        message = self.tm.get_translation(key, **kwargs)
        self.logger.error(message)
    
    def debug(self, key: str, **kwargs):
        """Log debug message with translation."""
        message = self.tm.get_translation(key, **kwargs)
        self.logger.debug(message)
    
    def critical(self, key: str, **kwargs):
        """Log critical message with translation."""
        message = self.tm.get_translation(key, **kwargs)
        self.logger.critical(message)


def get_localized_logger(name: str) -> LocalizedLogger:
    """Get a localized logger for the given name."""
    logger = logging.getLogger(name)
    return LocalizedLogger(logger, translation_manager)


# Auto-detect system locale
def detect_system_locale() -> str:
    """Detect system locale and return appropriate language code."""
    try:
        import locale
        system_locale = locale.getdefaultlocale()[0]
        
        if system_locale:
            # Extract language code (first part before underscore)
            lang_code = system_locale.split('_')[0].lower()
            
            # Map common locale codes
            locale_mapping = {
                'zh': 'zh-CN',  # Default Chinese to Simplified
                'pt': 'pt',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'ja': 'ja',
                'ko': 'ko',
                'ru': 'ru',
                'it': 'it',
                'nl': 'nl',
                'ar': 'ar',
                'hi': 'hi',
            }
            
            mapped_lang = locale_mapping.get(lang_code, lang_code)
            
            # Check if mapped language is supported
            if translation_manager.is_language_supported(mapped_lang):
                return mapped_lang
        
    except Exception as e:
        logger.debug(f"Failed to detect system locale: {e}")
    
    return SupportedLanguage.ENGLISH.value


# Initialize with system locale if available
try:
    system_lang = detect_system_locale()
    if system_lang != SupportedLanguage.ENGLISH.value:
        translation_manager.set_language(system_lang)
        logger.info(f"Auto-detected language: {system_lang}")
except Exception as e:
    logger.debug(f"Using default language due to detection error: {e}")


# Export main functions and classes
__all__ = [
    'SupportedLanguage',
    'TranslationManager',
    'LocalizedLogger',
    'translation_manager',
    'get_localized_logger',
    'detect_system_locale',
    '_'  # Global translation function
]
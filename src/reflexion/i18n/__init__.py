"""Internationalization (i18n) module for Autonomous SDLC."""

from .translations import (
    SupportedLanguage,
    TranslationManager,
    LocalizedLogger,
    translation_manager,
    get_localized_logger,
    detect_system_locale,
    _
)

__all__ = [
    'SupportedLanguage',
    'TranslationManager',
    'LocalizedLogger',
    'translation_manager',
    'get_localized_logger',
    'detect_system_locale',
    '_'
]
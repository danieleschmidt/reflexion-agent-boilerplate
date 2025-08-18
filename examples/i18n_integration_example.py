"""Example: Integrating i18n Support into Autonomous SDLC Components.

This example demonstrates how to integrate the i18n translation system
into the existing autonomous SDLC components.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import i18n components
from reflexion.i18n import (
    translation_manager,
    get_localized_logger,
    SupportedLanguage,
    _
)

# Import existing components
from reflexion.research.advanced_research_execution import AutonomousResearchOrchestrator
from reflexion.core.advanced_error_recovery_v2 import AdvancedErrorRecoverySystem
from reflexion.core.comprehensive_monitoring_v2 import ComprehensiveMonitoringSystem
from reflexion.scaling.distributed_reflexion_engine import DistributedReflexionEngine


class InternationalizedResearchOrchestrator:
    """Example of how to add i18n support to research orchestrator."""
    
    def __init__(self, research_directory: str = None):
        self.orchestrator = AutonomousResearchOrchestrator(research_directory or "/tmp/research")
        self.logger = get_localized_logger(__name__)
    
    async def start_research_cycle(self):
        """Start research cycle with localized messages."""
        self.logger.info("research.cycle.started")
        
        try:
            # Simulate research work
            await asyncio.sleep(0.1)
            
            # Log hypothesis generation
            hypothesis = "Neural networks can improve reflexion quality"
            self.logger.info("research.hypothesis.generated", hypothesis=hypothesis)
            
            # Simulate experiment
            experiment_id = "exp_001"
            self.logger.info("research.experiment.started", experiment_id=experiment_id)
            
            await asyncio.sleep(0.1)
            
            results = "95% accuracy improvement"
            self.logger.info("research.experiment.completed", 
                           experiment_id=experiment_id, results=results)
            
            self.logger.info("research.validation.success")
            
            duration = 2.5
            self.logger.info("research.cycle.completed", duration=duration)
            
            return True
            
        except Exception as e:
            self.logger.error("error.processing", details=str(e))
            return False


class InternationalizedErrorRecovery:
    """Example of adding i18n support to error recovery system."""
    
    def __init__(self):
        self.recovery_system = AdvancedErrorRecoverySystem()
        self.logger = get_localized_logger(__name__)
    
    async def handle_error_with_i18n(self, component: str, error: Exception):
        """Handle error with localized messages."""
        self.logger.info("recovery.self_healing.started")
        
        try:
            # Attempt recovery
            async with self.recovery_system.protected_execution(component):
                # Simulate recovery work
                await asyncio.sleep(0.1)
                
            self.logger.info("recovery.recovery.successful", component=component)
            return True
            
        except Exception as recovery_error:
            self.logger.error("recovery.recovery.failed", 
                            component=component, error=str(recovery_error))
            return False
        finally:
            self.logger.info("recovery.self_healing.completed")


class InternationalizedMonitoring:
    """Example of adding i18n support to monitoring system."""
    
    def __init__(self):
        self.monitoring = ComprehensiveMonitoringSystem()
        self.logger = get_localized_logger(__name__)
    
    def check_system_health_with_i18n(self):
        """Check system health with localized messages."""
        health = self.monitoring.get_system_health()
        
        if health["overall_status"] == "healthy":
            self.logger.info("monitoring.system.healthy")
        elif health["overall_status"] == "warning":
            details = f"Active alerts: {health['alerts']['total_active']}"
            self.logger.warning("monitoring.system.warning", details=details)
        else:
            details = f"Critical issues detected"
            self.logger.critical("monitoring.system.critical", details=details)
        
        # Log metrics collection
        metrics_count = len(health.get("monitoring", {}).get("metrics_tracked", []))
        self.logger.info("monitoring.metrics.collected", count=metrics_count)
        
        return health


class InternationalizedDistributedEngine:
    """Example of adding i18n support to distributed processing."""
    
    def __init__(self):
        self.engine = DistributedReflexionEngine()
        self.logger = get_localized_logger(__name__)
    
    async def process_task_with_i18n(self, task_data: dict):
        """Process task with localized messages."""
        try:
            # Submit task
            task_id = await self.engine.submit_task("analysis", task_data)
            self.logger.info("distributed.task.submitted", task_id=task_id)
            
            # Wait for completion
            import time
            start_time = time.time()
            
            result = await self.engine.get_task_result(task_id, timeout=5.0)
            
            if result:
                duration = time.time() - start_time
                self.logger.info("distributed.task.completed", 
                               task_id=task_id, duration=duration)
                return result
            else:
                self.logger.error("distributed.task.failed", 
                                task_id=task_id, error="Timeout or processing error")
                return None
                
        except Exception as e:
            self.logger.error("distributed.task.failed", 
                            task_id="unknown", error=str(e))
            return None


async def demonstrate_language_switching():
    """Demonstrate switching between different languages."""
    print("=== Demonstrating Multi-language Support ===\\n")
    
    # Test in different languages
    languages = [
        (SupportedLanguage.ENGLISH, "English"),
        (SupportedLanguage.SPANISH, "Spanish"),
        (SupportedLanguage.FRENCH, "French"),
        (SupportedLanguage.CHINESE_SIMPLIFIED, "Chinese (Simplified)")
    ]
    
    for lang, lang_name in languages:
        print(f"--- {lang_name} ---")
        translation_manager.set_language(lang)
        
        # Test various message types
        print("System:", _("system.startup"))
        print("Success:", _("success.operation.completed"))
        print("Error:", _("error.connection", details="Network unreachable"))
        print("Progress:", _("info.progress.update", current=75, total=100, percentage=75))
        print()
    
    # Reset to English
    translation_manager.set_language(SupportedLanguage.ENGLISH)


async def demonstrate_integrated_systems():
    """Demonstrate i18n integration with all systems."""
    print("=== Demonstrating Integrated Systems ===\\n")
    
    # Initialize internationalized components
    research = InternationalizedResearchOrchestrator()
    recovery = InternationalizedErrorRecovery()
    monitoring = InternationalizedMonitoring()
    distributed = InternationalizedDistributedEngine()
    
    # Test research system
    print("--- Research System (Spanish) ---")
    translation_manager.set_language(SupportedLanguage.SPANISH)
    await research.start_research_cycle()
    print()
    
    # Test error recovery
    print("--- Error Recovery (French) ---")
    translation_manager.set_language(SupportedLanguage.FRENCH)
    await recovery.handle_error_with_i18n("test_component", ValueError("Test error"))
    print()
    
    # Test monitoring
    print("--- Monitoring System (Chinese) ---")
    translation_manager.set_language(SupportedLanguage.CHINESE_SIMPLIFIED)
    monitoring.check_system_health_with_i18n()
    print()
    
    # Test distributed processing
    print("--- Distributed Processing (English) ---")
    translation_manager.set_language(SupportedLanguage.ENGLISH)
    await distributed.process_task_with_i18n({"type": "analysis", "data": "test"})
    print()


async def demonstrate_validation():
    """Demonstrate translation validation and management."""
    print("=== Translation Management ===\\n")
    
    # Show available languages
    print("Available languages:")
    for lang in translation_manager.get_available_languages():
        info = translation_manager.get_language_info(lang)
        rtl_indicator = " (RTL)" if info.get("rtl") else ""
        print(f"  {lang}: {info['name']} ({info['native']}){rtl_indicator}")
    print()
    
    # Validate translations
    print("Translation validation:")
    validation = translation_manager.validate_translations()
    
    for lang, results in validation.items():
        completion = results["completion_rate"]
        missing = len(results["missing_keys"])
        print(f"  {lang}: {completion:.1f}% complete, {missing} missing keys")
    print()
    
    # Create template for new language
    print("Creating translation template...")
    template = translation_manager.create_translation_template()
    print(f"Template contains {len(template.split('\\n'))} lines")


async def main():
    """Main demonstration function."""
    # Configure logging to show i18n messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(name)s: %(message)s'
    )
    
    print("🌍 Global-First Autonomous SDLC i18n Demo\\n")
    
    # Demonstrate language switching
    await demonstrate_language_switching()
    
    # Demonstrate integrated systems
    await demonstrate_integrated_systems()
    
    # Demonstrate validation
    await demonstrate_validation()
    
    print("✅ i18n integration demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
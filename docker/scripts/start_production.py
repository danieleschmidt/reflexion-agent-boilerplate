#!/usr/bin/env python3
"""Production startup script for Reflexion Agent Framework."""

import os
import sys
import asyncio
import signal
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, '/app')

from reflexion.core.agent import ReflexionAgent
from reflexion.enterprise.governance import GovernanceFramework
from reflexion.enterprise.multi_tenant import MultiTenantManager
from reflexion.scaling.auto_scaler import AutoScaler
from reflexion.scaling.distributed import DistributedReflexionManager


class ProductionServer:
    """Production server for Reflexion Agent Framework."""
    
    def __init__(self):
        """Initialize production server."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.governance: Optional[GovernanceFramework] = None
        self.tenant_manager: Optional[MultiTenantManager] = None
        self.auto_scaler: Optional[AutoScaler] = None
        self.distributed_manager: Optional[DistributedReflexionManager] = None
        
        # Server state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        self.logger.info("Production server initialized")
    
    def setup_logging(self):
        """Set up production logging."""
        log_level = os.getenv('REFLEXION_LOG_LEVEL', 'INFO')
        log_dir = Path(os.getenv('REFLEXION_LOG_DIR', '/app/logs'))
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / 'reflexion.log')
            ]
        )
        
        # Configure specific loggers
        logging.getLogger('reflexion').setLevel(logging.INFO)
        logging.getLogger('uvicorn').setLevel(logging.INFO)
    
    async def initialize_components(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components...")
        
        try:
            # Initialize governance framework
            self.governance = GovernanceFramework({
                'audit_enabled': True,
                'compliance_monitoring': True,
                'security_enabled': os.getenv('REFLEXION_SECURITY_ENABLED', 'true').lower() == 'true'
            })
            self.logger.info("Governance framework initialized")
            
            # Initialize multi-tenant manager
            self.tenant_manager = MultiTenantManager({
                'default_tier': 'professional',
                'resource_monitoring': True
            })
            self.logger.info("Multi-tenant manager initialized")
            
            # Initialize distributed processing
            self.distributed_manager = DistributedReflexionManager({
                'worker_nodes': int(os.getenv('REFLEXION_WORKER_NODES', '3')),
                'load_balancing': 'least_loaded'
            })
            await self.distributed_manager.start()
            self.logger.info("Distributed processing manager initialized")
            
            # Initialize auto-scaler if enabled
            if os.getenv('REFLEXION_AUTO_SCALING', 'true').lower() == 'true':
                self.auto_scaler = AutoScaler([])
                default_policy = self.auto_scaler.create_default_policy(
                    min_instances=1,
                    max_instances=int(os.getenv('REFLEXION_MAX_INSTANCES', '10'))
                )
                self.auto_scaler.policies['production'] = default_policy
                
                # Register scaling callback
                self.auto_scaler.register_scaling_callback(self._handle_scaling_event)
                self.logger.info("Auto-scaler initialized")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    async def _handle_scaling_event(self, old_instances: int, new_instances: int):
        """Handle auto-scaling events."""
        self.logger.info(f"Scaling event: {old_instances} -> {new_instances} instances")
        
        # In a real implementation, this would:
        # - Scale worker nodes in distributed manager
        # - Update load balancer configuration
        # - Notify monitoring systems
        
        if new_instances > old_instances:
            # Scale up - add worker nodes
            for i in range(old_instances, new_instances):
                node_id = f"auto_worker_{i+1}"
                self.distributed_manager.add_worker_node(
                    node_id=node_id,
                    host="localhost",
                    port=8000 + i,
                    capacity=5
                )
                self.logger.info(f"Added worker node: {node_id}")
    
    async def start_web_server(self):
        """Start the web server."""
        try:
            import uvicorn
            from reflexion.api import create_app
            
            # Create FastAPI app with all components
            app = create_app(
                governance=self.governance,
                tenant_manager=self.tenant_manager,
                distributed_manager=self.distributed_manager
            )
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=int(os.getenv('REFLEXION_PORT', '8000')),
                log_config=None,  # Use our logging config
                access_log=True,
                loop="asyncio"
            )
            
            server = uvicorn.Server(config)
            
            self.logger.info("Starting web server...")
            await server.serve()
            
        except ImportError:
            self.logger.warning("FastAPI/uvicorn not available, running in standalone mode")
            await self._run_standalone()
    
    async def _run_standalone(self):
        """Run in standalone mode without web server."""
        self.logger.info("Running in standalone mode")
        
        # Keep the server running
        await self.shutdown_event.wait()
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown of all components."""
        if not self.running:
            return
        
        self.logger.info("Shutting down production server...")
        self.running = False
        
        try:
            # Stop auto-scaler
            if self.auto_scaler:
                self.auto_scaler.stop_monitoring()
                self.logger.info("Auto-scaler stopped")
            
            # Stop distributed manager
            if self.distributed_manager:
                await self.distributed_manager.stop()
                self.logger.info("Distributed manager stopped")
            
            # Generate final reports
            if self.governance:
                report = self.governance.generate_governance_report()
                self.logger.info(f"Final governance report generated with {len(report.get('recommendations', []))} recommendations")
            
            self.logger.info("Production server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        finally:
            self.shutdown_event.set()
    
    async def run(self):
        """Main server run method."""
        try:
            self.logger.info("Starting Reflexion Production Server")
            self.running = True
            
            # Initialize all components
            await self.initialize_components()
            
            # Set up signal handlers
            self.setup_signal_handlers()
            
            # Start monitoring tasks
            if self.auto_scaler:
                asyncio.create_task(self.auto_scaler.start_monitoring())
            
            # Start web server
            await self.start_web_server()
            
        except Exception as e:
            self.logger.error(f"Production server failed: {e}")
            await self.shutdown()
            sys.exit(1)


def main():
    """Main entry point."""
    # Validate environment
    required_vars = ['REFLEXION_ENV']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    # Create and run server
    server = ProductionServer()
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Server interrupted by user")
    except Exception as e:
        logging.getLogger(__name__).error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
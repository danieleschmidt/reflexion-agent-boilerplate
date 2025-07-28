#!/usr/bin/env python3
"""
Cleanup script for old artifacts, logs, and temporary files.
Used for maintenance and keeping the development environment clean.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_file_age_days(file_path: Path) -> int:
    """Get the age of a file in days."""
    try:
        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        return (datetime.now() - file_time).days
    except (OSError, ValueError):
        return 0


def cleanup_cache_directories(max_age_days: int = 7, dry_run: bool = False) -> Dict[str, Any]:
    """Clean up Python cache directories."""
    logger = logging.getLogger(__name__)
    results = {"removed_dirs": 0, "removed_files": 0, "freed_bytes": 0}
    
    cache_patterns = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".coverage*",
        "*.pyc",
        "*.pyo",
        "*.pyd"
    ]
    
    logger.info("Cleaning up cache directories and files...")
    
    for root, dirs, files in os.walk("."):
        root_path = Path(root)
        
        # Skip certain directories
        if any(skip in str(root_path) for skip in [".git", ".venv", "venv", "node_modules"]):
            continue
        
        # Remove cache directories
        for cache_dir in cache_patterns:
            if cache_dir in dirs:
                cache_path = root_path / cache_dir
                if get_file_age_days(cache_path) >= max_age_days:
                    try:
                        size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                        logger.debug(f"Found cache directory: {cache_path} ({size} bytes)")
                        
                        if not dry_run:
                            shutil.rmtree(cache_path)
                            logger.info(f"Removed cache directory: {cache_path}")
                        
                        results["removed_dirs"] += 1
                        results["freed_bytes"] += size
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not remove {cache_path}: {e}")
        
        # Remove cache files
        for file_name in files:
            if any(file_name.endswith(pattern.replace("*", "")) for pattern in cache_patterns if "*" in pattern):
                file_path = root_path / file_name
                if get_file_age_days(file_path) >= max_age_days:
                    try:
                        size = file_path.stat().st_size
                        logger.debug(f"Found cache file: {file_path} ({size} bytes)")
                        
                        if not dry_run:
                            file_path.unlink()
                            logger.info(f"Removed cache file: {file_path}")
                        
                        results["removed_files"] += 1
                        results["freed_bytes"] += size
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
    
    return results


def cleanup_build_artifacts(max_age_days: int = 30, dry_run: bool = False) -> Dict[str, Any]:
    """Clean up build artifacts."""
    logger = logging.getLogger(__name__)
    results = {"removed_dirs": 0, "removed_files": 0, "freed_bytes": 0}
    
    build_directories = [
        "build",
        "dist",
        "*.egg-info",
        ".tox",
        ".nox"
    ]
    
    logger.info("Cleaning up build artifacts...")
    
    for pattern in build_directories:
        if "*" in pattern:
            # Handle glob patterns
            for path in Path(".").glob(pattern):
                if path.is_dir() and get_file_age_days(path) >= max_age_days:
                    try:
                        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        logger.debug(f"Found build directory: {path} ({size} bytes)")
                        
                        if not dry_run:
                            shutil.rmtree(path)
                            logger.info(f"Removed build directory: {path}")
                        
                        results["removed_dirs"] += 1
                        results["freed_bytes"] += size
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not remove {path}: {e}")
        else:
            # Handle exact directory names
            path = Path(pattern)
            if path.exists() and path.is_dir() and get_file_age_days(path) >= max_age_days:
                try:
                    size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                    logger.debug(f"Found build directory: {path} ({size} bytes)")
                    
                    if not dry_run:
                        shutil.rmtree(path)
                        logger.info(f"Removed build directory: {path}")
                    
                    results["removed_dirs"] += 1
                    results["freed_bytes"] += size
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not remove {path}: {e}")
    
    return results


def cleanup_log_files(max_age_days: int = 14, max_size_mb: int = 100, dry_run: bool = False) -> Dict[str, Any]:
    """Clean up old log files."""
    logger = logging.getLogger(__name__)
    results = {"removed_files": 0, "truncated_files": 0, "freed_bytes": 0}
    
    log_patterns = ["*.log", "*.log.*"]
    log_directories = ["logs", "tmp", "temp", "/var/log"]
    
    logger.info("Cleaning up log files...")
    
    for log_dir in log_directories:
        log_path = Path(log_dir)
        if not log_path.exists():
            continue
        
        for pattern in log_patterns:
            for log_file in log_path.rglob(pattern):
                if not log_file.is_file():
                    continue
                
                try:
                    file_size = log_file.stat().st_size
                    file_age = get_file_age_days(log_file)
                    file_size_mb = file_size / (1024 * 1024)
                    
                    # Remove old log files
                    if file_age >= max_age_days:
                        logger.debug(f"Found old log file: {log_file} ({file_age} days old)")
                        
                        if not dry_run:
                            log_file.unlink()
                            logger.info(f"Removed old log file: {log_file}")
                        
                        results["removed_files"] += 1
                        results["freed_bytes"] += file_size
                    
                    # Truncate large log files
                    elif file_size_mb > max_size_mb:
                        logger.debug(f"Found large log file: {log_file} ({file_size_mb:.1f} MB)")
                        
                        if not dry_run:
                            # Keep only the last 1000 lines
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                            
                            with open(log_file, 'w') as f:
                                f.writelines(lines[-1000:])
                            
                            new_size = log_file.stat().st_size
                            freed = file_size - new_size
                            logger.info(f"Truncated log file: {log_file} (freed {freed} bytes)")
                        
                        results["truncated_files"] += 1
                        results["freed_bytes"] += file_size - new_size if not dry_run else file_size * 0.8
                
                except (OSError, PermissionError) as e:
                    logger.warning(f"Could not process log file {log_file}: {e}")
    
    return results


def cleanup_temporary_files(max_age_days: int = 1, dry_run: bool = False) -> Dict[str, Any]:
    """Clean up temporary files."""
    logger = logging.getLogger(__name__)
    results = {"removed_files": 0, "freed_bytes": 0}
    
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "*.swp",
        "*.swo"
    ]
    
    logger.info("Cleaning up temporary files...")
    
    for root, dirs, files in os.walk("."):
        root_path = Path(root)
        
        # Skip certain directories
        if any(skip in str(root_path) for skip in [".git", ".venv", "venv"]):
            continue
        
        for file_name in files:
            if any(file_name.endswith(pattern.replace("*", "")) or file_name == pattern.replace("*", "") 
                   for pattern in temp_patterns):
                file_path = root_path / file_name
                
                if get_file_age_days(file_path) >= max_age_days:
                    try:
                        size = file_path.stat().st_size
                        logger.debug(f"Found temporary file: {file_path} ({size} bytes)")
                        
                        if not dry_run:
                            file_path.unlink()
                            logger.info(f"Removed temporary file: {file_path}")
                        
                        results["removed_files"] += 1
                        results["freed_bytes"] += size
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
    
    return results


def cleanup_docker_artifacts(dry_run: bool = False) -> Dict[str, Any]:
    """Clean up Docker artifacts."""
    logger = logging.getLogger(__name__)
    results = {"commands_run": 0, "errors": 0}
    
    docker_commands = [
        ("docker system prune -f", "Remove unused Docker objects"),
        ("docker image prune -f", "Remove unused Docker images"),
        ("docker volume prune -f", "Remove unused Docker volumes"),
        ("docker network prune -f", "Remove unused Docker networks")
    ]
    
    logger.info("Cleaning up Docker artifacts...")
    
    for command, description in docker_commands:
        try:
            logger.debug(f"Running: {command} ({description})")
            
            if not dry_run:
                import subprocess
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"Completed: {description}")
                if result.stdout.strip():
                    logger.debug(f"Output: {result.stdout.strip()}")
            else:
                logger.info(f"Would run: {command}")
            
            results["commands_run"] += 1
        
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to run '{command}': {e}")
            results["errors"] += 1
    
    return results


def format_bytes(bytes_count: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Cleanup old artifacts and temporary files")
    parser.add_argument("--cache-age", type=int, default=7, help="Max age for cache files in days")
    parser.add_argument("--build-age", type=int, default=30, help="Max age for build artifacts in days")
    parser.add_argument("--log-age", type=int, default=14, help="Max age for log files in days")
    parser.add_argument("--temp-age", type=int, default=1, help="Max age for temporary files in days")
    parser.add_argument("--log-size", type=int, default=100, help="Max size for log files in MB")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without doing it")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be actually removed")
    
    total_results = {
        "removed_dirs": 0,
        "removed_files": 0,
        "truncated_files": 0,
        "freed_bytes": 0,
        "commands_run": 0,
        "errors": 0
    }
    
    # Run cleanup operations
    cleanup_operations = [
        ("Cache cleanup", lambda: cleanup_cache_directories(args.cache_age, args.dry_run)),
        ("Build artifacts cleanup", lambda: cleanup_build_artifacts(args.build_age, args.dry_run)),
        ("Log files cleanup", lambda: cleanup_log_files(args.log_age, args.log_size, args.dry_run)),
        ("Temporary files cleanup", lambda: cleanup_temporary_files(args.temp_age, args.dry_run)),
    ]
    
    if not args.skip_docker:
        cleanup_operations.append(("Docker cleanup", lambda: cleanup_docker_artifacts(args.dry_run)))
    
    for operation_name, operation_func in cleanup_operations:
        logger.info(f"Starting {operation_name}...")
        try:
            results = operation_func()
            for key, value in results.items():
                total_results[key] = total_results.get(key, 0) + value
            logger.info(f"Completed {operation_name}")
        except Exception as e:
            logger.error(f"Error during {operation_name}: {e}")
            total_results["errors"] += 1
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("CLEANUP SUMMARY")
    logger.info("="*50)
    logger.info(f"Directories removed: {total_results['removed_dirs']}")
    logger.info(f"Files removed: {total_results['removed_files']}")
    logger.info(f"Files truncated: {total_results.get('truncated_files', 0)}")
    logger.info(f"Space freed: {format_bytes(total_results['freed_bytes'])}")
    logger.info(f"Commands run: {total_results.get('commands_run', 0)}")
    logger.info(f"Errors encountered: {total_results['errors']}")
    
    if args.dry_run:
        logger.info("\nNote: This was a dry run. Use without --dry-run to actually clean files.")
    
    return 0 if total_results["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
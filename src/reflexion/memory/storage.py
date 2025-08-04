"""Simple storage backend for reflexion memory systems."""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """Store data with the given key."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        pass
    
    @abstractmethod
    def query(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query data with filters."""
        pass


class JSONMemoryBackend(MemoryBackend):
    """Simple JSON file-based memory backend."""
    
    def __init__(self, storage_path: str = "./reflexion_memory.json"):
        self.storage_path = Path(storage_path)
        self.data = self._load_data()
    
    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """Store data with the given key."""
        try:
            self.data[key] = data
            self._save_data()
            return True
        except Exception:
            return False
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        return self.data.get(key)
    
    def query(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query data with simple filters."""
        results = []
        for key, item in self.data.items():
            match = True
            for filter_key, filter_value in filters.items():
                if filter_key not in item or item[filter_key] != filter_value:
                    match = False
                    break
            if match:
                results.append(item)
        return results
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_data(self):
        """Save data to JSON file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save data to {self.storage_path}: {e}")


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite-based memory backend for better performance and querying."""
    
    def __init__(self, storage_path: str = "./reflexion_memory.db"):
        self.storage_path = storage_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """Store data with the given key."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                data_json = json.dumps(data)
                conn.execute("""
                    INSERT OR REPLACE INTO memory (key, data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (key, data_json))
                conn.commit()
                return True
        except Exception:
            return False
    
    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("SELECT data FROM memory WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
        except Exception:
            return None
    
    def query(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query data with filters (simplified implementation)."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("SELECT data FROM memory")
                results = []
                for row in cursor.fetchall():
                    item = json.loads(row[0])
                    match = True
                    for filter_key, filter_value in filters.items():
                        if filter_key not in item or item[filter_key] != filter_value:
                            match = False
                            break
                    if match:
                        results.append(item)
                return results
        except Exception:
            return []


class MemoryStore:
    """High-level interface for memory storage operations."""
    
    def __init__(self, backend: Optional[MemoryBackend] = None):
        self.backend = backend or JSONMemoryBackend()
    
    def store_reflection(self, task_id: str, reflection: Dict[str, Any]) -> bool:
        """Store a reflection with task-based key."""
        key = f"reflection:{task_id}"
        return self.backend.store(key, reflection)
    
    def get_reflections_for_task(self, task_pattern: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get reflections matching a task pattern."""
        # Simple implementation - in production would use more sophisticated matching
        all_reflections = self.backend.query({"type": "reflection"})
        
        # Filter by task pattern (simple keyword matching)
        matching = []
        pattern_words = set(task_pattern.lower().split())
        
        for reflection in all_reflections:
            task = reflection.get("task", "").lower()
            task_words = set(task.split())
            if pattern_words.intersection(task_words):
                matching.append(reflection)
        
        return matching[:limit]
    
    def store_episode(self, episode_id: str, episode_data: Dict[str, Any]) -> bool:
        """Store an execution episode."""
        key = f"episode:{episode_id}"
        episode_data["type"] = "episode"
        return self.backend.store(key, episode_data)
    
    def get_success_episodes(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get successful episodes for pattern analysis."""
        episodes = self.backend.query({"type": "episode", "success": True})
        return episodes[:limit]
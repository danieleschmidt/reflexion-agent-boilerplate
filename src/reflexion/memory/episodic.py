"""Episodic memory implementation for reflexion agents."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..core.types import Reflection, ReflexionResult


class Episode:
    """Represents a single execution episode."""
    
    def __init__(self, task: str, result: ReflexionResult, metadata: Optional[Dict] = None):
        self.task = task
        self.result = result
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.episode_id = f"{hash(task + self.timestamp)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for storage."""
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "success": self.result.success,
            "iterations": self.result.iterations,
            "reflections": [
                {
                    "issues": r.issues,
                    "improvements": r.improvements,
                    "confidence": r.confidence,
                    "score": r.score
                } for r in self.result.reflections
            ],
            "total_time": self.result.total_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class EpisodicMemory:
    """Simple episodic memory system for storing and retrieving execution episodes."""
    
    def __init__(self, storage_path: str = "./memory.json", max_episodes: int = 1000):
        self.storage_path = Path(storage_path)
        self.max_episodes = max_episodes
        self.episodes: List[Episode] = []
        self._load_episodes()
    
    def store_episode(self, task: str, result: ReflexionResult, metadata: Optional[Dict] = None):
        """Store a new execution episode."""
        episode = Episode(task, result, metadata)
        self.episodes.append(episode)
        
        # Maintain max episodes limit
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
        
        self._save_episodes()
    
    def recall_similar(self, task: str, k: int = 5) -> List[Episode]:
        """Recall similar episodes based on task similarity."""
        # Simple keyword-based similarity matching
        task_words = set(task.lower().split())
        
        scored_episodes = []
        for episode in self.episodes:
            episode_words = set(episode.task.lower().split())
            similarity = len(task_words.intersection(episode_words)) / max(len(task_words), 1)
            scored_episodes.append((similarity, episode))
        
        # Sort by similarity and return top k
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [episode for _, episode in scored_episodes[:k]]
    
    def get_success_patterns(self) -> Dict[str, Any]:
        """Extract patterns from successful episodes."""
        successful_episodes = [ep for ep in self.episodes if ep.result.success]
        
        if not successful_episodes:
            return {"patterns": [], "success_rate": 0.0}
        
        # Analyze common improvements in successful episodes
        all_improvements = []
        for episode in successful_episodes:
            for reflection in episode.result.reflections:
                all_improvements.extend(reflection.improvements)
        
        # Count improvement frequency
        improvement_counts = {}
        for improvement in all_improvements:
            improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1
        
        # Sort by frequency
        common_patterns = sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "patterns": common_patterns,
            "success_rate": len(successful_episodes) / len(self.episodes),
            "total_episodes": len(self.episodes),
            "successful_episodes": len(successful_episodes)
        }
    
    def _load_episodes(self):
        """Load episodes from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                # For simplicity, we'll just store the episode data without full reconstruction
                self.episodes = []  # Start fresh each time for now
        except (json.JSONDecodeError, FileNotFoundError):
            self.episodes = []
    
    def _save_episodes(self):
        """Save episodes to storage."""
        try:
            with open(self.storage_path, 'w') as f:
                episodes_data = [ep.to_dict() for ep in self.episodes]
                json.dump(episodes_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save episodes to {self.storage_path}: {e}")
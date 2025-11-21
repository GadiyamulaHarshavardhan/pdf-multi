"""
Memory management system for multi-agent workflow
"""
import json
import pickle
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
from collections import defaultdict
import asyncio
from memory_profiler import profile

class StateManager:
    """
    Advanced state management system with memory optimization and workflow tracking
    """
    def __init__(self, max_memory_size: int = 1000000000):  # 1GB default
        self.max_memory_size = max_memory_size
        self._states: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._workflow_history: List[Dict[str, Any]] = []
        self._agent_memory_usage = defaultdict(int)
        
    def store_state(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Store state with memory management"""
        with self._lock:
            # Calculate size of new value
            size = len(pickle.dumps(value))
            
            # Check if adding this would exceed memory limits
            current_size = sum(len(pickle.dumps(v)) for v in self._states.values())
            if current_size + size > self.max_memory_size:
                # Evict oldest entries to make space
                self._evict_old_states(size)
            
            self._states[key] = value
            self._metadata[key] = metadata or {}
            self._metadata[key]['timestamp'] = datetime.now().isoformat()
            self._metadata[key]['size'] = size
            
    def retrieve_state(self, key: str) -> Optional[Any]:
        """Retrieve state by key"""
        with self._lock:
            return self._states.get(key)
    
    def update_state(self, key: str, value: Any):
        """Update existing state"""
        with self._lock:
            if key in self._states:
                old_size = len(pickle.dumps(self._states[key]))
                new_size = len(pickle.dumps(value))
                self._states[key] = value
                self._metadata[key]['size'] = new_size
                self._metadata[key]['last_updated'] = datetime.now().isoformat()
    
    def delete_state(self, key: str):
        """Delete state by key"""
        with self._lock:
            if key in self._states:
                del self._states[key]
                if key in self._metadata:
                    del self._metadata[key]
    
    def _evict_old_states(self, needed_size: int):
        """Evict oldest states to free memory"""
        sorted_keys = sorted(
            self._states.keys(),
            key=lambda k: self._metadata[k].get('timestamp', datetime.min.isoformat())
        )
        
        current_size = sum(len(pickle.dumps(v)) for v in self._states.values())
        target_size = self.max_memory_size - needed_size
        
        # Remove oldest entries until under target size
        while current_size > target_size and sorted_keys:
            key = sorted_keys.pop(0)
            size = len(pickle.dumps(self._states[key]))
            del self._states[key]
            if key in self._metadata:
                del self._metadata[key]
            current_size -= size
    
    def get_recent_states(self, limit: int = 10) -> Dict[str, Any]:
        """Get most recent states"""
        with self._lock:
            sorted_keys = sorted(
                self._states.keys(),
                key=lambda k: self._metadata[k].get('timestamp', datetime.min.isoformat()),
                reverse=True
            )
            return {k: self._states[k] for k in sorted_keys[:limit]}
    
    def save_to_disk(self, filepath: str):
        """Save all states to disk"""
        with self._lock:
            data = {
                'states': self._states,
                'metadata': self._metadata,
                'workflow_history': self._workflow_history
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
    
    def load_from_disk(self, filepath: str):
        """Load all states from disk"""
        with self._lock:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self._states = data.get('states', {})
                    self._metadata = data.get('metadata', {})
                    self._workflow_history = data.get('workflow_history', [])
    
    def add_workflow_history(self, entry: Dict[str, Any]):
        """Add workflow execution history"""
        with self._lock:
            entry['timestamp'] = datetime.now().isoformat()
            self._workflow_history.append(entry)
            # Keep only last 1000 entries
            if len(self._workflow_history) > 1000:
                self._workflow_history = self._workflow_history[-1000:]
    
    def get_workflow_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent workflow history"""
        with self._lock:
            return self._workflow_history[-limit:]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        with self._lock:
            total_size = sum(len(pickle.dumps(v)) for v in self._states.values())
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'state_count': len(self._states),
                'max_size_bytes': self.max_memory_size,
                'max_size_mb': self.max_memory_size / (1024 * 1024),
                'usage_percentage': (total_size / self.max_memory_size) * 100 if self.max_memory_size > 0 else 0
            }
    
    def cleanup(self):
        """Cleanup resources"""
        with self._lock:
            self._states.clear()
            self._metadata.clear()
            self._workflow_history.clear()


class MemoryOptimizedWebCrawler:
    """
    Memory-optimized web crawler with advanced caching and state management
    """
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.visited_urls = set()
        self.url_cache = {}
        self.page_content_cache = {}
        self.max_cache_size = 1000  # Maximum URLs to cache
        
    def add_visited_url(self, url: str):
        """Add URL to visited set and manage memory"""
        self.visited_urls.add(url)
        
        # If cache is too large, clean up oldest entries
        if len(self.visited_urls) > self.max_cache_size:
            # Convert to list and remove oldest half
            urls_list = list(self.visited_urls)
            self.visited_urls = set(urls_list[len(urls_list)//2:])
    
    def is_visited(self, url: str) -> bool:
        """Check if URL has been visited"""
        return url in self.visited_urls
    
    def cache_page_content(self, url: str, content: str):
        """Cache page content with memory management"""
        self.page_content_cache[url] = content
        
        # Manage cache size
        if len(self.page_content_cache) > self.max_cache_size:
            # Remove oldest entries (by insertion order in Python 3.7+)
            oldest_keys = list(self.page_content_cache.keys())[:len(self.page_content_cache)//2]
            for key in oldest_keys:
                del self.page_content_cache[key]
    
    def get_cached_content(self, url: str) -> Optional[str]:
        """Get cached page content"""
        return self.page_content_cache.get(url)
    
    def clear_cache(self):
        """Clear all caches"""
        self.page_content_cache.clear()
        self.visited_urls.clear()


class WorkflowStateManager:
    """
    State manager specifically for workflow operations
    """
    def __init__(self):
        self.state_manager = StateManager()
        self.current_workflow_id = None
        self.agent_states = {}
        
    def start_workflow(self, workflow_id: str, initial_state: Dict[str, Any] = None):
        """Start a new workflow"""
        self.current_workflow_id = workflow_id
        state = initial_state or {}
        state['workflow_id'] = workflow_id
        state['start_time'] = datetime.now().isoformat()
        self.state_manager.store_state(f"workflow_{workflow_id}", state)
        
        # Add to history
        self.state_manager.add_workflow_history({
            'workflow_id': workflow_id,
            'action': 'started',
            'state': state
        })
    
    def update_workflow_state(self, state_update: Dict[str, Any]):
        """Update current workflow state"""
        if not self.current_workflow_id:
            raise ValueError("No workflow started")
        
        # Retrieve current state
        current_state = self.state_manager.retrieve_state(f"workflow_{self.current_workflow_id}")
        if not current_state:
            current_state = {}
        
        # Update with new values
        current_state.update(state_update)
        current_state['last_updated'] = datetime.now().isoformat()
        
        # Store updated state
        self.state_manager.store_state(f"workflow_{self.current_workflow_id}", current_state)
        
        # Add to history
        self.state_manager.add_workflow_history({
            'workflow_id': self.current_workflow_id,
            'action': 'updated',
            'update': state_update
        })
    
    def get_workflow_state(self, workflow_id: str = None) -> Optional[Dict[str, Any]]:
        """Get workflow state"""
        wf_id = workflow_id or self.current_workflow_id
        if not wf_id:
            return None
        return self.state_manager.retrieve_state(f"workflow_{wf_id}")
    
    def complete_workflow(self, final_state: Dict[str, Any] = None):
        """Complete current workflow"""
        if not self.current_workflow_id:
            raise ValueError("No workflow started")
        
        if final_state:
            self.update_workflow_state(final_state)
        
        # Add to history
        self.state_manager.add_workflow_history({
            'workflow_id': self.current_workflow_id,
            'action': 'completed',
            'final_state': final_state
        })
        
        # Clean up
        self.current_workflow_id = None


# Global state manager instance
global_state_manager = StateManager()
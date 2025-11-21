"""
Synchronous wrapper for ClassifierAgent
"""

import asyncio
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

class SyncClassifierAgent:
    def __init__(self, llm=None):
        # Import the specific agent directly
        try:
            from agents.classifier_agent import ClassifierAgent
        except ImportError:
            from classifier_agent import ClassifierAgent
        
        # Try different constructor signatures
        try:
            # Try without LLM first
            self.async_agent = ClassifierAgent()
        except TypeError:
            # If that fails, try with LLM
            if llm:
                self.async_agent = ClassifierAgent(llm)
            else:
                raise ValueError("ClassifierAgent requires LLM but none provided")
                
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
    
    def execute(self, state: Dict) -> Dict:
        """Synchronous execute method"""
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.async_agent.execute(state))
            finally:
                loop.close()
        
        return self.thread_pool.submit(run_in_thread).result()
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
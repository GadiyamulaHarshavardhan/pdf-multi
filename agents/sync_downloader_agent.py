"""
Synchronous wrapper for DownloaderAgent
"""

import asyncio
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

class SyncDownloaderAgent:
    def __init__(self, llm=None):
        # Import the specific agent directly
        try:
            from agents.downloader_agent import DownloaderAgent
        except ImportError:
            from downloader_agent import DownloaderAgent
        
        # Try different constructor signatures
        try:
            # Try without LLM first
            self.async_agent = DownloaderAgent()
        except TypeError:
            # If that fails, try with LLM
            if llm:
                self.async_agent = DownloaderAgent(llm)
            else:
                # If no LLM provided but needed, create a dummy one or re-raise
                raise ValueError("DownloaderAgent requires LLM but none provided")
                
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
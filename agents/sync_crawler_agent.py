"""
Synchronous wrapper for CrawlerAgent
"""

import asyncio
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from agents.crawler_agent import CrawlerAgent
from .autonomous_agent import AutonomousAgent

class SyncCrawlerAgent:
    def __init__(self, llm=None):
        if llm:
            self.async_agent = CrawlerAgent(llm)
        else:
            self.async_agent = CrawlerAgent()
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
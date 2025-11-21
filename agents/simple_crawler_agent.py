"""
Simplified crawler agent without abstract methods for quick testing
"""

from typing import Dict, Any, List, Set
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import json
from datetime import datetime

class SimpleCrawlerAgent:
    def __init__(self, llm=None):
        self.llm = llm
        self.visited_urls: Set[str] = set()
        self.pdf_urls: List[str] = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger("SimpleCrawlerAgent")
        handler = logging.FileHandler('logs/simple_crawler_agent.jsonl')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def log_action(self, action: str, data: Dict, status: str, error: str = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": "simple_crawler",
            "action": action,
            "data": data,
            "status": status,
            "error": error
        }
        self.logger.info(json.dumps(log_entry))
    
    async def execute(self, state: Dict) -> Dict:
        """Simple execute method without abstract requirements"""
        base_url = state.get('base_url')
        max_depth = state.get('max_depth', 2)
        
        # Reset state
        self.visited_urls.clear()
        self.pdf_urls.clear()
        
        # Simple crawling
        await self._simple_crawl(base_url, max_depth)
        
        return {
            "crawled_urls_count": len(self.visited_urls),
            "pdf_urls_found": self.pdf_urls,
            "pdf_count": len(self.pdf_urls),
            "execution_summary": f"Found {len(self.pdf_urls)} PDFs from {len(self.visited_urls)} pages"
        }
    
    async def _simple_crawl(self, url: str, max_depth: int, current_depth: int = 0):
        """Simple recursive crawling"""
        if current_depth > max_depth or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        print(f"🔍 Crawling {url} (depth {current_depth})")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        links = set()
                        
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(url, href)
                            if self._is_same_domain(full_url, url):
                                links.add(full_url)
                                
                                # Check for PDF
                                if full_url.lower().endswith('.pdf'):
                                    self.pdf_urls.append(full_url)
                        
                        # Crawl child pages
                        if current_depth < max_depth:
                            for link in links:
                                if link not in self.visited_urls:
                                    await self._simple_crawl(link, max_depth, current_depth + 1)
                                    await asyncio.sleep(1)  # Rate limiting
                    
        except Exception as e:
            print(f"❌ Crawling failed for {url}: {e}")
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain"""
        try:
            domain1 = urlparse(url1).netloc
            domain2 = urlparse(url2).netloc
            return domain1 == domain2
        except:
            return False
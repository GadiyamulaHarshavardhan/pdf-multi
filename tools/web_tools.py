import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
from typing import List, Set
import logging

class WebTools:
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger("WebTools")
    
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> str:
        """Fetch webpage content asynchronously"""
        try:
            session = await self.get_session()
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise
    
    def extract_internal_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all internal links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Check if it's an internal link
            if urlparse(full_url).netloc == base_domain:
                links.add(full_url)
        
        return links
    
    def filter_pdf_links(self, links: Set[str]) -> List[str]:
        """Filter links that point to PDF files"""
        pdf_links = []
        for link in links:
            if link.lower().endswith('.pdf'):
                pdf_links.append(link)
            # Also check content-disposition headers in actual implementation
        return pdf_links
    
    async def download_pdf(self, url: str, filepath: str) -> bool:
        """Download PDF file asynchronously"""
        try:
            session = await self.get_session()
            async with session.get(url) as response:
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024):
                        f.write(chunk)
                
                return True
        except Exception as e:
            self.logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return False
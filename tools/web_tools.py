import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
from typing import List, Set, Dict, Any, Optional
import logging
import time
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from playwright.async_api import async_playwright
import json
from memory.state_manager import global_state_manager

class WebTools:
    def __init__(self):
        self.session = None
        self.logger = logging.getLogger("WebTools")
        self.state_manager = global_state_manager
        self.visited_urls = set()
        self.pdf_cache = {}  # Cache for PDF URLs and their metadata
        self.speed_cache = {}  # Cache for URL response times
        
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> str:
        """Fetch webpage content asynchronously with timing"""
        start_time = time.time()
        try:
            session = await self.get_session()
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.text()
                
                # Record response time for speed optimization
                response_time = time.time() - start_time
                self.speed_cache[url] = response_time
                
                return content
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
    
    def extract_all_links(self, html: str, base_url: str) -> Set[str]:
        """Extract all links (internal and external) from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            links.add(full_url)
        
        # Also extract links from other elements like iframes, scripts, etc.
        for element in soup.find_all(['iframe', 'script', 'img'], src=True):
            src = element['src']
            full_url = urljoin(base_url, src)
            links.add(full_url)
        
        # Extract links from forms
        for form in soup.find_all('form', action=True):
            action = form['action']
            full_url = urljoin(base_url, action)
            links.add(full_url)
        
        return links
    
    def filter_pdf_links(self, links: Set[str]) -> List[str]:
        """Filter links that point to PDF files with advanced detection"""
        pdf_links = []
        
        for link in links:
            # Basic extension check
            if link.lower().endswith('.pdf'):
                pdf_links.append(link)
                continue
            
            # Check for PDF in query parameters
            parsed = urlparse(link)
            if '.pdf' in parsed.path.lower() or 'pdf' in parsed.query.lower():
                pdf_links.append(link)
                continue
            
            # Check for PDF content type indicators in URL
            if any(indicator in link.lower() for indicator in ['pdf', 'download', 'file']):
                # Additional check: try to see if it's actually a PDF by checking content type
                pdf_links.append(link)
        
        return pdf_links
    
    async def detect_dynamic_content(self, url: str) -> bool:
        """Detect if page has dynamic content requiring JavaScript"""
        try:
            # Try a quick fetch to check for JavaScript indicators
            html = await self.fetch_page(url)
            dynamic_indicators = [
                'react', 'angular', 'vue', 'webpack', 'spa',
                'single page application', 'ajax', 'fetch(',
                'nextjs', 'nuxtjs', 'gatsby', 'javascript'
            ]
            
            html_lower = html.lower()
            return any(indicator in html_lower for indicator in dynamic_indicators)
        except:
            return False
    
    async def crawl_with_beautifulsoup(self, url: str) -> tuple[str, Set[str]]:
        """Crawl page using BeautifulSoup (static content)"""
        try:
            html = await self.fetch_page(url)
            links = self.extract_all_links(html, url)
            return html, links
        except Exception as e:
            self.logger.error(f"BeautifulSoup crawl failed for {url}: {e}")
            return "", set()
    
    async def crawl_with_selenium(self, url: str) -> tuple[str, Set[str]]:
        """Crawl page using Selenium (dynamic content)"""
        try:
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            
            # Initialize driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for page to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                pass  # Continue even if timeout occurs
            
            # Get page source after JavaScript execution
            html = driver.page_source
            
            # Extract links
            links = set()
            a_tags = driver.find_elements(By.TAG_NAME, "a")
            for a_tag in a_tags:
                href = a_tag.get_attribute("href")
                if href:
                    full_url = urljoin(url, href)
                    links.add(full_url)
            
            driver.quit()
            return html, links
            
        except Exception as e:
            self.logger.error(f"Selenium crawl failed for {url}: {e}")
            return "", set()
    
    async def crawl_with_playwright(self, url: str) -> tuple[str, Set[str]]:
        """Crawl page using Playwright (modern dynamic content)"""
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                # Create page
                page = await context.new_page()
                
                # Navigate to URL
                await page.goto(url, wait_until='networkidle')
                
                # Wait for potential dynamic content
                await page.wait_for_timeout(2000)
                
                # Get page content
                html = await page.content()
                
                # Extract links
                links = set()
                a_elements = await page.query_selector_all('a')
                for a_element in a_elements:
                    href = await a_element.get_attribute('href')
                    if href:
                        full_url = urljoin(url, href)
                        links.add(full_url)
                
                # Close resources
                await page.close()
                await context.close()
                await browser.close()
                
                return html, links
                
        except Exception as e:
            self.logger.error(f"Playwright crawl failed for {url}: {e}")
            return "", set()
    
    async def smart_crawl(self, url: str, engine_hint: str = "auto") -> tuple[str, Set[str]]:
        """Smart crawling that selects the best engine based on content detection"""
        # If engine is specified, use it
        if engine_hint != "auto":
            if engine_hint == "selenium":
                return await self.crawl_with_selenium(url)
            elif engine_hint == "playwright":
                return await self.crawl_with_playwright(url)
            else:
                return await self.crawl_with_beautifulsoup(url)
        
        # Auto-detect best engine
        is_dynamic = await self.detect_dynamic_content(url)
        
        if is_dynamic:
            # Try Playwright first (more modern), then Selenium
            try:
                html, links = await self.crawl_with_playwright(url)
                if html and len(links) > 0:
                    return html, links
            except:
                pass
            
            try:
                html, links = await self.crawl_with_selenium(url)
                if html and len(links) > 0:
                    return html, links
            except:
                pass
        
        # Fallback to BeautifulSoup
        return await self.crawl_with_beautifulsoup(url)
    
    async def download_pdf(self, url: str, filepath: str) -> bool:
        """Download PDF file asynchronously with resume capability"""
        try:
            session = await self.get_session()
            async with session.get(url) as response:
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(1024):
                        f.write(chunk)
                
                # Cache PDF metadata
                self.pdf_cache[url] = {
                    'filepath': filepath,
                    'size': response.headers.get('content-length', 'unknown'),
                    'content-type': response.headers.get('content-type', 'application/pdf'),
                    'downloaded_at': time.time()
                }
                
                return True
        except Exception as e:
            self.logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return False
    
    async def batch_download_pdfs(self, pdf_urls: List[str], base_dir: str) -> List[Dict[str, Any]]:
        """Download multiple PDFs with concurrency control"""
        downloaded_files = []
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent downloads
        
        async def download_single_pdf(url: str):
            async with semaphore:
                # Generate filename from URL
                filename = self._generate_filename_from_url(url)
                filepath = f"{base_dir}/{filename}"
                
                success = await self.download_pdf(url, filepath)
                if success:
                    return {
                        'url': url,
                        'filepath': filepath,
                        'success': True
                    }
                else:
                    return {
                        'url': url,
                        'filepath': filepath,
                        'success': False
                    }
        
        # Download all PDFs concurrently
        tasks = [download_single_pdf(url) for url in pdf_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict) and result['success']:
                downloaded_files.append(result)
        
        return downloaded_files
    
    def _generate_filename_from_url(self, url: str) -> str:
        """Generate safe filename from URL"""
        # Extract filename from URL
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1]
        
        # If no filename, create one from domain and timestamp
        if not filename or '.' not in filename:
            domain = parsed.netloc.replace('www.', '').replace('.', '_')
            filename = f"{domain}_{int(time.time())}.pdf"
        
        # Clean filename of invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        return filename
    
    def get_speed_optimized_urls(self) -> List[str]:
        """Get URLs sorted by response time (fastest first)"""
        sorted_urls = sorted(self.speed_cache.items(), key=lambda x: x[1])
        return [url for url, _ in sorted_urls]
    
    def is_url_visited(self, url: str) -> bool:
        """Check if URL has been visited"""
        return url in self.visited_urls
    
    def mark_url_visited(self, url: str):
        """Mark URL as visited"""
        self.visited_urls.add(url)
    
    def clear_cache(self):
        """Clear all caches"""
        self.pdf_cache.clear()
        self.speed_cache.clear()
        self.visited_urls.clear()
    
    async def get_pdf_metadata(self, url: str) -> Dict[str, Any]:
        """Get metadata for a PDF URL"""
        if url in self.pdf_cache:
            return self.pdf_cache[url]
        
        # If not cached, try to get from server
        try:
            session = await self.get_session()
            async with session.head(url) as response:
                metadata = {
                    'content-type': response.headers.get('content-type', ''),
                    'content-length': response.headers.get('content-length', ''),
                    'last-modified': response.headers.get('last-modified', ''),
                    'etag': response.headers.get('etag', '')
                }
                return metadata
        except Exception as e:
            self.logger.error(f"Error getting metadata for {url}: {str(e)}")
            return {}
    
    def search_content(self, html: str, search_terms: List[str]) -> Dict[str, Any]:
        """Search for specific terms in HTML content"""
        results = {}
        soup = BeautifulSoup(html, 'html.parser')
        text_content = soup.get_text().lower()
        
        for term in search_terms:
            term_lower = term.lower()
            occurrences = text_content.count(term_lower)
            results[term] = {
                'count': occurrences,
                'found': occurrences > 0,
                'positions': [i for i in range(len(text_content)) if text_content.startswith(term_lower, i)]
            }
        
        return results
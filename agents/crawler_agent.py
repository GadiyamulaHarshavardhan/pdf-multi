from .autonomous_agent import AutonomousAgent
from typing import Dict, Any, List, Set
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import json
from datetime import datetime
import time

class CrawlerAgent(AutonomousAgent):
    def __init__(self, llm):
        super().__init__(
            name="crawler_agent",
            role="Web Crawler",
            goal="Find all internal links leading to PDFs using multiple crawling engines"
        )
        self.llm = llm
        self.visited_urls: Set[str] = set()
        self.pdf_urls: List[str] = []
        self.web_tools = WebTools()
        
        # Available crawling engines
        self.crawling_engines = {
            'beautifulsoup': BeautifulSoupCrawler(),
            'selenium': SeleniumCrawler(),
            'playwright': PlaywrightCrawler()
        }
    
    async def _analyze_problem(self, context: Dict) -> Dict:
        """Analyze the crawling task and requirements"""
        base_url = context.get('base_url', 'Unknown')
        max_depth = context.get('max_depth', 2)
        
        return {
            "task": "crawl_website_for_pdfs",
            "target_url": base_url,
            "max_depth": max_depth,
            "crawling_strategy": "recursive_breadth_first",
            "engine_selection": "adaptive_based_on_content",
            "rate_limiting": "respectful_crawling"
        }
    
    async def _create_execution_plan(self, analysis: Dict) -> Dict:
        """Create execution plan for crawling"""
        target_url = analysis.get('target_url', '')
        max_depth = analysis.get('max_depth', 2)
        
        # Determine initial engine based on URL pattern
        initial_engine = self._suggest_initial_engine(target_url)
        
        return {
            "strategy": "adaptive_crawling",
            "initial_engine": initial_engine,
            "max_depth": max_depth,
            "batch_size": 3,
            "rate_limit_delay": 1,
            "fallback_engines": ["beautifulsoup", "selenium", "playwright"],
            "pdf_detection_methods": ["url_extension", "content_type", "link_analysis"]
        }
    
    async def _self_check_plan(self, plan: Dict) -> Dict:
        """Validate the crawling plan"""
        issues = []
        
        if plan["max_depth"] > 5:
            issues.append("Max depth too high, may crawl too many pages")
        
        if plan["rate_limit_delay"] < 0.5:
            issues.append("Rate limit too aggressive, may overwhelm server")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Monitor server responses", "Respect robots.txt"]
        }
    
    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute the crawling plan"""
        # Extract context from plan
        base_url = plan.get('base_url', '')
        max_depth = plan.get('max_depth', 2)
        initial_engine = plan.get('initial_engine', 'beautifulsoup')
        
        # Reset state for new crawl
        self.visited_urls.clear()
        self.pdf_urls.clear()
        
        # First, reason about the task and choose engine
        reasoning = await self.think_and_reason({
            'base_url': base_url,
            'max_depth': max_depth
        })
        recommended_engine = reasoning.get('analysis', {}).get('recommended_engine', initial_engine)
        
        # Execute crawling
        await self._crawl_recursive(base_url, max_depth, 0, recommended_engine)
        
        return {
            "reasoning": reasoning,
            "crawling_engine_used": recommended_engine,
            "crawled_urls_count": len(self.visited_urls),
            "pdf_urls_found": self.pdf_urls,
            "pdf_count": len(self.pdf_urls),
            "execution_summary": f"Found {len(self.pdf_urls)} PDFs from {len(self.visited_urls)} pages using {recommended_engine}"
        }
    
    async def _verify_result(self, result: Dict) -> Dict:
        """Verify crawling results"""
        issues = []
        
        if result["pdf_count"] == 0:
            issues.append("No PDFs found during crawling")
        
        if result["crawled_urls_count"] == 0:
            issues.append("No URLs were crawled")
        
        if len(result.get("pdf_urls_found", [])) != result["pdf_count"]:
            issues.append("PDF count mismatch")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Try different crawling engine", "Increase max depth", "Check website structure"]
        }
    
    async def _debug_and_adapt(self, plan: Dict, error: Exception):
        """Debug and adapt crawling strategy based on errors"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            plan["rate_limit_delay"] += 1  # Increase delay
            self.log_action("adapt_strategy", {"action": "increase_delay"}, "timeout_encountered")
        
        if "javascript" in error_str or "dynamic" in error_str:
            # Switch to dynamic content engine
            plan["initial_engine"] = "selenium"
            self.log_action("adapt_strategy", {"action": "switch_to_selenium"}, "dynamic_content_detected")
        
        if "404" in error_str or "not found" in error_str:
            self.log_action("adapt_strategy", {"action": "skip_broken_links"}, "broken_links_found")
    
    def _suggest_initial_engine(self, url: str) -> str:
        """Suggest initial crawling engine based on URL patterns"""
        # Known JavaScript-heavy sites
        js_sites = ['react', 'angular', 'vue', 'spa', 'webpack', 'nextjs']
        url_lower = url.lower()
        
        if any(site in url_lower for site in js_sites):
            return "selenium"
        
        # Modern web apps often use these patterns
        modern_patterns = ['app.', 'dashboard', 'admin', 'console']
        if any(pattern in url_lower for pattern in modern_patterns):
            return "playwright"
        
        # Default to beautifulsoup for most sites
        return "beautifulsoup"
    
    async def think_and_reason(self, context: Dict) -> Dict:
        """Use Ollama for reasoning about crawling strategy"""
        target_url = context.get('base_url', 'Unknown')
        
        prompt = f"""
        As a Web Crawler Agent, analyze this crawling task:
        
        Target URL: {target_url}
        Max Depth: {context.get('max_depth', 2)}
        Goal: Find PDF files
        
        Available Crawling Engines:
        - BeautifulSoup: Fast for static content, good for simple sites
        - Selenium: Handles JavaScript-heavy sites, real browser automation
        - Playwright: Modern browser automation, handles SPAs and dynamic content
        
        Please provide:
        1. Which crawling engine to use based on the domain
        2. Potential challenges for this domain
        3. Optimal crawling strategy
        4. PDF detection methods to use
        
        Respond in JSON format:
        {{
            "analysis": {{
                "recommended_engine": "beautifulsoup|selenium|playwright",
                "engine_reason": "why this engine is recommended",
                "challenges": [],
                "strategy": "",
                "pdf_detection_methods": [],
                "risks": [],
                "mitigation": []
            }}
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            self.log_action("reasoning", {"context": context}, "completed")
            
            # Parse JSON response, handle potential formatting issues
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return proper JSON
                return {
                    "analysis": {
                        "recommended_engine": "beautifulsoup",
                        "engine_reason": "Fallback: LLM response parsing failed",
                        "challenges": ["Unknown domain structure"],
                        "strategy": "Conservative crawling with fallbacks",
                        "pdf_detection_methods": ["url_extension", "content_analysis"],
                        "risks": ["Server blocking", "Incomplete crawling"],
                        "mitigation": ["Respectful delays", "Multiple engine fallbacks"]
                    }
                }
                
        except Exception as e:
            self.log_action("reasoning", {"context": context}, "failed", str(e))
            return {
                "analysis": {
                    "error": str(e),
                    "recommended_engine": "beautifulsoup",
                    "engine_reason": "Fallback: Reasoning failed",
                    "challenges": ["Reasoning system unavailable"],
                    "strategy": "Default conservative approach",
                    "pdf_detection_methods": ["basic_url_scan"],
                    "risks": ["Limited intelligence"],
                    "mitigation": ["Use multiple engines", "Manual review if needed"]
                }
            }
    
    async def execute(self, state: Dict) -> Dict:
        """Main execution method - overrides parent to maintain compatibility"""
        # Use the autonomous workflow from parent class
        return await super().execute(state)
    
    async def _crawl_recursive(self, url: str, max_depth: int, current_depth: int, engine: str):
        """Recursive crawling implementation with engine selection"""
        if current_depth > max_depth or url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        print(f"🔍 Crawling {url} (depth {current_depth}) with {engine}")
        
        try:
            # Choose crawling engine based on recommendation
            crawler = self.crawling_engines.get(engine, self.crawling_engines['beautifulsoup'])
            
            # Get page content and links
            html_content, links = await crawler.crawl_page(url)
            
            if html_content:
                # Find PDFs on current page
                current_pdfs = self.web_tools.filter_pdf_links(links)
                self.pdf_urls.extend(current_pdfs)
                
                if current_pdfs:
                    print(f"📄 Found {len(current_pdfs)} PDFs at {url}")
            
            # Crawl child pages with concurrency control
            if current_depth < max_depth:
                child_tasks = []
                for link in links:
                    if link not in self.visited_urls and self._is_same_domain(link, url):
                        task = self._crawl_recursive(link, max_depth, current_depth + 1, engine)
                        child_tasks.append(task)
                
                # Process in batches with rate limiting
                for i in range(0, len(child_tasks), 3):
                    await asyncio.gather(*child_tasks[i:i+3])
                    await asyncio.sleep(1)  # Rate limiting
                    
        except Exception as e:
            self.log_action("crawl_page", {"url": url, "depth": current_depth, "engine": engine}, "failed", str(e))
            print(f"❌ Crawling failed for {url}: {e}")
            
            # Fallback to simpler engine if complex one fails
            if engine != 'beautifulsoup':
                print(f"🔄 Falling back to BeautifulSoup for {url}")
                await self._crawl_recursive(url, max_depth, current_depth, 'beautifulsoup')
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain"""
        try:
            domain1 = urlparse(url1).netloc
            domain2 = urlparse(url2).netloc
            return domain1 == domain2
        except:
            return False

# Base Crawler Interface
class BaseCrawler:
    async def crawl_page(self, url: str) -> tuple[str, Set[str]]:
        """Crawl a page and return HTML content and links"""
        raise NotImplementedError

# BeautifulSoup Crawler (Static Content)
class BeautifulSoupCrawler(BaseCrawler):
    def __init__(self):
        self.session = None
    
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def crawl_page(self, url: str) -> tuple[str, Set[str]]:
        """Crawl page using BeautifulSoup (static content)"""
        try:
            session = await self.get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                else:
                    print(f"❌ HTTP {response.status} for {url}")
                    return "", set()
            
            soup = BeautifulSoup(html, 'html.parser')
            links = set()
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                links.add(full_url)
            
            return html, links
            
        except Exception as e:
            print(f"BeautifulSoup crawl failed for {url}: {e}")
            return "", set()

# Selenium Crawler (Dynamic Content)
class SeleniumCrawler(BaseCrawler):
    def __init__(self):
        self.driver = None
    
    async def crawl_page(self, url: str) -> tuple[str, Set[str]]:
        """Crawl page using Selenium (dynamic content)"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            
            # Setup Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.get(url)
            
            # Wait for page to load
            self.driver.implicitly_wait(10)
            
            # Get page source after JavaScript execution
            html = self.driver.page_source
            
            # Extract links
            links = set()
            a_tags = self.driver.find_elements(By.TAG_NAME, "a")
            for a_tag in a_tags:
                href = a_tag.get_attribute("href")
                if href:
                    full_url = urljoin(url, href)
                    links.add(full_url)
            
            return html, links
            
        except Exception as e:
            print(f"Selenium crawl failed for {url}: {e}")
            return "", set()
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

# Playwright Crawler (Modern Dynamic Content)
class PlaywrightCrawler(BaseCrawler):
    def __init__(self):
        self.browser = None
        self.context = None
    
    async def crawl_page(self, url: str) -> tuple[str, Set[str]]:
        """Crawl page using Playwright (modern dynamic content)"""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Launch browser
                self.browser = await p.chromium.launch(headless=True)
                self.context = await self.browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                # Create page
                page = await self.context.new_page()
                
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
                
                # Close page
                await page.close()
                
                return html, links
                
        except Exception as e:
            print(f"Playwright crawl failed for {url}: {e}")
            return "", set()
        finally:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()

# Web Tools (Utilities)
class WebTools:
    def __init__(self):
        self.session = None
    
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def fetch_page(self, url: str) -> str:
        session = await self.get_session()
        async with session.get(url) as response:
            return await response.text()
    
    def extract_internal_links(self, html: str, base_url: str) -> Set[str]:
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == base_domain:
                links.add(full_url)
        
        return links
    
    def filter_pdf_links(self, links: Set[str]) -> List[str]:
        pdf_links = []
        for link in links:
            if link.lower().endswith('.pdf'):
                pdf_links.append(link)
            # Also check for PDF in URL parameters
            elif '.pdf?' in link.lower():
                pdf_links.append(link)
        return pdf_links
    
    def detect_dynamic_content(self, html: str) -> bool:
        """Detect if page likely has dynamic content"""
        indicators = [
            '<script src=',
            'react',
            'angular',
            'vue',
            'webpack',
            'spa',
            'single page application',
            'dynamic content',
            'ajax',
            'fetch('
        ]
        
        html_lower = html.lower()
        return any(indicator in html_lower for indicator in indicators)
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
from tools.web_tools import WebTools
from memory.state_manager import global_state_manager

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
        
        # Memory management
        self.state_manager = global_state_manager
    
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
        initial_engine = await self._suggest_initial_engine(target_url)
        
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
    
    async def _suggest_initial_engine(self, url: str) -> str:
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
        
        # Check if the site has dynamic content
        try:
            has_dynamic = await self.web_tools.detect_dynamic_content(url)
            if has_dynamic:
                return "playwright"  # Use more modern engine for dynamic content
        except:
            pass
        
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
        if current_depth > max_depth or url in self.visited_urls or self.web_tools.is_url_visited(url):
            return
        
        self.visited_urls.add(url)
        self.web_tools.mark_url_visited(url)
        print(f"🔍 Crawling {url} (depth {current_depth}) with {engine}")
        
        try:
            # Use the smart crawling engine that automatically selects the best method
            html_content, links = await self.web_tools.smart_crawl(url, engine)
            
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
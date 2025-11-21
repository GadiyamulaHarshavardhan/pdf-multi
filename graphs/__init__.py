"""
Graphs module for PDF Agent System.

This module contains workflow definitions for orchestrating
the multi-agent PDF processing system.
"""

import sys

try:
    from .workflow_graph import PDFWorkflowGraph
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LangGraph import warning: {e}")
    print("🔄 Falling back to simple workflow...")
    
    # Create a fallback class
    class PDFWorkflowGraph:
        def __init__(self):
            from config import get_ollama_llm
            from agents.crawler_agent import CrawlerAgent
            from agents.downloader_agent import DownloaderAgent
            from agents.classifier_agent import ClassifierAgent
            import asyncio
            
            self.llm = get_ollama_llm()
            self.crawler = CrawlerAgent(self.llm)
            self.downloader = DownloaderAgent(self.llm)
            self.classifier = ClassifierAgent(self.llm)
        
        async def execute_workflow(self, urls):
            """Simple sequential workflow"""
            results = {}
            for url in urls:
                print(f"🔍 Processing: {url}")
                
                crawl_result = await self.crawler.execute({"base_url": url, "max_depth": 2})
                results[url] = {"crawling": crawl_result}
                
                if crawl_result.get("pdf_urls_found"):
                    download_result = await self.downloader.execute({
                        "pdf_urls": crawl_result["pdf_urls_found"]
                    })
                    results[url]["downloading"] = download_result
                    
                    if download_result.get("downloaded_files"):
                        classify_result = await self.classifier.execute({
                            "downloaded_files": download_result["downloaded_files"]
                        })
                        results[url]["classification"] = classify_result
                
            return results

__all__ = ["PDFWorkflowGraph"]
__version__ = "1.0.0"
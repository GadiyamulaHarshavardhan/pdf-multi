from .autonomous_agent import AutonomousAgent
from tools.web_tools import WebTools
import os
from typing import Dict, List
import asyncio

class DownloaderAgent(AutonomousAgent):
    def __init__(self):
        super().__init__(
            name="downloader_agent",
            role="File Downloader", 
            goal="Download PDFs reliably from identified URLs"
        )
        self.web_tools = WebTools()
        self.download_dir = "data/pdfs"
        os.makedirs(self.download_dir, exist_ok=True)
    
    async def _analyze_problem(self, context: Dict) -> Dict:
        """Analyze download task"""
        pdf_urls = context.get('pdf_urls', [])
        
        return {
            "task": "download_pdf_files",
            "total_files": len(pdf_urls),
            "total_size_estimate": len(pdf_urls) * 2,  # MB estimate
            "network_requirements": "stable_connection",
            "storage_requirements": f"{len(pdf_urls) * 2}MB minimum"
        }
    
    async def _create_execution_plan(self, analysis: Dict) -> Dict:
        """Create download execution plan"""
        return {
            "strategy": "batched_concurrent_download",
            "batch_size": 3,
            "retry_strategy": "exponential_backoff",
            "verification_method": "file_size_check",
            "min_file_size_kb": 1  # Avoid empty files
        }
    
    async def _self_check_plan(self, plan: Dict) -> Dict:
        """Validate download plan"""
        issues = []
        
        if plan["batch_size"] > 5:
            issues.append("High batch size may overwhelm server")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Monitor server responses", "Implement respectful delays"]
        }
    
    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute download plan"""
        pdf_urls = plan.get('pdf_urls', [])
        batch_size = plan.get('batch_size', 3)
        
        downloaded_files = []
        failed_downloads = []
        
        for i in range(0, len(pdf_urls), batch_size):
            batch = pdf_urls[i:i + batch_size]
            batch_tasks = []
            
            for url in batch:
                filename = self._generate_filename(url)
                filepath = os.path.join(self.download_dir, filename)
                task = self._download_single_pdf(url, filepath)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                url = batch[j]
                if isinstance(result, Exception) or not result:
                    failed_downloads.append({"url": url, "error": str(result)})
                    self.log_action("download_file", {"url": url}, "failed", str(result))
                else:
                    downloaded_files.append({
                        "url": url,
                        "filepath": result["filepath"],
                        "size_kb": result["size_kb"]
                    })
                    self.log_action("download_file", {"url": url, "size_kb": result["size_kb"]}, "completed")
            
            # Rate limiting between batches
            if i + batch_size < len(pdf_urls):
                await asyncio.sleep(2)
        
        return {
            "downloaded_files": downloaded_files,
            "failed_downloads": failed_downloads,
            "success_rate": len(downloaded_files) / len(pdf_urls) if pdf_urls else 0
        }
    
    async def _download_single_pdf(self, url: str, filepath: str) -> Dict:
        """Download single PDF with verification"""
        success = await self.web_tools.download_pdf(url, filepath)
        
        if success and os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            return {
                "filepath": filepath,
                "size_kb": round(size_kb, 2),
                "url": url
            }
        return False
    
    def _generate_filename(self, url: str) -> str:
        """Generate safe filename from URL"""
        import re
        from urllib.parse import urlparse
        
        path = urlparse(url).path
        filename = os.path.basename(path) or "document.pdf"
        
        # Clean filename
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        return filename
    
    async def _verify_result(self, result: Dict) -> Dict:
        """Verify download results"""
        issues = []
        
        if result["success_rate"] < 0.5:
            issues.append(f"Low success rate: {result['success_rate']:.1%}")
        
        if len(result["failed_downloads"]) > len(result["downloaded_files"]):
            issues.append("More failures than successes")
        
        # Verify file sizes
        for file_info in result["downloaded_files"]:
            if file_info["size_kb"] < 1:
                issues.append(f"Small file detected: {file_info['filepath']}")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Check network connectivity", "Verify URLs are accessible", "Retry failed downloads"]
        }
    
    async def _debug_and_adapt(self, plan: Dict, error: Exception):
        """Adapt download strategy based on errors"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            # Increase timeouts for slow servers
            self.log_action("adapt_strategy", {"action": "slower_downloads"}, "timeout_detected")
        
        if "404" in error_str or "not found" in error_str:
            # Skip invalid URLs in future batches
            plan["skip_invalid_urls"] = True
            self.log_action("adapt_strategy", {"action": "skip_invalid_urls"}, "404_detected")
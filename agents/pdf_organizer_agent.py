from .autonomous_agent import AutonomousAgent
from typing import Dict, Any, List
import os
import asyncio
from datetime import datetime
import json
from tools.web_tools import WebTools
from memory.state_manager import global_state_manager

class PDFOrganizerAgent(AutonomousAgent):
    def __init__(self, llm):
        super().__init__(
            name="pdf_organizer_agent",
            role="PDF Organizer",
            goal="Download, organize, and manage PDF files found during crawling"
        )
        self.llm = llm
        self.web_tools = WebTools()
        self.state_manager = global_state_manager
        self.download_dir = "/workspace/downloads"
        
        # Create download directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)
    
    async def _analyze_problem(self, context: Dict) -> Dict:
        """Analyze the PDF organization task"""
        pdf_urls = context.get('pdf_urls', [])
        base_url = context.get('base_url', 'unknown')
        
        return {
            "task": "organize_pdfs",
            "pdf_count": len(pdf_urls),
            "base_url": base_url,
            "download_strategy": "batch_download_with_organization",
            "organization_method": "by_domain_and_category"
        }
    
    async def _create_execution_plan(self, analysis: Dict) -> Dict:
        """Create execution plan for PDF organization"""
        return {
            "strategy": "batch_download",
            "max_concurrent_downloads": 5,
            "organization_scheme": "domain_category_date",
            "naming_scheme": "original_filename_with_prefix",
            "backup_strategy": "keep_original_url_in_metadata",
            "verification_method": "checksum_and_size_check"
        }
    
    async def _self_check_plan(self, plan: Dict) -> Dict:
        """Validate the organization plan"""
        issues = []
        
        if plan["max_concurrent_downloads"] > 10:
            issues.append("Too many concurrent downloads may overwhelm server")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Monitor download speeds", "Respect server rate limits"]
        }
    
    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute the PDF organization plan"""
        # Get PDF URLs from state
        pdf_urls = self.state_manager.retrieve_state("found_pdfs") or []
        
        if not pdf_urls:
            return {
                "downloaded_count": 0,
                "organized_count": 0,
                "failed_count": 0,
                "message": "No PDF URLs found to organize"
            }
        
        # Download PDFs
        download_results = await self._download_pdfs(pdf_urls, self.download_dir)
        
        # Organize downloaded files
        organized_files = await self._organize_files(download_results)
        
        # Update state with results
        self.state_manager.store_state("downloaded_pdfs", download_results)
        self.state_manager.store_state("organized_pdfs", organized_files)
        
        return {
            "downloaded_count": len([r for r in download_results if r['success']]),
            "organized_count": len(organized_files),
            "failed_count": len([r for r in download_results if not r['success']]),
            "download_results": download_results,
            "organized_files": organized_files,
            "execution_summary": f"Downloaded {len([r for r in download_results if r['success']])} PDFs, organized {len(organized_files)} files"
        }
    
    async def _verify_result(self, result: Dict) -> Dict:
        """Verify PDF organization results"""
        issues = []
        
        if result["downloaded_count"] == 0 and result["failed_count"] > 0:
            issues.append("All PDF downloads failed")
        
        success_rate = result["downloaded_count"] / (result["downloaded_count"] + result["failed_count"]) if (result["downloaded_count"] + result["failed_count"]) > 0 else 0
        if success_rate < 0.5:  # Less than 50% success rate
            issues.append(f"Low download success rate: {success_rate:.2%}")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "success_rate": success_rate,
            "suggestions": ["Check server connectivity", "Retry failed downloads", "Verify file permissions"]
        }
    
    async def _debug_and_adapt(self, plan: Dict, error: Exception):
        """Debug and adapt organization strategy based on errors"""
        error_str = str(error).lower()
        
        if "permission" in error_str:
            plan["organization_scheme"] = "use_temp_directory"
            self.log_action("adapt_strategy", {"action": "change_to_temp_dir"}, "permission_error")
        
        if "disk space" in error_str or "quota" in error_str:
            plan["max_concurrent_downloads"] = max(1, plan["max_concurrent_downloads"] // 2)
            self.log_action("adapt_strategy", {"action": "reduce_concurrent_downloads"}, "disk_space_error")
    
    async def _download_pdfs(self, pdf_urls: List[str], base_dir: str) -> List[Dict[str, Any]]:
        """Download PDF files with batch processing"""
        print(f"📥 Starting download of {len(pdf_urls)} PDFs...")
        
        # Use the enhanced batch download method from web_tools
        results = await self.web_tools.batch_download_pdfs(pdf_urls, base_dir)
        
        print(f"✅ Downloaded {len([r for r in results if r['success']])} PDFs, {len([r for r in results if not r['success']])} failed")
        return results
    
    async def _organize_files(self, download_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize downloaded files by domain, category, and date"""
        organized_files = []
        
        for result in download_results:
            if result['success']:
                original_filename = os.path.basename(result['filepath'])
                file_info = {
                    'original_url': result['url'],
                    'original_filename': original_filename,
                    'downloaded_path': result['filepath'],
                    'organized_path': '',
                    'size': os.path.getsize(result['filepath']),
                    'download_date': datetime.now().isoformat()
                }
                
                # Create organized path based on domain and date
                domain = self._extract_domain(result['url'])
                date_folder = datetime.now().strftime("%Y-%m-%d")
                
                organized_dir = os.path.join(self.download_dir, domain, date_folder)
                os.makedirs(organized_dir, exist_ok=True)
                
                # Move file to organized location
                organized_path = os.path.join(organized_dir, original_filename)
                
                # If file is already in the right place, just update the path
                if result['filepath'] != organized_path:
                    os.rename(result['filepath'], organized_path)
                    file_info['organized_path'] = organized_path
                else:
                    file_info['organized_path'] = result['filepath']
                
                organized_files.append(file_info)
        
        return organized_files
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '').replace('.', '_')
        return domain if domain else 'unknown'
    
    async def execute(self, state: Dict) -> Dict:
        """Main execution method"""
        # Use the autonomous workflow from parent class
        return await super().execute(state)
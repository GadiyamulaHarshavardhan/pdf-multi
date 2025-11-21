"""
Main orchestrator for PDF Agent System
Processes URLs from url.txt through the multi-agent workflow
"""

import asyncio
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PDFSystem")

class Config:
    """Configuration settings for the PDF Agent System"""
    DATA_DIR = "data"
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")
    CATEGORIZED_DIR = os.path.join(DATA_DIR, "categorized")
    LOGS_DIR = "logs"
    OUTPUT_DIR = "output"
    
    # Processing settings
    MAX_CONCURRENT_URLS = 1  # Process one URL at a time
    REQUEST_DELAY = 2  # Delay between URL processing (seconds)
    MAX_RETRIES = 3  # Maximum retries per URL
    
    # Agent settings
    CRAWL_DEPTH = 2
    TIMEOUT = 30
    
    # LLM Settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2048

class PDFAgentOrchestrator:
    """
    Main orchestrator that processes URLs through the multi-agent workflow.
    Supports both LangGraph and sequential execution modes.
    """
    
    def __init__(self, use_langgraph: bool = True):
        self.config = Config()
        self.use_langgraph = use_langgraph
        self.setup_directories()
        self.workflow_stats = {
            "total_urls": 0,
            "processed_urls": 0,
            "successful_urls": 0,
            "failed_urls": 0,
            "total_pdfs_found": 0,
            "total_pdfs_downloaded": 0,
            "total_pdfs_classified": 0,
            "start_time": None,
            "end_time": None,
            "execution_mode": "langgraph" if use_langgraph else "sequential"
        }
        
        # Initialize LLM first
        self.llm = self._initialize_llm()
        
        # Initialize workflow based on availability
        if use_langgraph:
            try:
                from graphs.workflow_graph import PDFWorkflowGraph
                print("🔄 Initializing LangGraph workflow...")
                self.workflow = PDFWorkflowGraph()
                if self.workflow.workflow is not None:
                    print("✅ LangGraph workflow initialized successfully")
                else:
                    print("❌ LangGraph workflow creation failed")
                    self.use_langgraph = False
                    self.workflow = None
            except Exception as e:
                print(f"❌ LangGraph initialization failed: {e}")
                self.use_langgraph = False
                self.workflow = None
        else:
            self.workflow = None
            print("ℹ️  Using sequential execution mode")
        
    def _initialize_llm(self):
        """Initialize the Ollama LLM with proper imports"""
        try:
            # Try the new langchain-ollama import first
            from langchain_ollama import OllamaLLM
            
            llm = OllamaLLM(
                base_url=self.config.OLLAMA_BASE_URL,
                model=self.config.OLLAMA_MODEL,
                temperature=self.config.TEMPERATURE,
                num_predict=self.config.MAX_TOKENS
            )
            logger.info("Initialized Ollama LLM using langchain-ollama")
            return llm
            
        except ImportError:
            try:
                # Fallback to community version
                from langchain_community.llms import Ollama
                
                llm = Ollama(
                    base_url=self.config.OLLAMA_BASE_URL,
                    model=self.config.OLLAMA_MODEL,
                    temperature=self.config.TEMPERATURE,
                    num_predict=self.config.MAX_TOKENS
                )
                logger.info("Initialized Ollama LLM using langchain-community")
                return llm
                
            except ImportError as e:
                logger.error(f"Failed to initialize LLM: {e}")
                raise ImportError("Could not initialize Ollama LLM. Please install langchain-ollama or langchain-community")
    
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [
            self.config.DATA_DIR,
            self.config.PDF_DIR,
            self.config.CATEGORIZED_DIR,
            self.config.LOGS_DIR,
            self.config.OUTPUT_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def read_urls_from_file(self, input_file: str = "url.txt") -> List[str]:
        """
        Read URLs from input file with validation and cleaning
        """
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            self._create_sample_url_file(input_file)
            return []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                urls = []
                for line_num, line in enumerate(f, 1):
                    url = line.strip()
                    if url and not url.startswith('#'):  # Skip empty lines and comments
                        if self._validate_url(url):
                            urls.append(url)
                        else:
                            logger.warning(f"Invalid URL on line {line_num}: {url}")
                
            logger.info(f"Read {len(urls)} valid URLs from {input_file}")
            return urls
            
        except Exception as e:
            logger.error(f"Error reading URLs from {input_file}: {str(e)}")
            return []
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        import re
        url_pattern = re.compile(
            r'^(https?|ftp)://'  # http://, https://, or ftp://
            r'([A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+'  # domain
            r'[A-Za-z]{2,}'  # TLD
            r'(:\d+)?'  # port
            r'(/.*)?$'  # path
        )
        return bool(url_pattern.match(url))
    
    def _create_sample_url_file(self, input_file: str):
        """Create a sample URL file if it doesn't exist"""
        sample_urls = [
            "https://arxiv.org",
            "https://www.researchgate.net",
            "https://www.overleaf.com",
            "# Add your URLs here, one per line",
            "# Lines starting with # are comments"
        ]
        
        try:
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sample_urls))
            logger.info(f"Created sample URL file: {input_file}")
            print(f"📝 Created sample URL file: {input_file}")
            print("   Please add your target URLs to this file and run again.")
        except Exception as e:
            logger.error(f"Failed to create sample URL file: {str(e)}")
    
    def _initialize_agent(self, agent_name: str):
        """Initialize an agent with comprehensive error handling"""
        try:
            if agent_name == "crawler":
                # Try sync version first
                try:
                    from agents.sync_crawler_agent import SyncCrawlerAgent
                    agent = SyncCrawlerAgent(self.llm)
                    print(f"✅ Using SyncCrawlerAgent")
                    return agent
                except ImportError as e:
                    print(f"❌ SyncCrawlerAgent import failed: {e}")
                    # Fallback to simple crawler
                    try:
                        from agents.simple_crawler_agent import SimpleCrawlerAgent
                        agent = SimpleCrawlerAgent(self.llm)
                        print(f"✅ Using SimpleCrawlerAgent")
                        return agent
                    except ImportError:
                        # Final fallback to original crawler
                        from agents.crawler_agent import CrawlerAgent
                        try:
                            agent = CrawlerAgent(self.llm)
                            print(f"⚠️  Using CrawlerAgent with LLM")
                            return agent
                        except TypeError:
                            agent = CrawlerAgent()
                            print(f"⚠️  Using CrawlerAgent without LLM")
                            return agent
                    
            elif agent_name == "downloader":
                # Try sync version first
                try:
                    from agents.sync_downloader_agent import SyncDownloaderAgent
                    # Try with LLM, fallback to without LLM
                    try:
                        agent = SyncDownloaderAgent(self.llm)
                        print(f"✅ Using SyncDownloaderAgent with LLM")
                    except (TypeError, ValueError):
                        agent = SyncDownloaderAgent()
                        print(f"✅ Using SyncDownloaderAgent without LLM")
                    return agent
                except ImportError as e:
                    print(f"❌ SyncDownloaderAgent import failed: {e}")
                    # Fallback to original downloader
                    from agents.downloader_agent import DownloaderAgent
                    try:
                        agent = DownloaderAgent()
                        print(f"⚠️  Using DownloaderAgent without LLM")
                        return agent
                    except TypeError:
                        agent = DownloaderAgent(self.llm)
                        print(f"⚠️  Using DownloaderAgent with LLM")
                        return agent
                    
            elif agent_name == "classifier":
                # Try sync version first
                try:
                    from agents.sync_classifier_agent import SyncClassifierAgent
                    # Try with LLM, fallback to without LLM
                    try:
                        agent = SyncClassifierAgent(self.llm)
                        print(f"✅ Using SyncClassifierAgent with LLM")
                    except (TypeError, ValueError):
                        agent = SyncClassifierAgent()
                        print(f"✅ Using SyncClassifierAgent without LLM")
                    return agent
                except ImportError as e:
                    print(f"❌ SyncClassifierAgent import failed: {e}")
                    # Fallback to original classifier
                    from agents.classifier_agent import ClassifierAgent
                    try:
                        agent = ClassifierAgent()
                        print(f"⚠️  Using ClassifierAgent without LLM")
                        return agent
                    except TypeError:
                        agent = ClassifierAgent(self.llm)
                        print(f"⚠️  Using ClassifierAgent with LLM")
                        return agent
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {agent_name} agent: {str(e)}")
            raise
    
    async def process_single_url_langgraph(self, url: str) -> Dict[str, Any]:
        """Process URL using LangGraph workflow"""
        try:
            if not self.workflow:
                raise ValueError("LangGraph workflow not available")
            
            print(f"🔄 Processing with LangGraph: {url}")
            result = self.workflow.execute_workflow([url])
            
            # Convert LangGraph result to our standard format
            return {
                "status": "success",
                "crawling": result.get("crawler_result", {}),
                "downloading": result.get("downloader_result", {}),
                "classification": result.get("classifier_result", {}),
                "reasoning": result.get("reasoning", {}),
                "workflow_type": "langgraph"
            }
            
        except Exception as e:
            logger.error(f"LangGraph processing failed for {url}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_type": "langgraph"
            }
    
    async def process_single_url_sequential(self, url: str) -> Dict[str, Any]:
        """Process URL using sequential workflow with smart engine selection"""
        print(f"🔄 Processing sequentially: {url}")
        result = {
            "url": url,
            "processing_start": datetime.now().isoformat(),
            "processing_end": None,
            "processing_duration": None,
            "status": "unknown",
            "error": None,
            "crawling": {},
            "downloading": {},
            "classification": {},
            "workflow_type": "sequential"
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Crawling with smart engine detection
            print("🕷️  Step 1/3: Crawling for PDFs...")
            crawl_result = await self._crawl_url_smart(url)
            result["crawling"] = crawl_result
            
            if not crawl_result.get("success", False):
                result["status"] = "crawling_failed"
                result["error"] = crawl_result.get("error", "Crawling failed")
                return result
            
            pdf_urls = crawl_result.get("pdf_urls_found", [])
            engine_used = crawl_result.get("crawling_engine_used", "beautifulsoup")
            print(f"   Found {len(pdf_urls)} PDFs using {engine_used}")
            
            if not pdf_urls:
                result["status"] = "no_pdfs_found"
                return result
            
            # Step 2: Downloading
            print(f"📥 Step 2/3: Downloading {len(pdf_urls)} PDFs...")
            download_result = await self._download_pdfs(pdf_urls)
            result["downloading"] = download_result
            
            if not download_result.get("success", False):
                result["status"] = "download_failed"
                result["error"] = download_result.get("error", "Download failed")
                return result
            
            downloaded_files = download_result.get("downloaded_files", [])
            print(f"   Downloaded {len(downloaded_files)} PDFs")
            
            if not downloaded_files:
                result["status"] = "no_files_downloaded"
                return result
            
            # Step 3: Classification
            print(f"🏷️  Step 3/3: Classifying {len(downloaded_files)} PDFs...")
            classification_result = await self._classify_pdfs(downloaded_files)
            result["classification"] = classification_result
            
            if not classification_result.get("success", False):
                result["status"] = "classification_failed"
                result["error"] = classification_result.get("error", "Classification failed")
                return result
            
            result["status"] = "success"
            print("✅ Sequential processing completed successfully")
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"❌ Sequential processing failed: {e}")
        
        finally:
            end_time = time.time()
            result["processing_end"] = datetime.now().isoformat()
            result["processing_duration"] = round(end_time - start_time, 2)
            
        return result

    async def _crawl_url_smart(self, url: str) -> Dict[str, Any]:
        """Smart crawling that detects and uses appropriate engine"""
        try:
            crawler = self._initialize_agent("crawler")
            
            # First, try to detect if the site needs dynamic crawling
            needs_dynamic = await self._detect_dynamic_content(url)
            engine_hint = "selenium" if needs_dynamic else "beautifulsoup"
            
            crawl_result = await crawler.execute({
                "base_url": url,
                "max_depth": self.config.CRAWL_DEPTH,
                "engine_hint": engine_hint
            })
            
            return {
                "success": True,
                **crawl_result
            }
            
        except Exception as e:
            logger.error(f"Crawling error for {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pdf_urls_found": [],
                "pdf_count": 0
            }
    
    async def _detect_dynamic_content(self, url: str) -> bool:
        """Detect if URL likely needs dynamic content crawling"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Check for dynamic content indicators
                        dynamic_indicators = [
                            'react', 'angular', 'vue', 'webpack', 'spa',
                            'single page application', 'ajax', 'fetch(',
                            'nextjs', 'nuxtjs', 'gatsby'
                        ]
                        
                        html_lower = html.lower()
                        return any(indicator in html_lower for indicator in dynamic_indicators)
                    else:
                        return False  # If can't fetch, assume static
                    
        except Exception as e:
            print(f"⚠️  Dynamic content detection failed for {url}: {e}")
            return False  # Default to static on error
    
    async def process_single_url(self, url: str, url_index: int, total_urls: int) -> Dict[str, Any]:
        """
        Process a single URL through the complete workflow
        """
        logger.info(f"Processing URL {url_index}/{total_urls}: {url}")
        print(f"\n{'='*60}")
        print(f"🔍 Processing URL {url_index}/{total_urls}: {url}")
        print(f"⚡ Mode: {'LangGraph' if self.use_langgraph else 'Sequential'}")
        print(f"{'='*60}")
        
        # Choose processing method
        if self.use_langgraph and self.workflow:
            result = await self.process_single_url_langgraph(url)
        else:
            result = await self.process_single_url_sequential(url)
        
        result["url"] = url
        
        # Update statistics
        self._update_statistics(result)
        
        # Log and save results
        self._log_processing_result(url_index, total_urls, result)
        await self._save_url_results(url, result)
        
        return result
    
    def _update_statistics(self, result: Dict[str, Any]):
        """Update workflow statistics based on processing result"""
        self.workflow_stats["processed_urls"] += 1
        
        if result["status"] == "success":
            self.workflow_stats["successful_urls"] += 1
            
            # Count PDFs from different workflow types
            if result["workflow_type"] == "langgraph":
                crawling = result.get("crawling", {})
                downloading = result.get("downloading", {})
                classification = result.get("classification", {})
            else:
                crawling = result.get("crawling", {})
                downloading = result.get("downloading", {})
                classification = result.get("classification", {})
            
            self.workflow_stats["total_pdfs_found"] += len(crawling.get("pdf_urls_found", []))
            self.workflow_stats["total_pdfs_downloaded"] += len(downloading.get("downloaded_files", []))
            self.workflow_stats["total_pdfs_classified"] += len(classification.get("classifications", []))
        else:
            self.workflow_stats["failed_urls"] += 1
    
    def _log_processing_result(self, url_index: int, total_urls: int, result: Dict[str, Any]):
        """Log processing result with appropriate emoji"""
        status = result["status"]
        
        if status == "success":
            print(f"✅ SUCCESS")
            pdfs_found = len(result.get("crawling", {}).get("pdf_urls_found", []))
            pdfs_downloaded = len(result.get("downloading", {}).get("downloaded_files", []))
            pdfs_classified = len(result.get("classification", {}).get("classifications", []))
            print(f"   📊 Summary: {pdfs_found} found, {pdfs_downloaded} downloaded, {pdfs_classified} classified")
        elif status == "no_pdfs_found":
            print(f"ℹ️  NO PDFS FOUND")
        elif status == "crawling_failed":
            print(f"❌ CRAWLING FAILED")
        elif status == "download_failed":
            print(f"❌ DOWNLOAD FAILED")
        elif status == "classification_failed":
            print(f"❌ CLASSIFICATION FAILED")
        else:
            print(f"💥 ERROR: {status}")
        
        if result.get("error"):
            print(f"   Error: {result['error']}")
    
    async def _crawl_url(self, url: str) -> Dict[str, Any]:
        """Crawl a single URL to find PDF links"""
        try:
            crawler = self._initialize_agent("crawler")
            crawl_result = await crawler.execute({
                "base_url": url,
                "max_depth": self.config.CRAWL_DEPTH
            })
            
            return {
                "success": True,
                **crawl_result
            }
            
        except Exception as e:
            logger.error(f"Crawling error for {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pdf_urls_found": [],
                "pdf_count": 0
            }
    
    async def _download_pdfs(self, pdf_urls: List[str]) -> Dict[str, Any]:
        """Download PDF files from URLs"""
        try:
            downloader = self._initialize_agent("downloader")
            download_result = await downloader.execute({
                "pdf_urls": pdf_urls
            })
            
            return {
                "success": True,
                **download_result
            }
            
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "downloaded_files": [],
                "failed_downloads": []
            }
    
    async def _classify_pdfs(self, downloaded_files: List[Dict]) -> Dict[str, Any]:
        """Classify downloaded PDF files"""
        try:
            classifier = self._initialize_agent("classifier")
            classification_result = await classifier.execute({
                "downloaded_files": downloaded_files
            })
            
            return {
                "success": True,
                **classification_result
            }
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "classifications": [],
                "organized_results": {}
            }
    
    async def _save_url_results(self, url: str, result: Dict[str, Any]):
        """Save results for a single URL"""
        try:
            domain = urlparse(url).netloc.replace('.', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}_{timestamp}.json"
            filepath = os.path.join(self.config.OUTPUT_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved results for {url} to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save results for {url}: {str(e)}")
    
    async def process_all_urls(self, urls: List[str]):
        """
        Process all URLs one by one with progress tracking
        """
        if not urls:
            print("❌ No URLs to process. Please check url.txt file.")
            return
        
        self.workflow_stats["total_urls"] = len(urls)
        self.workflow_stats["start_time"] = datetime.now().isoformat()
        
        print(f"\n🚀 Starting PDF Agent System")
        print(f"📋 Total URLs to process: {len(urls)}")
        print(f"⚡ Execution mode: {'LangGraph' if self.use_langgraph else 'Sequential'}")
        print(f"🔧 Processing one URL at a time")
        print(f"⏰ Request delay: {self.config.REQUEST_DELAY} seconds")
        print(f"{'='*60}")
        
        all_results = []
        
        for index, url in enumerate(urls, 1):
            # Process single URL
            result = await self.process_single_url(url, index, len(urls))
            all_results.append(result)
            
            # Add delay between URLs (except for the last one)
            if index < len(urls):
                print(f"\n⏳ Waiting {self.config.REQUEST_DELAY} seconds before next URL...")
                await asyncio.sleep(self.config.REQUEST_DELAY)
        
        # Final statistics and reporting
        self.workflow_stats["end_time"] = datetime.now().isoformat()
        await self._generate_final_report(all_results)
    
    async def _generate_final_report(self, all_results: List[Dict[str, Any]]):
        """Generate final workflow report"""
        try:
            report = {
                "workflow_statistics": self.workflow_stats,
                "url_results": all_results,
                "summary": self._create_summary(all_results),
                "generated_at": datetime.now().isoformat()
            }
            
            # Save detailed report
            report_file = os.path.join(self.config.OUTPUT_DIR, f"workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Print summary to console
            self._print_console_summary(report["summary"])
            
            logger.info(f"Final report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {str(e)}")
    
    def _create_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from all results"""
        status_counts = {}
        total_duration = 0
        
        for result in all_results:
            status = result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            total_duration += result.get("processing_duration", 0)
        
        avg_duration = total_duration / len(all_results) if all_results else 0
        
        return {
            "total_urls_processed": len(all_results),
            "status_distribution": status_counts,
            "success_rate": self.workflow_stats["successful_urls"] / len(all_results) if all_results else 0,
            "average_processing_time_seconds": round(avg_duration, 2),
            "total_processing_time_seconds": round(total_duration, 2),
            "pdf_statistics": {
                "total_found": self.workflow_stats["total_pdfs_found"],
                "total_downloaded": self.workflow_stats["total_pdfs_downloaded"],
                "total_classified": self.workflow_stats["total_pdfs_classified"],
                "download_success_rate": self.workflow_stats["total_pdfs_downloaded"] / self.workflow_stats["total_pdfs_found"] if self.workflow_stats["total_pdfs_found"] > 0 else 0
            }
        }
    
    def _print_console_summary(self, summary: Dict[str, Any]):
        """Print beautiful summary to console"""
        print(f"\n{'='*80}")
        print(f"🎉 PDF AGENT SYSTEM - WORKFLOW COMPLETED")
        print(f"{'='*80}")
        
        # Basic statistics
        print(f"📊 PROCESSING SUMMARY:")
        print(f"   • Total URLs processed: {summary['total_urls_processed']}")
        print(f"   • Successful URLs: {self.workflow_stats['successful_urls']}")
        print(f"   • Failed URLs: {self.workflow_stats['failed_urls']}")
        print(f"   • Success rate: {summary['success_rate']:.1%}")
        print(f"   • Execution mode: {self.workflow_stats['execution_mode']}")
        
        # PDF statistics
        pdf_stats = summary['pdf_statistics']
        print(f"\n📄 PDF STATISTICS:")
        print(f"   • PDFs found: {pdf_stats['total_found']}")
        print(f"   • PDFs downloaded: {pdf_stats['total_downloaded']}")
        print(f"   • PDFs classified: {pdf_stats['total_classified']}")
        if pdf_stats['download_success_rate'] > 0:
            print(f"   • Download success rate: {pdf_stats['download_success_rate']:.1%}")
        
        # Timing information
        print(f"\n⏱️  TIMING INFORMATION:")
        print(f"   • Average processing time: {summary['average_processing_time_seconds']} seconds")
        print(f"   • Total processing time: {summary['total_processing_time_seconds']} seconds")
        
        # Status distribution
        print(f"\n📈 STATUS DISTRIBUTION:")
        for status, count in summary['status_distribution'].items():
            percentage = (count / summary['total_urls_processed']) * 100
            status_display = status.replace('_', ' ').title()
            print(f"   • {status_display}: {count} ({percentage:.1f}%)")
        
        print(f"\n💾 Output files saved to: {self.config.OUTPUT_DIR}/")
        print(f"📁 Downloaded PDFs in: {self.config.PDF_DIR}/")
        print(f"🏷️  Categorized results in: {self.config.CATEGORIZED_DIR}/")
        print(f"{'='*80}")

async def main():
    """
    Main entry point for the PDF Agent System
    """
    print("🤖 PDF Agent System - Multi-Agent PDF Processor")
    print("==============================================")
    
    # Ask user for execution mode
    use_langgraph_input = input("Use LangGraph workflow? (y/n, default=y): ").strip().lower()
    use_langgraph = use_langgraph_input != 'n'
    
    # Initialize orchestrator
    orchestrator = PDFAgentOrchestrator(use_langgraph=use_langgraph)
    
    # Read URLs from file
    input_file = "url.txt"
    urls = orchestrator.read_urls_from_file(input_file)
    
    if not urls:
        print(f"❌ No valid URLs found in {input_file}.")
        print("   Please check the file format and try again.")
        return
    
    # Process all URLs
    try:
        await orchestrator.process_all_urls(urls)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user. Shutting down gracefully...")
        logger.info("Process interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        logger.error(f"Unexpected error in main: {str(e)}")
    finally:
        print("\n✨ PDF Agent System finished.")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
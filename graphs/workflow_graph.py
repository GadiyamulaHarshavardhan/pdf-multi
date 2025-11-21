try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for different versions
    try:
        from langgraph import graph
        StateGraph = graph.StateGraph
        END = graph.END
        LANGGRAPH_AVAILABLE = True
    except ImportError:
        LANGGRAPH_AVAILABLE = False

from typing import Dict, Any, List
import json
from datetime import datetime
from config import get_ollama_llm, Config

class PDFWorkflowGraph:
    def __init__(self):
        self.llm = get_ollama_llm()
        
        # Initialize SYNC agents to avoid async issues
        print("🔄 Initializing sync agents for LangGraph compatibility...")
        
        # Initialize agents with better error handling
        self.crawler = self._initialize_agent("crawler")
        self.downloader = self._initialize_agent("downloader") 
        self.classifier = self._initialize_agent("classifier")
            
        if LANGGRAPH_AVAILABLE:
            try:
                self.workflow = self._build_graph()
                print("✅ LangGraph workflow built successfully")
            except Exception as e:
                print(f"❌ LangGraph build failed: {e}")
                self.workflow = None
        else:
            print("❌ LangGraph not available")
            self.workflow = None

    def _initialize_agent(self, agent_name: str):
        """Initialize agent with proper error handling for constructor signatures"""
        try:
            if agent_name == "crawler":
                from agents.sync_crawler_agent import SyncCrawlerAgent
                agent = SyncCrawlerAgent(self.llm)
                print("✅ SyncCrawlerAgent initialized")
                return agent
                
            elif agent_name == "downloader":
                from agents.sync_downloader_agent import SyncDownloaderAgent
                # Try with LLM, fallback to without LLM
                try:
                    agent = SyncDownloaderAgent(self.llm)
                    print("✅ SyncDownloaderAgent initialized with LLM")
                except (TypeError, ValueError) as e:
                    print(f"⚠️  DownloaderAgent doesn't need LLM: {e}")
                    agent = SyncDownloaderAgent()
                    print("✅ SyncDownloaderAgent initialized without LLM")
                return agent
                
            elif agent_name == "classifier":
                from agents.sync_classifier_agent import SyncClassifierAgent
                # Try with LLM, fallback to without LLM
                try:
                    agent = SyncClassifierAgent(self.llm)
                    print("✅ SyncClassifierAgent initialized with LLM")
                except (TypeError, ValueError) as e:
                    print(f"⚠️  ClassifierAgent doesn't need LLM: {e}")
                    agent = SyncClassifierAgent()
                    print("✅ SyncClassifierAgent initialized without LLM")
                return agent
                
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
                
        except Exception as e:
            print(f"❌ Failed to initialize {agent_name}: {e}")
            # Fallback to async agent
            return self._initialize_async_agent(agent_name)

    def _initialize_async_agent(self, agent_name: str):
        """Fallback to async agent initialization"""
        print(f"🔄 Falling back to async {agent_name} agent")
        try:
            if agent_name == "crawler":
                from agents.crawler_agent import CrawlerAgent
                try:
                    return CrawlerAgent()
                except TypeError:
                    return CrawlerAgent(self.llm)
                    
            elif agent_name == "downloader":
                from agents.downloader_agent import DownloaderAgent
                try:
                    return DownloaderAgent()
                except TypeError:
                    return DownloaderAgent(self.llm)
                    
            elif agent_name == "classifier":
                from agents.classifier_agent import ClassifierAgent
                try:
                    return ClassifierAgent()
                except TypeError:
                    return ClassifierAgent(self.llm)
                    
        except Exception as e:
            print(f"❌ Async agent fallback also failed: {e}")
            raise
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(Dict[str, Any])
        
        # Define nodes
        builder.add_node("reasoning", self.reasoning_node)
        builder.add_node("crawler", self.crawler_node)
        builder.add_node("downloader", self.downloader_node)
        builder.add_node("classifier", self.classifier_node)
        
        # Define edges
        builder.set_entry_point("reasoning")
        builder.add_edge("reasoning", "crawler")
        builder.add_conditional_edges(
            "crawler",
            self.should_continue_to_download
        )
        builder.add_conditional_edges(
            "downloader",
            self.should_continue_to_classify
        )
        builder.add_edge("classifier", END)
        
        return builder.compile()
    
    def reasoning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initial reasoning about the workflow"""
        print("🧠 Reasoning node executing...")
        
        reasoning_prompt = f"""
        Analyze the PDF processing workflow for URL: {state.get('base_url', 'Unknown')}
        
        Current state:
        - URLs to process: {len(state.get('urls', []))}
        - Workflow: Crawl → Download → Classify
        
        Think step by step:
        1. What are potential challenges for this domain?
        2. What PDF patterns should we look for?
        3. How should we handle errors and retries?
        
        Provide a concise reasoning plan.
        """
        
        try:
            response = self.llm.invoke(reasoning_prompt)
            
            state["reasoning"] = {
                "plan": response,
                "timestamp": datetime.now().isoformat(),
                "workflow_stage": "initial_reasoning"
            }
            
            print("✅ Reasoning completed")
            self._log_state("reasoning", state)
            return state
            
        except Exception as e:
            print(f"❌ Reasoning error: {e}")
            state["reasoning_error"] = str(e)
            return state
    
    def crawler_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute crawler agent"""
        try:
            print("🕷️  Crawler node executing...")
            result = self.crawler.execute(state)
            state["crawler_result"] = result
            state["pdf_urls"] = result.get("pdf_urls_found", [])
            
            print(f"✅ Crawler found {len(state['pdf_urls'])} PDFs")
            self._log_state("crawler", state)
            return state
            
        except Exception as e:
            print(f"❌ Crawler error: {e}")
            state["crawler_error"] = str(e)
            state["pdf_urls"] = []
            self._log_state("crawler_error", state)
            return state
    
    def downloader_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute downloader agent"""
        try:
            print("📥 Downloader node executing...")
            
            download_context = {
                "pdf_urls": state.get("pdf_urls", [])
            }
            
            result = self.downloader.execute(download_context)
            state["downloader_result"] = result
            state["downloaded_files"] = result.get("downloaded_files", [])
            
            print(f"✅ Downloader processed {len(state['downloaded_files'])} files")
            self._log_state("downloader", state)
            return state
            
        except Exception as e:
            print(f"❌ Downloader error: {e}")
            state["downloader_error"] = str(e)
            state["downloaded_files"] = []
            self._log_state("downloader_error", state)
            return state
    
    def classifier_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute classifier agent"""
        try:
            print("🏷️  Classifier node executing...")
            
            classify_context = {
                "downloaded_files": state.get("downloaded_files", [])
            }
            
            result = self.classifier.execute(classify_context)
            state["classifier_result"] = result
            
            print(f"✅ Classifier completed processing")
            self._log_state("classifier", state)
            return state
            
        except Exception as e:
            print(f"❌ Classifier error: {e}")
            state["classifier_error"] = str(e)
            self._log_state("classifier_error", state)
            return state
    
    def should_continue_to_download(self, state: Dict[str, Any]) -> str:
        """Decision: Continue to download or stop"""
        pdf_urls = state.get("pdf_urls", [])
        
        if len(pdf_urls) > 0:
            print(f"➡️  Continuing to download with {len(pdf_urls)} PDFs")
            return "downloader"
        else:
            print("⏹️  No PDFs found, stopping workflow")
            return END
    
    def should_continue_to_classify(self, state: Dict[str, Any]) -> str:
        """Decision: Continue to classify or stop"""
        downloaded_files = state.get("downloaded_files", [])
        
        if len(downloaded_files) > 0:
            print(f"➡️  Continuing to classify {len(downloaded_files)} files")
            return "classifier"
        else:
            print("⏹️  No files downloaded, stopping workflow")
            return END
    
    def _log_state(self, node: str, state: Dict[str, Any]):
        """Log state for debugging"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "node": node,
            "state_keys": list(state.keys()),
            "pdf_urls_count": len(state.get("pdf_urls", [])),
            "downloaded_files_count": len(state.get("downloaded_files", []))
        }
        
        # Save to JSON log
        log_file = f"{Config.LOGS_DIR}/graph_state.jsonl"
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def execute_workflow(self, urls: List[str]) -> Dict[str, Any]:
        """Execute the complete workflow - sync version"""
        if not LANGGRAPH_AVAILABLE or self.workflow is None:
            print("⚠️  LangGraph not available, using sequential fallback")
            return self._execute_simple_workflow(urls)
        
        initial_state = {
            "urls": urls,
            "workflow_start": datetime.now().isoformat()
        }
        
        print("🚀 Starting LangGraph workflow...")
        result = self.workflow.invoke(initial_state)
        print("✅ LangGraph workflow completed")
        return result

    async def execute_workflow_async(self, urls: List[str]) -> Dict[str, Any]:
        """Async version - uses sequential for better compatibility"""
        print("🔄 Using async sequential workflow")
        return await self._execute_async_workflow(urls)
    
    def _execute_simple_workflow(self, urls: List[str]) -> Dict[str, Any]:
        """Fallback simple workflow without LangGraph"""
        print("🔄 Using sequential fallback workflow")
        results = {}
        for url in urls:
            print(f"Processing: {url}")
            
            crawl_result = self.crawler.execute({"base_url": url, "max_depth": 2})
            results[url] = {"crawling": crawl_result}
            
            if not crawl_result.get("pdf_urls_found"):
                print(f"No PDFs found at {url}")
                continue
            
            download_result = self.downloader.execute({
                "pdf_urls": crawl_result["pdf_urls_found"]
            })
            results[url]["downloading"] = download_result
            
            if not download_result.get("downloaded_files"):
                print(f"No files downloaded from {url}")
                continue
            
            classify_result = self.classifier.execute({
                "downloaded_files": download_result["downloaded_files"]
            })
            results[url]["classification"] = classify_result
            
            print(f"Completed processing {url}")
        
        return results

    async def _execute_async_workflow(self, urls: List[str]) -> Dict[str, Any]:
        """Async fallback workflow using original async agents"""
        print("🔄 Using async sequential workflow")
        # Import async agents directly
        from agents.crawler_agent import CrawlerAgent
        from agents.downloader_agent import DownloaderAgent
        from agents.classifier_agent import ClassifierAgent
        
        # Create async agents
        try:
            crawler = CrawlerAgent()
        except TypeError:
            crawler = CrawlerAgent(self.llm)
            
        try:
            downloader = DownloaderAgent()
        except TypeError:
            downloader = DownloaderAgent(self.llm)
            
        try:
            classifier = ClassifierAgent()
        except TypeError:
            classifier = ClassifierAgent(self.llm)
        
        results = {}
        for url in urls:
            print(f"Processing: {url}")
            
            # 1. Crawling phase
            crawl_result = await crawler.execute({"base_url": url, "max_depth": 2})
            results[url] = {"crawling": crawl_result}
            
            if not crawl_result.get("pdf_urls_found"):
                print(f"No PDFs found at {url}")
                continue
            
            # 2. Download phase
            download_result = await downloader.execute({
                "pdf_urls": crawl_result["pdf_urls_found"]
            })
            results[url]["downloading"] = download_result
            
            if not download_result.get("downloaded_files"):
                print(f"No files downloaded from {url}")
                continue
            
            # 3. Classification phase
            classify_result = await classifier.execute({
                "downloaded_files": download_result["downloaded_files"]
            })
            results[url]["classification"] = classify_result
            
            print(f"Completed processing {url}")
        
        return results
    
    def __del__(self):
        """Cleanup resources"""
        pass
"""
Multi-Agent Web Scraping and PDF Organization System
This system implements a complete workflow with:
- Selenium, Playwright, and Beautiful Soup for web crawling
- Memory management with state tracking
- Speed optimization with caching
- Multi-agent coordination
- PDF downloading and organization
"""
import asyncio
from langchain_ollama import ChatOllama
from agents.crawler_agent import CrawlerAgent
from agents.pdf_organizer_agent import PDFOrganizerAgent
from agents.orchestrator_agent import OrchestratorAgent
from memory.state_manager import global_state_manager
from config import Config
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    print("🚀 Starting Multi-Agent Web Scraping and PDF Organization System")
    print("="*60)
    
    # Initialize LLM
    llm = ChatOllama(model=Config.OLLAMA_MODEL)
    
    # Create agents
    crawler_agent = CrawlerAgent(llm)
    pdf_organizer_agent = PDFOrganizerAgent(llm)
    
    # Create orchestrator with all agents
    agents = {
        "crawler_agent": crawler_agent,
        "pdf_organizer_agent": pdf_organizer_agent
    }
    
    orchestrator = OrchestratorAgent(llm, agents)
    
    # Display system status
    print(f"🧠 LLM Model: {Config.OLLAMA_MODEL}")
    print(f"📊 Memory Usage: {global_state_manager.get_memory_usage()}")
    print(f"🤖 Available Agents: {list(agents.keys())}")
    
    # Define workflow context
    workflow_context = {
        "target_url": "https://example.com",  # You can change this to any website you want to crawl
        "max_depth": 2
    }
    
    print(f"\n🎯 Workflow Context: {workflow_context}")
    
    try:
        # Execute the multi-agent workflow
        print("\n🔄 Starting Multi-Agent Workflow...")
        result = await orchestrator.execute(workflow_context)
        
        print("\n✅ Workflow completed successfully!")
        print(f"📋 Results: {result}")
        
        # Display final state information
        workflow_state = orchestrator.get_workflow_state()
        print(f"\n📊 Final Workflow State:")
        print(f"   - Agents: {workflow_state['agents']}")
        print(f"   - Memory Usage: {workflow_state['state_manager_usage']['usage_percentage']:.2f}%")
        print(f"   - Recent States: {len(workflow_state['recent_states'])}")
        print(f"   - Workflow History: {len(workflow_state['workflow_history'])}")
        
        # Display organized files if available
        organized_pdfs = global_state_manager.retrieve_state("organized_pdfs")
        if organized_pdfs:
            print(f"\n📁 Organized PDFs ({len(organized_pdfs)} files):")
            for i, pdf_info in enumerate(organized_pdfs[:5]):  # Show first 5
                print(f"   {i+1}. {pdf_info['original_filename']} -> {pdf_info['organized_path']} ({pdf_info['size']} bytes)")
            if len(organized_pdfs) > 5:
                print(f"   ... and {len(organized_pdfs) - 5} more files")
        
        # Show memory usage statistics
        memory_stats = global_state_manager.get_memory_usage()
        print(f"\n💾 Memory Statistics:")
        print(f"   - Total Size: {memory_stats['total_size_mb']:.2f} MB")
        print(f"   - Usage: {memory_stats['usage_percentage']:.2f}% of max ({memory_stats['max_size_mb']:.2f} MB)")
        print(f"   - State Count: {memory_stats['state_count']}")
        
    except Exception as e:
        print(f"\n❌ Workflow failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Multi-Agent System Execution Complete")

async def demo_with_specific_site():
    """Demo function with a specific website"""
    print("🚀 Multi-Agent Demo with Example Site")
    print("="*60)
    
    # Initialize LLM
    llm = ChatOllama(model=Config.OLLAMA_MODEL)
    
    # Create agents
    crawler_agent = CrawlerAgent(llm)
    pdf_organizer_agent = PDFOrganizerAgent(llm)
    
    # Create orchestrator
    agents = {
        "crawler_agent": crawler_agent,
        "pdf_organizer_agent": pdf_organizer_agent
    }
    
    orchestrator = OrchestratorAgent(llm, agents)
    
    # Test with a site that has PDFs (or use example.com for basic functionality)
    test_context = {
        "target_url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/",  # Site with test PDFs
        "max_depth": 1
    }
    
    print(f"🔍 Crawling: {test_context['target_url']}")
    print(f"📊 Max Depth: {test_context['max_depth']}")
    
    try:
        result = await orchestrator.execute(test_context)
        print(f"\n✅ Demo completed: {result}")
        
        # Show what was found
        found_pdfs = global_state_manager.retrieve_state("found_pdfs")
        if found_pdfs:
            print(f"\n📄 PDFs Found: {len(found_pdfs)}")
            for pdf in found_pdfs[:10]:  # Show first 10
                print(f"   - {pdf}")
        
        downloaded_pdfs = global_state_manager.retrieve_state("downloaded_pdfs")
        if downloaded_pdfs:
            successful = [d for d in downloaded_pdfs if d['success']]
            failed = [d for d in downloaded_pdfs if not d['success']]
            print(f"\n📥 Download Results:")
            print(f"   - Successful: {len(successful)}")
            print(f"   - Failed: {len(failed)}")
    
    except Exception as e:
        print(f"⚠️ Demo encountered an issue: {str(e)}")
        print("This is expected if the test site doesn't have accessible PDFs")

if __name__ == "__main__":
    # Run the main workflow
    asyncio.run(main())
    
    print("\n" + "="*60)
    
    # Run a demo with a specific site
    asyncio.run(demo_with_specific_site())
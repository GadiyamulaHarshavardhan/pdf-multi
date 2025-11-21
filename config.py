import os
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class Config:
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "gpt-oss:20b"  # or "mistral", "codellama", etc.
    
    # Agent Configuration
    TEMPERATURE = 0.0  # Deterministic output
    MAX_TOKENS = 2048
    
    # File Paths
    DATA_DIR = "data"
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")
    CATEGORIZED_DIR = os.path.join(DATA_DIR, "categorized")
    LOGS_DIR = "logs"
    
    # Web Configuration
    REQUEST_TIMEOUT = 30
    MAX_CONCURRENT_REQUESTS = 3
    
    # Crawling Engine Settings
    CRAWLING_ENGINES = {
        'beautifulsoup': {
            'timeout': 30,
            'max_retries': 3
        },
        'selenium': {
            'headless': True,
            'timeout': 60,
            'window_size': '1920,1080'
        },
        'playwright': {
            'headless': True,
            'timeout': 60000,
            'wait_until': 'networkidle'
        }
    }
    
    # Dynamic content detection
    DYNAMIC_CONTENT_INDICATORS = [
        'react', 'angular', 'vue', 'webpack', 'spa',
        'single page application', 'ajax', 'fetch('
    ]

def get_ollama_llm():
    """Get Ollama LLM instance with temperature 0"""
    return Ollama(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_MODEL,
        temperature=Config.TEMPERATURE,
        num_predict=Config.MAX_TOKENS
    )

def get_chat_ollama():
    """Get ChatOllama instance for structured responses"""
    return ChatOllama(
        base_url=Config.OLLAMA_BASE_URL,
        model=Config.OLLAMA_MODEL,
        temperature=Config.TEMPERATURE,
        num_predict=Config.MAX_TOKENS
    )
    

from .autonomous_agent import AutonomousAgent
from .crawler_agent import CrawlerAgent
from .downloader_agent import DownloaderAgent
from .classifier_agent import ClassifierAgent

# Try to import sync agents if they exist
try:
    from .sync_crawler_agent import SyncCrawlerAgent
    from .sync_downloader_agent import SyncDownloaderAgent
    from .sync_classifier_agent import SyncClassifierAgent
except ImportError:
    # Sync agents might not be created yet
    SyncCrawlerAgent = None
    SyncDownloaderAgent = None
    SyncClassifierAgent = None

__all__ = [
    'AutonomousAgent',
    'CrawlerAgent',
    'DownloaderAgent', 
    'ClassifierAgent',
    'SyncCrawlerAgent',
    'SyncDownloaderAgent',
    'SyncClassifierAgent'
]
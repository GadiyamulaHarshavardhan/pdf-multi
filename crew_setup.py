from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from config import Config

class PDFCrew:
    def __init__(self):
        self.llm = Ollama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=Config.TEMPERATURE
        )
        
    def create_crew(self):
        # Define Agents
        crawler_agent = Agent(
            role='Web Crawler',
            goal='Find all internal links leading to PDFs on websites',
            backstory="""You are an expert web crawler that can systematically 
            explore websites and identify PDF documents through intelligent 
            link analysis and pattern recognition.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        downloader_agent = Agent(
            role='File Downloader',
            goal='Download PDF files reliably from identified URLs',
            backstory="""You are a reliable file downloader that can handle 
            various file formats, manage network issues, and ensure complete 
            file downloads with verification.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        classifier_agent = Agent(
            role='PDF Categorizer', 
            goal='Analyze and categorize PDFs by content type and domain',
            backstory="""You are an expert document analyst that can read, 
            understand, and categorize PDF documents based on their content, 
            structure, and metadata.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Define Tasks
        crawl_task = Task(
            description="""Crawl the website {base_url} and find all PDF links. 
            Use systematic internal link following up to depth {max_depth}.""",
            agent=crawler_agent,
            expected_output="List of PDF URLs found during crawling"
        )
        
        download_task = Task(
            description="""Download all PDF files from the provided URLs. 
            Ensure files are saved to the data/pdfs directory with proper naming.
            Handle any download errors gracefully.""",
            agent=downloader_agent,
            expected_output="List of successfully downloaded PDF files with metadata"
        )
        
        classify_task = Task(
            description="""Analyze the downloaded PDF files and categorize them 
            by content type (research, technical, business, etc.) and organize 
            by source website.""",
            agent=classifier_agent,
            expected_output="Categorized PDF files with classification metadata"
        )
        
        # Create Crew
        crew = Crew(
            agents=[crawler_agent, downloader_agent, classifier_agent],
            tasks=[crawl_task, download_task, classify_task],
            process=Process.sequential,
            verbose=True
        )
        
        return crew
    
    def execute_workflow(self, base_url: str, max_depth: int = 2):
        crew = self.create_crew()
        
        inputs = {
            'base_url': base_url,
            'max_depth': max_depth
        }
        
        result = crew.kickoff(inputs=inputs)
        return result
from .autonomous_agent import AutonomousAgent
import os
import PyPDF2
import json
from typing import Dict, List
import re

class ClassifierAgent(AutonomousAgent):
    def __init__(self):
        super().__init__(
            name="classifier_agent",
            role="PDF Categorizer",
            goal="Analyze and label PDFs by content type"
        )
        self.categories = {
            "research_paper": ["abstract", "introduction", "methodology", "results", "conclusion", "references"],
            "technical_document": ["specification", "technical", "api", "integration", "documentation"],
            "business_report": ["executive summary", "financial", "quarterly", "annual", "revenue", "profit"],
            "presentation": ["slide", "agenda", "overview", "key points", "next steps"],
            "form": ["form", "application", "request", "submit", "field"],
            "manual": ["instruction", "guide", "tutorial", "how to", "setup"],
            "unknown": []
        }
    
    async def _analyze_problem(self, context: Dict) -> Dict:
        """Analyze classification task"""
        files_to_classify = context.get('downloaded_files', [])
        
        return {
            "task": "categorize_pdfs_by_content",
            "file_count": len(files_to_classify),
            "available_categories": list(self.categories.keys()),
            "analysis_methods": ["text_analysis", "keyword_matching", "structure_analysis"]
        }
    
    async def _create_execution_plan(self, analysis: Dict) -> Dict:
        """Create classification execution plan"""
        return {
            "strategy": "multi_method_classification",
            "methods": ["keyword_analysis", "text_patterns", "metadata_analysis"],
            "confidence_threshold": 0.6,
            "fallback_category": "unknown"
        }
    
    async def _self_check_plan(self, plan: Dict) -> Dict:
        """Validate classification plan"""
        issues = []
        
        if plan["confidence_threshold"] < 0.5:
            issues.append("Low confidence threshold may cause misclassification")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Use ensemble method for better accuracy", "Add manual review for low-confidence cases"]
        }
    
    async def _execute_plan(self, plan: Dict) -> Dict:
        """Execute classification plan"""
        files_to_classify = plan.get('downloaded_files', [])
        classifications = []
        
        for file_info in files_to_classify:
            filepath = file_info['filepath']
            
            if not os.path.exists(filepath):
                self.log_action("classify_file", {"filepath": filepath}, "failed", "File not found")
                continue
            
            try:
                classification = await self._classify_single_pdf(filepath, plan)
                classifications.append({
                    **file_info,
                    "classification": classification
                })
                
                self.log_action("classify_file", {
                    "filepath": filepath,
                    "category": classification["primary_category"],
                    "confidence": classification["confidence"]
                }, "completed")
                
            except Exception as e:
                self.log_action("classify_file", {"filepath": filepath}, "failed", str(e))
                classifications.append({
                    **file_info,
                    "classification": {
                        "primary_category": "unknown",
                        "confidence": 0.0,
                        "error": str(e)
                    }
                })
        
        # Organize by website/domain
        organized_results = self._organize_by_website(classifications)
        
        return {
            "classifications": classifications,
            "organized_results": organized_results,
            "category_summary": self._create_category_summary(classifications)
        }
    
    async def _classify_single_pdf(self, filepath: str, plan: Dict) -> Dict:
        """Classify a single PDF file"""
        text_content = await self._extract_pdf_text(filepath)
        metadata = await self._extract_metadata(filepath)
        
        # Multiple classification methods
        keyword_scores = self._keyword_based_classification(text_content)
        pattern_scores = self._pattern_based_classification(text_content)
        metadata_scores = self._metadata_based_classification(metadata)
        
        # Combine scores
        combined_scores = {}
        for category in self.categories.keys():
            combined_scores[category] = (
                keyword_scores.get(category, 0) * 0.5 +
                pattern_scores.get(category, 0) * 0.3 +
                metadata_scores.get(category, 0) * 0.2
            )
        
        # Determine primary category
        primary_category = max(combined_scores.items(), key=lambda x: x[1])
        
        return {
            "primary_category": primary_category[0],
            "confidence": primary_category[1],
            "all_scores": combined_scores,
            "text_sample": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }
    
    async def _extract_pdf_text(self, filepath: str) -> str:
        """Extract text from PDF"""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.log_action("extract_text", {"filepath": filepath}, "failed", str(e))
            return ""
    
    async def _extract_metadata(self, filepath: str) -> Dict:
        """Extract PDF metadata"""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                return {
                    "title": getattr(metadata, 'title', ''),
                    "author": getattr(metadata, 'author', ''),
                    "subject": getattr(metadata, 'subject', ''),
                    "pages": len(pdf_reader.pages)
                }
        except:
            return {}
    
    def _keyword_based_classification(self, text: str) -> Dict[str, float]:
        """Classify based on keyword presence"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
            
            scores[category] = matches / len(keywords) if keywords else 0
        
        return scores
    
    def _pattern_based_classification(self, text: str) -> Dict[str, float]:
        """Classify based on text patterns and structure"""
        scores = {category: 0.0 for category in self.categories.keys()}
        
        # Research paper patterns
        if re.search(r'abstract.*introduction.*method', text, re.IGNORECASE | re.DOTALL):
            scores["research_paper"] += 0.8
        
        # Technical document patterns
        if re.search(r'specification|api|integration|technical', text, re.IGNORECASE):
            scores["technical_document"] += 0.7
        
        # Business report patterns
        if re.search(r'executive summary|financial|quarterly|annual', text, re.IGNORECASE):
            scores["business_report"] += 0.7
        
        return scores
    
    def _metadata_based_classification(self, metadata: Dict) -> Dict[str, float]:
        """Classify based on metadata"""
        scores = {category: 0.0 for category in self.categories.keys()}
        
        title = metadata.get('title', '').lower()
        subject = metadata.get('subject', '').lower()
        
        # Title-based classification
        if any(word in title for word in ['research', 'study', 'paper']):
            scores["research_paper"] += 0.6
        
        if any(word in title for word in ['manual', 'guide', 'tutorial']):
            scores["manual"] += 0.6
        
        return scores
    
    def _organize_by_website(self, classifications: List[Dict]) -> Dict:
        """Organize results by website/domain"""
        from urllib.parse import urlparse
        organized = {}
        
        for item in classifications:
            domain = urlparse(item['url']).netloc
            if domain not in organized:
                organized[domain] = []
            
            organized[domain].append(item)
        
        return organized
    
    def _create_category_summary(self, classifications: List[Dict]) -> Dict:
        """Create summary of classification results"""
        summary = {category: 0 for category in self.categories.keys()}
        
        for item in classifications:
            category = item['classification']['primary_category']
            summary[category] += 1
        
        return summary
    
    async def _verify_result(self, result: Dict) -> Dict:
        """Verify classification results"""
        issues = []
        
        if result["category_summary"]["unknown"] > len(result["classifications"]) * 0.5:
            issues.append("High number of unknown classifications")
        
        if not any(count > 0 for count in result["category_summary"].values()):
            issues.append("No successful classifications")
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "suggestions": ["Review classification keywords", "Check PDF text extraction", "Add more categories"]
        }
    
    async def _debug_and_adapt(self, plan: Dict, error: Exception):
        """Adapt classification strategy based on errors"""
        error_str = str(error).lower()
        
        if "extract" in error_str or "text" in error_str:
            plan["methods"].remove("keyword_analysis")
            plan["methods"].append("metadata_fallback")
            self.log_action("adapt_strategy", {"new_methods": plan["methods"]}, "text_extraction_issue")
"""
File Tools for PDF Agent System.

This module provides file operations, PDF processing, and file management
utilities for the multi-agent system.
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import PyPDF2
import logging

class FileTools:
    def __init__(self):
        self.logger = logging.getLogger("FileTools")
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for the system"""
        directories = [
            "data/pdfs",
            "data/categorized",
            "data/temp",
            "logs",
            "output"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def save_json(self, data: Any, filepath: str, indent: int = 2) -> bool:
        """
        Save data to JSON file with error handling
        
        Args:
            data: Data to save
            filepath: Path to save file
            indent: JSON indentation
            
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            self.logger.info(f"JSON saved successfully: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save JSON {filepath}: {str(e)}")
            return False
    
    def load_json(self, filepath: str) -> Optional[Any]:
        """
        Load data from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded data or None if error
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"JSON loaded successfully: {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON {filepath}: {str(e)}")
            return None
    
    def save_text(self, text: str, filepath: str) -> bool:
        """
        Save text to file
        
        Args:
            text: Text content to save
            filepath: Path to save file
            
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"Text saved successfully: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save text {filepath}: {str(e)}")
            return False
    
    def read_text(self, filepath: str) -> Optional[str]:
        """
        Read text from file
        
        Args:
            filepath: Path to text file
            
        Returns:
            Text content or None if error
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            self.logger.error(f"Failed to read text {filepath}: {str(e)}")
            return None
    
    def file_exists(self, filepath: str) -> bool:
        """Check if file exists and is accessible"""
        return os.path.isfile(filepath) and os.access(filepath, os.R_OK)
    
    def get_file_size(self, filepath: str) -> Optional[int]:
        """Get file size in bytes"""
        try:
            return os.path.getsize(filepath)
        except OSError:
            return None
    
    def get_file_hash(self, filepath: str, algorithm: str = "md5") -> Optional[str]:
        """
        Calculate file hash for duplicate detection
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            Hash string or None if error
        """
        try:
            hash_func = getattr(hashlib, algorithm)()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {filepath}: {str(e)}")
            return None
    
    def extract_pdf_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                result = {
                    "file_path": filepath,
                    "file_size": self.get_file_size(filepath),
                    "page_count": len(pdf_reader.pages),
                    "title": getattr(metadata, 'title', ''),
                    "author": getattr(metadata, 'author', ''),
                    "subject": getattr(metadata, 'subject', ''),
                    "creator": getattr(metadata, 'creator', ''),
                    "producer": getattr(metadata, 'producer', ''),
                    "creation_date": getattr(metadata, 'creation_date', ''),
                    "modification_date": getattr(metadata, 'modification_date', ''),
                    "extraction_timestamp": datetime.now().isoformat()
                }
                
                self.logger.info(f"PDF metadata extracted: {filepath}")
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to extract PDF metadata {filepath}: {str(e)}")
            return {
                "file_path": filepath,
                "error": str(e),
                "extraction_timestamp": datetime.now().isoformat()
            }
    
    def extract_pdf_text(self, filepath: str, max_pages: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract text content from PDF file
        
        Args:
            filepath: Path to PDF file
            max_pages: Maximum number of pages to extract (None for all)
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                if max_pages is not None:
                    pages_to_extract = min(max_pages, total_pages)
                else:
                    pages_to_extract = total_pages
                
                text_content = ""
                for page_num in range(pages_to_extract):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n\n"
                
                result = {
                    "file_path": filepath,
                    "total_pages": total_pages,
                    "pages_extracted": pages_to_extract,
                    "text_length": len(text_content),
                    "text_content": text_content.strip(),
                    "extraction_timestamp": datetime.now().isoformat()
                }
                
                self.logger.info(f"PDF text extracted: {filepath} ({pages_to_extract}/{total_pages} pages)")
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to extract PDF text {filepath}: {str(e)}")
            return {
                "file_path": filepath,
                "error": str(e),
                "extraction_timestamp": datetime.now().isoformat()
            }
    
    def organize_files_by_category(self, files: List[Dict], base_dir: str = "data/categorized") -> Dict[str, List]:
        """
        Organize files into category-based directory structure
        
        Args:
            files: List of file dictionaries with classification info
            base_dir: Base directory for organization
            
        Returns:
            Dictionary mapping categories to file lists
        """
        organized = {}
        
        for file_info in files:
            category = file_info.get('classification', {}).get('primary_category', 'unknown')
            
            if category not in organized:
                organized[category] = []
            
            # Create category directory
            category_dir = os.path.join(base_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Copy file to category directory (if source exists)
            source_path = file_info.get('filepath')
            if source_path and os.path.exists(source_path):
                filename = os.path.basename(source_path)
                dest_path = os.path.join(category_dir, filename)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    file_info['organized_path'] = dest_path
                    organized[category].append(file_info)
                    self.logger.info(f"Organized file: {filename} -> {category}")
                except Exception as e:
                    self.logger.error(f"Failed to organize file {source_path}: {str(e)}")
                    file_info['organization_error'] = str(e)
                    organized[category].append(file_info)
            else:
                organized[category].append(file_info)
        
        return organized
    
    def cleanup_old_files(self, directory: str, max_age_days: int = 30) -> Dict[str, Any]:
        """
        Clean up old files in a directory
        
        Args:
            directory: Directory to clean
            max_age_days: Maximum age of files in days
            
        Returns:
            Cleanup statistics
        """
        if not os.path.exists(directory):
            return {"error": f"Directory does not exist: {directory}"}
        
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        deleted_files = []
        deleted_size = 0
        errors = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    file_stat = os.stat(filepath)
                    file_age = current_time - file_stat.st_mtime
                    
                    if file_age > max_age_seconds:
                        file_size = file_stat.st_size
                        os.remove(filepath)
                        deleted_files.append(filepath)
                        deleted_size += file_size
                        self.logger.info(f"Cleaned up old file: {filepath}")
                        
                except Exception as e:
                    errors.append(f"{filepath}: {str(e)}")
                    self.logger.error(f"Failed to clean up file {filepath}: {str(e)}")
        
        return {
            "deleted_files_count": len(deleted_files),
            "deleted_size_bytes": deleted_size,
            "deleted_size_mb": round(deleted_size / (1024 * 1024), 2),
            "deleted_files": deleted_files,
            "errors": errors,
            "cleanup_timestamp": datetime.now().isoformat()
        }
    
    def create_backup(self, source_dir: str, backup_dir: str) -> Dict[str, Any]:
        """
        Create backup of a directory
        
        Args:
            source_dir: Source directory to backup
            backup_dir: Backup destination directory
            
        Returns:
            Backup operation results
        """
        if not os.path.exists(source_dir):
            return {"error": f"Source directory does not exist: {source_dir}"}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
        
        try:
            shutil.copytree(source_dir, backup_path)
            self.logger.info(f"Backup created: {source_dir} -> {backup_path}")
            
            return {
                "source_directory": source_dir,
                "backup_directory": backup_path,
                "backup_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Backup failed: {source_dir} -> {backup_path}: {str(e)}")
            return {
                "source_directory": source_dir,
                "backup_directory": backup_path,
                "error": str(e),
                "backup_timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
    
    def get_directory_stats(self, directory: str) -> Dict[str, Any]:
        """
        Get statistics for a directory
        
        Args:
            directory: Directory to analyze
            
        Returns:
            Directory statistics
        """
        if not os.path.exists(directory):
            return {"error": f"Directory does not exist: {directory}"}
        
        total_files = 0
        total_size = 0
        file_types = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(filepath)
                    total_files += 1
                    total_size += file_size
                    
                    # Count by file extension
                    _, ext = os.path.splitext(file)
                    ext = ext.lower() if ext else "no_extension"
                    file_types[ext] = file_types.get(ext, 0) + 1
                    
                except OSError:
                    continue
        
        return {
            "directory": directory,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def validate_pdf_file(self, filepath: str) -> Dict[str, Any]:
        """
        Validate PDF file integrity and basic properties
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            Validation results
        """
        if not self.file_exists(filepath):
            return {
                "file_path": filepath,
                "valid": False,
                "error": "File does not exist or is not accessible"
            }
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                validation_result = {
                    "file_path": filepath,
                    "valid": True,
                    "file_size": self.get_file_size(filepath),
                    "page_count": len(pdf_reader.pages),
                    "is_encrypted": pdf_reader.is_encrypted,
                    "validation_timestamp": datetime.now().isoformat()
                }
                
                # Try to read first page as additional validation
                if pdf_reader.pages:
                    try:
                        first_page = pdf_reader.pages[0]
                        sample_text = first_page.extract_text()[:100]  # First 100 chars
                        validation_result["sample_text"] = sample_text
                    except:
                        validation_result["sample_text"] = "Unable to extract text"
                
                self.logger.info(f"PDF validated: {filepath}")
                return validation_result
                
        except Exception as e:
            self.logger.error(f"PDF validation failed {filepath}: {str(e)}")
            return {
                "file_path": filepath,
                "valid": False,
                "error": str(e),
                "validation_timestamp": datetime.now().isoformat()
            }


# Singleton instance for easy access
file_tools = FileTools()
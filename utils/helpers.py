import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.settings import settings

logger = logging.getLogger(__name__)

class ArabicTextProcessor:
    """Utility class for Arabic text processing"""
    
    @staticmethod
    def normalize_arabic_text(text: str) -> str:
        """Normalize Arabic text while preserving meaning"""
        if not text:
            return text
            
        # Normalize Arabic characters
        replacements = {
            'إ': 'ا', 'أ': 'ا', 'آ': 'ا', 'ة': 'ه',
            'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraphs
        
        return text.strip()
    
    @staticmethod
    def extract_arabic_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from Arabic text"""
        # Remove punctuation
        text = re.sub(r'[.,،;؛!؟?()\[\]{}«»]', ' ', text)
        
        # Split into words and filter
        words = text.split()
        arabic_words = [word for word in words if re.search(r'[\u0600-\u06FF]', word)]
        
        # Simple frequency-based keyword extraction
        from collections import Counter
        word_freq = Counter(arabic_words)
        
        # Remove common stop words
        stop_words = settings.ARABIC_STOP_WORDS
        keywords = [word for word, count in word_freq.most_common() 
                   if word not in stop_words and len(word) > 2]
        
        return keywords[:max_keywords]
    
    @staticmethod
    def detect_arabic_dialect(text: str) -> str:
        """Basic Arabic dialect detection"""
        text_lower = text.lower()
        
        # Kuwaiti dialect indicators
        kuwaiti_indicators = ['عسب', 'شلون', 'شكو', 'وايد', 'بشتغل', 'شنو']
        if any(indicator in text_lower for indicator in kuwaiti_indicators):
            return 'kuwaiti'
        
        # MSA indicators
        if re.search(r'\b(الذي|التي|أما|إذ|إذا)\b', text):
            return 'msa'
        
        return 'unknown'

class FileUtilities:
    """Utility functions for file operations"""
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Generate SHA256 hash of file content"""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    @staticmethod
    def safe_file_operation(func):
        """Decorator for safe file operations"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"File operation failed: {str(e)}")
                return None
        return wrapper
    
    @staticmethod
    def find_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
        """Find files by extension in directory"""
        found_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    found_files.append(os.path.join(root, file))
        return found_files

class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timer for an operation"""
        self.start_times[operation] = datetime.now()
    
    def stop_timer(self, operation: str) -> float:
        """Stop timer and return duration"""
        if operation not in self.start_times:
            return 0.0
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        self.metrics[operation] = duration
        return duration
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics"""
        return self.metrics.copy()

class ConfigValidator:
    """Configuration validation utilities"""
    
    @staticmethod
    def validate_settings() -> List[str]:
        """Validate system settings and return errors"""
        errors = []
        
        # Check required directories
        required_dirs = [settings.DATA_DIR, settings.PROCESSED_DIR, settings.VECTOR_STORE_PATH]
        for directory in required_dirs:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except Exception as e:
                    errors.append(f"Failed to create directory {directory}: {str(e)}")
        
        # Check supported extensions
        if not settings.SUPPORTED_EXTENSIONS:
            errors.append("No supported file extensions configured")
        
        # Check chunk sizes
        for content_type, sizes in settings.CHUNK_SIZES.items():
            if sizes['min'] >= sizes['max']:
                errors.append(f"Invalid chunk sizes for {content_type}: min >= max")
        
        return errors

# Utility functions
def create_backup(filepath: str) -> bool:
    """Create backup of a file"""
    try:
        if os.path.exists(filepath):
            backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(filepath, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return True
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
    return False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def safe_json_load(filepath: str) -> Any:
    """Safely load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON load failed: {filepath} - {str(e)}")
        return None
import os
import tempfile
from typing import Optional, List
from abc import ABC, abstractmethod
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from langchain_community.document_loaders import (
        PyPDFLoader, Docx2txtLoader, TextLoader, 
        UnstructuredFileLoader, UnstructuredEPubLoader,
        JSONLoader, BSHTMLLoader
    )
    from langchain_community.document_loaders.unstructured import UnstructuredLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available, using fallback loaders")

class BaseFileHandler(ABC):
    """Base class for file handlers"""
    
    @abstractmethod
    def extract_text(self, filepath: str) -> str:
        pass
    
    @staticmethod
    def safe_read(filepath: str) -> str:
        """Safely read file with encoding handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {filepath}: {str(e)}")

class LangChainFileHandler(BaseFileHandler):
    """Enhanced file handler using LangChain loaders"""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for this handler")
    
    def extract_text(self, filepath: str) -> str:
        """Extract text using LangChain loaders with error handling"""
        try:
            loader = self._get_loader(filepath)
            documents = loader.load()
            
            if not documents:
                return ""
            
            # Combine all document pages/sections
            text_content = "\n\n".join([doc.page_content for doc in documents])
            return self._clean_text(text_content)
            
        except Exception as e:
            print(f"❌ LangChain loader failed for {filepath}: {str(e)}")
            # Fallback to basic handler
            return FallbackFileHandler().extract_text(filepath)
    
    def _get_loader(self, filepath: str):
        """Get appropriate LangChain loader for file type"""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.txt': TextLoader,
            '.json': lambda path: JSONLoader(path, jq_schema='.', text_content=False),
            '.jsonl': lambda path: JSONLoader(path, jq_schema='.', text_content=False),
            '.html': BSHTMLLoader,
            '.htm': BSHTMLLoader,
            '.epub': UnstructuredEPubLoader,
        }
        
        if file_ext in loader_map:
            return loader_map[file_ext](filepath)
        else:
            # Use unstructured loader as fallback
            return UnstructuredFileLoader(filepath)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text while preserving Arabic content"""
        # Remove excessive whitespace but preserve Arabic text structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.isspace():
                # Preserve Arabic punctuation and diacritics
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

class FallbackFileHandler(BaseFileHandler):
    """Fallback handler when LangChain is not available"""
    
    def extract_text(self, filepath: str) -> str:
        """Basic text extraction for common formats"""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext in ['.txt', '.json', '.jsonl']:
            return self.safe_read(filepath)
        else:
            raise ValueError(f"Unsupported file format for fallback: {file_ext}")

class FileHandlerFactory:
    """Factory to create appropriate file handlers"""
    
    @staticmethod
    def get_handler(filepath: str) -> BaseFileHandler:
        """Get the best available handler for the file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check if LangChain is available and file type is supported
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.json', 
                               '.jsonl', '.html', '.htm', '.epub']
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext not in supported_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if LANGCHAIN_AVAILABLE:
            return LangChainFileHandler()
        else:
            if file_ext in ['.txt', '.json', '.jsonl']:
                return FallbackFileHandler()
            else:
                raise ValueError(f"LangChain required for {file_ext} files")

# Utility function for quick text extraction
def extract_text_from_file(filepath: str) -> str:
    """Convenience function to extract text from any supported file"""
    handler = FileHandlerFactory.get_handler(filepath)
    return handler.extract_text(filepath)
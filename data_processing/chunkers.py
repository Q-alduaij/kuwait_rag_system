import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from models.schemas import DocumentChunk, ContentType, SensitivityLevel, LanguageVariant
from config.settings import settings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class ArabicAwareTextSplitter:
    """Text splitter optimized for Arabic content"""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Arabic-specific separators (priority order)
        self.separators = [
            "\n\n",       # Double newlines (paragraphs)
            "\n",         # Single newlines
            "۔", "۔ ",    Arabic full stop
            ". ",         # Period with space
            "؟", "? ",    # Question marks
            "!", "! ",    # Exclamation marks
            "،", ", ",    # Arabic comma
            "; ",         # Semicolon
            " ",          # Spaces
            "",           # No separator (character level)
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text using Arabic-aware separators"""
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        chunks = []
        current_chunk = ""
        
        for char in text:
            current_chunk += char
            
            # Check if we need to split
            if len(current_chunk) >= self.chunk_size:
                # Try to find the best split point
                split_index = self._find_split_point(current_chunk)
                if split_index > 0:
                    chunk = current_chunk[:split_index].strip()
                    if chunk:
                        chunks.append(chunk)
                    current_chunk = current_chunk[split_index:]
                else:
                    # Force split at chunk size
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _find_split_point(self, text: str) -> int:
        """Find the best point to split Arabic text"""
        # Prefer splitting at Arabic punctuation
        for separator in self.separators:
            if separator:
                index = text.rfind(separator)
                if index > len(text) * 0.3:  # Don't split too early
                    return index + len(separator)
        return -1

class QuranChunker:
    """Chunker specialized for Quranic text"""
    
    @staticmethod
    def chunk_by_ayah(text: str, filename: str) -> List[DocumentChunk]:
        chunks = []
        
        # Enhanced pattern for Quranic verse detection
        patterns = [
            # Pattern: سورة البقرة آية 255: النص
            r"سورة\s+([^\s]+)\s+آية\s+(\d+):\s*(.*?)(?=سورة\s+[^\s]+\s+آية\s+\d+:|$)",
            # Pattern: Surah Al-Baqarah 255: Text
            r"Surah\s+([^\s]+)\s+Ayah\s+(\d+):\s*(.*?)(?=Surah\s+[^\s]+\s+Ayah\s+\d+:|$)",
            # Pattern: ﴿255﴾ النص
            r"﴿(\d+)﴾\s*(.*?)(?=﴿\d+﴾|$)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.UNICODE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    if "﴿" in pattern:  # Special verse numbering
                        ayah_number = int(match.group(1))
                        ayah_text = match.group(2).strip()
                        surah_name = "Unknown"  # Would need context
                    else:
                        surah_name = match.group(1)
                        ayah_number = int(match.group(2))
                        ayah_text = match.group(3).strip()
                    
                    if not ayah_text or len(ayah_text) < 5:
                        continue
                    
                    # Generate unique content ID
                    content_id = f"quran_{surah_name}_ayah_{ayah_number}"
                    
                    # Create SHA256 hash
                    content_hash = hashlib.sha256(ayah_text.encode('utf-8')).hexdigest()
                    
                    chunk = DocumentChunk(
                        content=ayah_text,
                        metadata={
                            "source_type": ContentType.QURAN,
                            "source_file": filename,
                            "content_id": content_id,
                            "sensitivity_level": SensitivityLevel.HIGH,
                            "language_variant": LanguageVariant.CLASSICAL_ARABIC,
                            "surah_name": surah_name,
                            "ayah_number": ayah_number,
                            "hafs_numbering": f"{surah_name}:{ayah_number}",
                            "content_tags": ["قرآن", "آية", surah_name],
                            "sha256_hash": content_hash,
                            "processing_timestamp": datetime.now().isoformat()
                        }
                    )
                    chunks.append(chunk)
        
        return chunks

class LegalChunker:
    """Chunker for Kuwaiti legal documents"""
    
    @staticmethod
    def chunk_by_article(text: str, filename: str) -> List[DocumentChunk]:
        chunks = []
        
        # Enhanced patterns for legal articles
        patterns = [
            # Kuwaiti Arabic pattern
            r"المادة\s+([\d]+)[:\-]\s*(.*?)(?=المادة\s+[\d]+[:\\-]|$)",
            # English pattern
            r"Article\s+([\d]+)[:\-]\s*(.*?)(?=Article\s+[\d]+[:\\-]|$)",
            # Law section pattern
            r"القسم\s+([\d]+)[:\-]\s*(.*?)(?=القسم\s+[\d]+[:\\-]|$)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.UNICODE)
            
            for match in matches:
                article_num = match.group(1)
                article_text = match.group(2).strip()
                
                if not article_text or len(article_text) < 10:
                    continue
                
                content_id = f"law_{filename}_article_{article_num}"
                content_hash = hashlib.sha256(article_text.encode('utf-8')).hexdigest()
                
                chunk = DocumentChunk(
                    content=article_text,
                    metadata={
                        "source_type": ContentType.LEGAL,
                        "source_file": filename,
                        "content_id": content_id,
                        "sensitivity_level": SensitivityLevel.MEDIUM,
                        "language_variant": LanguageVariant.MSA,
                        "law_name": filename,
                        "article_number": article_num,
                        "content_tags": ["قانون", "مادة", "كويتي", filename],
                        "sha256_hash": content_hash,
                        "processing_timestamp": datetime.now().isoformat()
                    }
                )
                chunks.append(chunk)
        
        return chunks

class DynamicChunker:
    """Content-aware dynamic chunker with LangChain integration"""
    
    def __init__(self):
        self.chunkers = {
            ContentType.QURAN: QuranChunker(),
            ContentType.LEGAL: LegalChunker(),
        }
        
        # Initialize LangChain text splitter if available
        if LANGCHAIN_AVAILABLE:
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                length_function=len,
                separators=["\n\n", "\n", "۔", ". ", "؟", "!", "،", " ", ""]
            )
        else:
            self.langchain_splitter = None
    
    def chunk_document(self, content: str, filename: str, content_type: ContentType) -> List[DocumentChunk]:
        """Chunk document based on content type with fallback strategies"""
        
        # Use specialized chunker if available
        if content_type in self.chunkers:
            try:
                chunks = self.chunkers[content_type].chunk_by_ayah(content, filename)
                if chunks:
                    return chunks
            except Exception as e:
                print(f"⚠️ Specialized chunker failed for {content_type}: {e}")
        
        # Fallback to appropriate chunking strategy
        chunk_config = settings.CHUNK_SIZES.get(content_type.value, settings.CHUNK_SIZES["mixed"])
        
        if self.langchain_splitter:
            # Use LangChain splitter with Arabic support
            return self._chunk_with_langchain(content, filename, content_type, chunk_config)
        else:
            # Use custom Arabic-aware splitter
            return self._chunk_with_custom(content, filename, content_type, chunk_config)
    
    def _chunk_with_langchain(self, content: str, filename: str, content_type: ContentType, 
                             config: Dict) -> List[DocumentChunk]:
        """Chunk using LangChain's text splitter"""
        # Adjust splitter parameters for this content type
        self.langchain_splitter._chunk_size = config["max"]
        self.langchain_splitter._chunk_overlap = config["overlap"]
        
        chunks = []
        text_chunks = self.langchain_splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    "source_type": content_type,
                    "source_file": filename,
                    "content_id": f"{content_type.value}_{i}_{hash(chunk_text)}",
                    "sensitivity_level": self._get_sensitivity_level(content_type),
                    "language_variant": self._get_language_variant(content_type),
                    "content_tags": [content_type.value, f"chunk_{i}"],
                    "sha256_hash": content_hash,
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_with_custom(self, content: str, filename: str, content_type: ContentType,
                          config: Dict) -> List[DocumentChunk]:
        """Chunk using custom Arabic-aware splitter"""
        splitter = ArabicAwareTextSplitter(
            chunk_size=config["max"],
            chunk_overlap=config["overlap"]
        )
        
        chunks = []
        text_chunks = splitter.split_text(content)
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
                
            content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata={
                    "source_type": content_type,
                    "source_file": filename,
                    "content_id": f"{content_type.value}_{i}",
                    "sensitivity_level": self._get_sensitivity_level(content_type),
                    "language_variant": self._get_language_variant(content_type),
                    "content_tags": [content_type.value],
                    "sha256_hash": content_hash,
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_sensitivity_level(self, content_type: ContentType) -> SensitivityLevel:
        """Determine sensitivity level based on content type"""
        sensitivity_map = {
            ContentType.QURAN: SensitivityLevel.HIGH,
            ContentType.TAFSIR: SensitivityLevel.HIGH,
            ContentType.HADITH: SensitivityLevel.HIGH,
            ContentType.ISLAMIC_HISTORY: SensitivityLevel.MEDIUM,
            ContentType.LEGAL: SensitivityLevel.MEDIUM,
            ContentType.HISTORY: SensitivityLevel.LOW,
            ContentType.CULTURE: SensitivityLevel.LOW,
            ContentType.KUWAITI_DIALECT: SensitivityLevel.LOW,
            ContentType.MIXED: SensitivityLevel.LOW
        }
        return sensitivity_map.get(content_type, SensitivityLevel.MEDIUM)
    
    def _get_language_variant(self, content_type: ContentType) -> LanguageVariant:
        """Determine language variant based on content type"""
        variant_map = {
            ContentType.QURAN: LanguageVariant.CLASSICAL_ARABIC,
            ContentType.TAFSIR: LanguageVariant.MSA,
            ContentType.HADITH: LanguageVariant.CLASSICAL_ARABIC,
            ContentType.ISLAMIC_HISTORY: LanguageVariant.MSA,
            ContentType.LEGAL: LanguageVariant.MSA,
            ContentType.HISTORY: LanguageVariant.MSA,
            ContentType.CULTURE: LanguageVariant.MSA,
            ContentType.KUWAITI_DIALECT: LanguageVariant.KUWAITI_DIALECT,
            ContentType.MIXED: LanguageVariant.MSA
        }
        return variant_map.get(content_type, LanguageVariant.MSA)
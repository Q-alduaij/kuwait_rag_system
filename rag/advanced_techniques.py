import logging
from typing import List, Dict, Any, Optional
from models.schemas import DocumentChunk, SearchQuery
from rag.generator import LocalLLMGenerator

logger = logging.getLogger(__name__)

class QueryEnhancementEngine:
    """
    Advanced query enhancement techniques :cite[1]
    Implements HyDE, query expansion, and decomposition
    """
    
    def __init__(self):
        self.llm_generator = LocalLLMGenerator()
    
    def enhance_query(self, original_query: str, technique: str = "hyde") -> str:
        """
        Enhance query using various techniques :cite[1]
        """
        if technique == "hyde":
            return self._hyde_enhancement(original_query)
        elif technique == "expansion":
            return self._query_expansion(original_query)
        elif technique == "decomposition":
            return self._query_decomposition(original_query)
        else:
            return original_query
    
    def _hyde_enhancement(self, query: str) -> str:
        """Hypothetical Document Embedding (HyDE) technique :cite[1]"""
        try:
            prompt = f"""
            بناءً على السؤال التالي، قم بإنشاء إجابة نموذجية تحتوي على المعلومات المطلوبة.
            السؤال: {query}
            
            الإجابة النموذجية يجب أن:
            - تكون باللغة العربية الفصحى
            - تحتوي على المصطلحات الرئيسية المتوقعة
            - تكون دقيقة ومفصلة
            - تشير إلى المصادر عندما يكون ذلك مناسباً
            
            الإجابة النموذجية:
            """
            
            hypothetical_answer = self.llm_generator.generate_answer(prompt, "")
            
            # Combine original query with hypothetical answer
            enhanced_query = f"{query} {hypothetical_answer}"
            return enhanced_query[:1000]  # Limit length
            
        except Exception as e:
            logger.error(f"HyDE enhancement failed: {str(e)}")
            return query
    
    def _query_expansion(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        # Arabic-specific query expansion
        expansion_terms = self._get_arabic_expansion_terms(query)
        expanded_query = query + " " + " ".join(expansion_terms)
        
        return expanded_query
    
    def _query_decomposition(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries :cite[1]"""
        try:
            prompt = f"""
            قم بتقسيم السؤال المعقد التالي إلى أسئلة فرعية أبسط:
            السؤال: {query}
            
            الأسئلة الفرعية:
            """
            
            decomposition_result = self.llm_generator.generate_answer(prompt, "")
            sub_queries = [q.strip() for q in decomposition_result.split('\n') if q.strip()]
            
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {str(e)}")
            return [query]
    
    def _get_arabic_expansion_terms(self, query: str) -> List[str]:
        """Get Arabic-specific expansion terms"""
        # Domain-specific expansion rules
        expansion_rules = {
            "قانون": ["تشريع", "نظام", "مادة قانونية", "بند"],
            "قرآن": ["آية", "سورة", "تفسير", "قرآن كريم"],
            "حديث": ["سنة", "رواية", "أثر", "حديث نبوي"],
            "تاريخ": ["حدث", "وقائع", "تواريخ", "تاريخي"]
        }
        
        expansion_terms = []
        for term, expansions in expansion_rules.items():
            if term in query:
                expansion_terms.extend(expansions)
        
        return expansion_terms[:5]  # Limit expansion terms

class AdaptiveChunkingEngine:
    """
    Adaptive chunking based on content type and structure :cite[1]
    """
    
    def adaptive_chunk(self, content: str, content_type: str, **kwargs) -> List[DocumentChunk]:
        """Adapt chunking strategy based on content type"""
        chunking_strategies = {
            "quran": self._chunk_quranic_text,
            "legal": self._chunk_legal_text,
            "historical": self._chunk_historical_text,
            "dialogue": self._chunk_dialogue_text
        }
        
        chunker = chunking_strategies.get(content_type, self._chunk_generic_text)
        return chunker(content, **kwargs)
    
    def _chunk_quranic_text(self, content: str, **kwargs) -> List[DocumentChunk]:
        """Chunk Quranic text by verses with context"""
        # Enhanced Quranic chunking with cross-verse context
        chunks = []
        verses = content.split('آية')
        
        for i, verse in enumerate(verses):
            if verse.strip():
                # Include context from previous and next verses
                start_idx = max(0, i-1)
                end_idx = min(len(verses), i+2)
                context_verse = ' '.join(verses[start_idx:end_idx])
                
                # Create chunk with context
                chunk = DocumentChunk(
                    content=context_verse,
                    metadata=self._create_metadata("quran", f"verse_{i}")
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_legal_text(self, content: str, **kwargs) -> List[DocumentChunk]:
        """Chunk legal text with hierarchical structure"""
        chunks = []
        
        # Split by articles and sections
        articles = re.split(r'المادة\s+\d+', content)
        
        for i, article in enumerate(articles):
            if article.strip():
                # Include hierarchical context
                chunk = DocumentChunk(
                    content=f"المادة {i}: {article}",
                    metadata=self._create_metadata("legal", f"article_{i}")
                )
                chunks.append(chunk)
        
        return chunks

class MultiModalRAGEngine:
    """
    Multi-modal RAG support for future expansion :cite[1]
    """
    
    def __init__(self):
        self.supported_modalities = ["text", "image", "audio"]
    
    def process_multimodal_content(self, content_path: str, modality: str) -> List[DocumentChunk]:
        """Process multi-modal content for RAG"""
        if modality == "text":
            return self._process_text_content(content_path)
        elif modality == "image":
            return self._process_image_content(content_path)
        elif modality == "audio":
            return self._process_audio_content(content_path)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _process_image_content(self, image_path: str) -> List[DocumentChunk]:
        """Process image content using OCR and image captioning"""
        # This would be implemented with OCR and vision models
        try:
            import pytesseract
            from PIL import Image
            
            # Extract text from image
            text = pytesseract.image_to_string(Image.open(image_path), lang='ara')
            
            # Create document chunk
            chunk = DocumentChunk(
                content=text,
                metadata=self._create_metadata("image", image_path)
            )
            
            return [chunk]
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return []

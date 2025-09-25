import re
import logging
from typing import List, Dict, Any, Optional
from rag.vector_store import EnhancedVectorStoreManager
from rag.generator import LocalLLMGenerator
from models.schemas import ContentType, SensitivityLevel

logger = logging.getLogger(__name__)

class ArabicQAEngine:
    """Enhanced QA engine with Arabic-specific optimizations"""
    
    def __init__(self):
        self.vector_store = EnhancedVectorStoreManager()
        self.llm_generator = LocalLLMGenerator()
        
        # Arabic-specific query patterns
        self.query_patterns = {
            "legal": [
                r".*(قانون|مادة|تشريع|دستور).*",
                r".*(law|article|legislation|constitution).*"
            ],
            "religious": [
                r".*(قرآن|آية|سورة|تفسير|حديث|فقه).*",
                r".*(quran|ayah|surah|tafsir|hadith|fiqh).*"
            ],
            "historical": [
                r".*(تاريخ|حدث|وثيقة|تراث).*",
                r".*(history|event|document|heritage).*"
            ],
            "cultural": [
                r".*(ثقافة|عادات|تقاليد|تراث).*",
                r".*(culture|customs|traditions|heritage).*"
            ]
        }

    def classify_query_type(self, query: str) -> Dict[str, Any]:
        """Classify query type based on Arabic content patterns"""
        query_lower = query.lower()
        
        query_type = "general"
        confidence = 0.0
        content_types = []
        
        for q_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    query_type = q_type
                    confidence = 0.8
                    break
        
        # Map query type to content types
        type_mapping = {
            "legal": [ContentType.LEGAL],
            "religious": [ContentType.QURAN, ContentType.TAFSIR, ContentType.HADITH],
            "historical": [ContentType.HISTORY, ContentType.ISLAMIC_HISTORY],
            "cultural": [ContentType.CULTURE, ContentType.KUWAITI_DIALECT],
            "general": []  # Search all types
        }
        
        content_types = type_mapping.get(query_type, [])
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "content_types": [ct.value for ct in content_types],
            "suggested_filters": self._get_suggested_filters(query_type)
        }

    def _get_suggested_filters(self, query_type: str) -> Dict[str, Any]:
        """Get suggested filters based on query type"""
        filters = {
            "legal": {"source_type": "legal", "sensitivity_level": {"$in": ["low", "medium"]}},
            "religious": {"source_type": {"$in": ["quran", "tafsir", "hadith"]}, "sensitivity_level": "high"},
            "historical": {"source_type": {"$in": ["history", "islamic_history"]}},
            "cultural": {"source_type": {"$in": ["culture", "kuwaiti_dialect"]}},
            "general": None
        }
        return filters.get(query_type)

    def build_arabic_context(self, results: Dict) -> str:
        """Build context optimized for Arabic LLM understanding"""
        if not results["documents"]:
            return "لا توجد معلومات كافية للإجابة على هذا السؤال."
        
        context_parts = ["المعلومات المتاحة:\n"]
        
        for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
            source_info = self._format_arabic_source(metadata)
            context_parts.append(f"{i+1}. {doc}\n   📚 {source_info}\n")
        
        return "\n".join(context_parts)

    def _format_arabic_source(self, metadata: Dict) -> str:
        """Format source information in Arabic"""
        source_type = metadata.get('source_type', 'غير معروف')
        source_file = metadata.get('source_file', 'غير معروف')
        
        if source_type == 'quran':
            surah = metadata.get('surah_name', 'غير معروف')
            ayah = metadata.get('ayah_number', 'غير معروف')
            return f"القرآن الكريم - سورة {surah} آية {ayah}"
        elif source_type == 'legal':
            law_name = metadata.get('law_name', 'غير معروف')
            article = metadata.get('article_number', 'غير معروف')
            return f"القانون الكويتي - {law_name} المادة {article}"
        elif source_type == 'tafsir':
            return f"كتاب تفسير - {source_file}"
        else:
            return f"{source_type} - {source_file}"

class EnhancedQASystem:
    """Main QA system with enhanced Arabic capabilities"""
    
    def __init__(self):
        self.qa_engine = ArabicQAEngine()
        self.vector_store = EnhancedVectorStoreManager()
        
    def answer_question(self, question: str, query_type: str = "auto", 
                       max_results: int = 7) -> Dict[str, Any]:
        """Answer question with enhanced Arabic handling"""
        try:
            # Auto-detect query type if not specified
            if query_type == "auto":
                classification = self.qa_engine.classify_query_type(question)
                query_type = classification["query_type"]
                content_types = classification["content_types"]
                filters = classification["suggested_filters"]
            else:
                content_types = self._map_query_type_to_content(query_type)
                filters = None
            
            # Retrieve relevant chunks
            results = self.vector_store.search(
                question, 
                n_results=max_results,
                filters=filters,
                content_types=content_types
            )
            
            if not results.get("documents"):
                return self._handle_no_results(question)
            
            # Build context
            context = self.qa_engine.build_arabic_context(results)
            
            # Generate answer
            answer = self.qa_engine.llm_generator.generate_answer(question, context, query_type)
            
            return {
                "question": question,
                "answer": answer,
                "query_type": query_type,
                "sources": results["metadatas"],
                "context_used": context,
                "retrieved_documents": len(results["documents"]),
                "confidence_scores": results.get("scores", []),
                "classification": self.qa_engine.classify_query_type(question) if query_type == "auto" else None
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": "عذرًا، حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة مرة أخرى.",
                "error": str(e),
                "sources": []
            }

    def _map_query_type_to_content(self, query_type: str) -> List[str]:
        """Map query type to content types"""
        mapping = {
            "legal": ["legal"],
            "religious": ["quran", "tafsir", "hadith"],
            "historical": ["history", "islamic_history"],
            "cultural": ["culture", "kuwaiti_dialect"],
            "general": []  # All types
        }
        return mapping.get(query_type, [])

    def _handle_no_results(self, question: str) -> Dict[str, Any]:
        """Handle cases where no results are found"""
        return {
            "question": question,
            "answer": "عذرًا، لم أتمكن من العثور على معلومات كافية للإجابة على سؤالك في المصادر المتاحة.",
            "sources": [],
            "suggestions": [
                "تأكد من صياغة السؤال باللغة العربية الفصحى أو اللهجة الكويتية",
                "حاول استخدام مصطلحات أكثر شيوعًا",
                "قسّم السؤال إلى أجزاء أصغر إذا كان معقدًا"
            ]
        }

# Specialized QA systems
class LegalQASystem(EnhancedQASystem):
    """Specialized QA system for legal questions"""
    
    def answer_legal_question(self, question: str, include_comparative: bool = True) -> Dict[str, Any]:
        """Answer legal questions with comparative analysis"""
        result = self.answer_question(question, "legal")
        
        if include_comparative and result.get("sources"):
            result["comparative_analysis"] = self._generate_comparative_analysis(result["sources"])
        
        return result
    
    def _generate_comparative_analysis(self, sources: List[Dict]) -> str:
        """Generate comparative analysis of legal sources"""
        # This would use the LLM to compare different legal provisions
        return "تحليل مقارن للقوانين ذات الصلة (يحتاج تطوير إضافي)"

class ReligiousQASystem(EnhancedQASystem):
    """Specialized QA system for religious questions"""
    
    def answer_religious_question(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Answer religious questions with proper source attribution"""
        result = self.answer_question(question, "religious")
        
        if include_sources and result.get("sources"):
            result["source_hierarchy"] = self._generate_source_hierarchy(result["sources"])
        
        return result
    
    def _generate_source_hierarchy(self, sources: List[Dict]) -> List[str]:
        """Generate hierarchical source attribution"""
        hierarchy = []
        for source in sources:
            if source.get('source_type') == 'quran':
                hierarchy.append(f"القرآن الكريم - سورة {source.get('surah_name')} آية {source.get('ayah_number')}")
            elif source.get('source_type') == 'hadith':
                hierarchy.append(f"الحديث النبوي - {source.get('source_file')}")
            elif source.get('source_type') == 'tafsir':
                hierarchy.append(f"التفسير - {source.get('source_file')}")
        
        return hierarchy

class CulturalQASystem(EnhancedQASystem):
    """Specialized QA system for cultural questions"""
    
    def answer_cultural_question(self, question: str, include_context: bool = True) -> Dict[str, Any]:
        """Answer cultural questions with historical context"""
        result = self.answer_question(question, "cultural")
        
        if include_context and result.get("sources"):
            result["historical_context"] = self._generate_historical_context(result["sources"])
        
        return result
    
    def _generate_historical_context(self, sources: List[Dict]) -> str:
        """Generate historical context for cultural information"""
        return "سياق تاريخي للمعلومات الثقافية (يحتاج تطوير إضافي)"
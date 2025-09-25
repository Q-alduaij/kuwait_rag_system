import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from models.schemas import DocumentMetadata, ContentType, SensitivityLevel, LanguageVariant

class MetadataExtractor:
    """Base class for metadata extraction"""
    
    def extract_metadata(self, content: str, filename: str, content_type: ContentType) -> DocumentMetadata:
        """Extract metadata from content"""
        raise NotImplementedError("Subclasses must implement this method")

class QuranMetadataExtractor(MetadataExtractor):
    """Metadata extractor for Quranic text"""
    
    def extract_metadata(self, content: str, filename: str, content_type: ContentType) -> DocumentMetadata:
        # Extract Quran-specific metadata
        surah_name, ayah_number = self._extract_quran_references(content)
        
        return DocumentMetadata(
            source_type=ContentType.QURAN,
            source_file=filename,
            content_id=f"quran_{surah_name}_{ayah_number}",
            sensitivity_level=SensitivityLevel.HIGH,
            language_variant=LanguageVariant.CLASSICAL_ARABIC,
            surah_name=surah_name,
            ayah_number=ayah_number,
            hafs_numbering=f"{surah_name}:{ayah_number}",
            content_tags=["قرآن", "آية", surah_name],
            sha256_hash=hashlib.sha256(content.encode('utf-8')).hexdigest(),
            processing_timestamp=datetime.now().isoformat(),
            geographical_context="مكة/المدينة",
            temporal_context="العهد النبوي"
        )
    
    def _extract_quran_references(self, content: str) -> tuple[str, int]:
        """Extract Surah and Ayah references from Quranic text"""
        patterns = [
            r"سورة\s+(\S+)\s+آية\s+(\d+)",
            r"Surah\s+(\S+)\s+Ayah\s+(\d+)",
            r"﴿(\d+)﴾"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                if "﴿" in pattern:
                    return "Unknown", int(match.group(1))
                else:
                    return match.group(1), int(match.group(2))
        
        return "Unknown", 0

class LegalMetadataExtractor(MetadataExtractor):
    """Metadata extractor for legal documents"""
    
    def extract_metadata(self, content: str, filename: str, content_type: ContentType) -> DocumentMetadata:
        law_name, article_number = self._extract_legal_references(content, filename)
        
        return DocumentMetadata(
            source_type=ContentType.LEGAL,
            source_file=filename,
            content_id=f"legal_{law_name}_{article_number}",
            sensitivity_level=SensitivityLevel.MEDIUM,
            language_variant=LanguageVariant.MSA,
            law_name=law_name,
            article_number=article_number,
            content_tags=["قانون", "كويتي", "مادة", law_name],
            sha256_hash=hashlib.sha256(content.encode('utf-8')).hexdigest(),
            processing_timestamp=datetime.now().isoformat(),
            geographical_context="الكويت",
            temporal_context=self._extract_legal_period(filename)
        )
    
    def _extract_legal_references(self, content: str, filename: str) -> tuple[str, str]:
        """Extract legal references from content"""
        # Extract law name from filename or content
        law_name = self._extract_law_name(filename, content)
        
        # Extract article number
        article_match = re.search(r"المادة\s+(\d+)", content)
        article_number = article_match.group(1) if article_match else "0"
        
        return law_name, article_number
    
    def _extract_law_name(self, filename: str, content: str) -> str:
        """Extract law name from filename or content"""
        # Try filename first
        name_patterns = [
            r"قانون\s+(\S+)",
            r"law\s+(\S+)",
            r"(\S+)\s+قانون",
            r"(\S+)\s+law"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Fallback to content
        for pattern in name_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return "قانون_غير_محدد"
    
    def _extract_legal_period(self, filename: str) -> str:
        """Extract legal period from filename"""
        year_match = re.search(r"(\d{4})", filename)
        if year_match:
            year = year_match.group(1)
            return f"سنة {year}"
        return "غير محدد"

class CulturalMetadataExtractor(MetadataExtractor):
    """Metadata extractor for cultural content"""
    
    def extract_metadata(self, content: str, filename: str, content_type: ContentType) -> DocumentMetadata:
        dialect = self._detect_dialect(content)
        
        return DocumentMetadata(
            source_type=content_type,
            source_file=filename,
            content_id=f"cultural_{hashlib.sha256(content.encode()).hexdigest()[:8]}",
            sensitivity_level=SensitivityLevel.LOW,
            language_variant=LanguageVariant.KUWAITI_DIALECT if dialect == "kuwaiti" else LanguageVariant.MSA,
            content_tags=["ثقافة", "تراث", "كويتي", dialect],
            sha256_hash=hashlib.sha256(content.encode('utf-8')).hexdigest(),
            processing_timestamp=datetime.now().isoformat(),
            geographical_context="الكويت",
            temporal_context=self._extract_cultural_period(content)
        )
    
    def _detect_dialect(self, content: str) -> str:
        """Detect Arabic dialect in content"""
        kuwaiti_indicators = ['عسب', 'شلون', 'شكو', 'وايد', 'شنو']
        content_lower = content.lower()
        
        if any(indicator in content_lower for indicator in kuwaiti_indicators):
            return "kuwaiti"
        return "msa"
    
    def _extract_cultural_period(self, content: str) -> str:
        """Extract cultural period from content"""
        period_patterns = [
            r"القرن\s+(\d+)",
            r"في\s+ال(\d+)",
            r"عقد\s+(\d+)"
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, content)
            if match:
                return f"القرن/الفترة: {match.group(1)}"
        
        return "غير محدد"

class MetadataExtractorFactory:
    """Factory for creating appropriate metadata extractors"""
    
    @staticmethod
    def get_extractor(content_type: ContentType) -> MetadataExtractor:
        extractors = {
            ContentType.QURAN: QuranMetadataExtractor(),
            ContentType.LEGAL: LegalMetadataExtractor(),
            ContentType.TAFSIR: CulturalMetadataExtractor(),  # Reuse cultural for now
            ContentType.HADITH: CulturalMetadataExtractor(),
            ContentType.ISLAMIC_HISTORY: CulturalMetadataExtractor(),
            ContentType.HISTORY: CulturalMetadataExtractor(),
            ContentType.CULTURE: CulturalMetadataExtractor(),
            ContentType.KUWAITI_DIALECT: CulturalMetadataExtractor(),
            ContentType.MIXED: CulturalMetadataExtractor()
        }
        
        return extractors.get(content_type, CulturalMetadataExtractor())

# Utility function
def extract_metadata(content: str, filename: str, content_type: ContentType) -> DocumentMetadata:
    """Convenience function for metadata extraction"""
    extractor = MetadataExtractorFactory.get_extractor(content_type)
    return extractor.extract_metadata(content, filename, content_type)
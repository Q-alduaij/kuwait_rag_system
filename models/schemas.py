from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ContentType(str, Enum):
    QURAN = "quran"
    TAFSIR = "tafsir"
    ISLAMIC_HISTORY = "islamic_history"
    HADITH = "hadith"
    HISTORY = "history"
    CULTURE = "culture"
    KUWAITI_DIALECT = "kuwaiti_dialect"
    LEGAL = "legal"
    MIXED = "mixed"

class SensitivityLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LanguageVariant(str, Enum):
    MSA = "msa"
    KUWAITI_DIALECT = "kuwaiti_dialect"
    CLASSICAL_ARABIC = "classical_arabic"

class DocumentMetadata(BaseModel):
    """Enhanced metadata model with proper validation"""
    source_type: ContentType
    source_file: str
    content_id: str = Field(..., description="Unique identifier for the content chunk")
    sensitivity_level: SensitivityLevel
    language_variant: LanguageVariant
    temporal_context: Optional[str] = None
    geographical_context: Optional[str] = None
    linked_references: List[str] = Field(default_factory=list)
    content_tags: List[str] = Field(default_factory=list)
    sha256_hash: str = Field(..., description="SHA256 hash of the content")
    processing_timestamp: str
    
    # Quran-specific fields
    surah_number: Optional[int] = None
    surah_name: Optional[str] = None
    ayah_number: Optional[int] = None
    hafs_numbering: Optional[str] = None
    
    # Legal-specific fields
    law_name: Optional[str] = None
    article_number: Optional[str] = None
    enactment_date: Optional[str] = None
    
    # General fields
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    
    @validator('sha256_hash')
    def validate_sha256_hash(cls, v):
        if len(v) != 64 or not all(c in '0123456789abcdef' for c in v):
            raise ValueError('Invalid SHA256 hash format')
        return v
    
    @validator('processing_timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('Invalid ISO timestamp format')
        return v

class DocumentChunk(BaseModel):
    """Main document chunk model with validated metadata"""
    content: str = Field(..., description="The text content of the chunk")
    metadata: DocumentMetadata
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        if len(v.strip()) < 10:  # Minimum content length
            raise ValueError('Content too short')
        return v.strip()

class ProcessingResult(BaseModel):
    """Result model for document processing operations"""
    success: bool
    chunks: List[DocumentChunk] = Field(default_factory=list)
    error_message: Optional[str] = None
    file_type: Optional[str] = None
    processing_time: float = 0.0
    warnings: List[str] = Field(default_factory=list)

class SearchQuery(BaseModel):
    """Model for search queries"""
    query: str
    content_types: List[ContentType] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    """Model for search results"""
    query: str
    documents: List[DocumentChunk]
    scores: List[float]
    total_results: int
    search_time: float

class QARequest(BaseModel):
    """Model for QA requests"""
    question: str
    context: Optional[str] = None
    query_type: ContentType = ContentType.MIXED
    include_sources: bool = True
    max_context_length: int = Field(default=4000, ge=1000, le=10000)

class QAResponse(BaseModel):
    """Model for QA responses"""
    question: str
    answer: str
    sources: List[DocumentChunk] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    context_used: Optional[str] = None

class SystemHealth(BaseModel):
    """System health status model"""
    status: str
    components: Dict[str, bool]
    timestamp: str
    version: str

# Response models for API
class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchProcessingRequest(BaseModel):
    """Model for batch processing requests"""
    file_paths: List[str]
    content_type: ContentType
    overwrite_existing: bool = False

class BatchProcessingResult(BaseModel):
    """Model for batch processing results"""
    total_files: int
    processed_files: int
    failed_files: int
    total_chunks: int
    processing_time: float
    results: List[ProcessingResult]
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime

from rag.qa_chain import EnhancedQASystem, LegalQASystem, ReligiousQASystem, CulturalQASystem
from rag.vector_store import EnhancedVectorStoreManager, populate_vector_store
from models.schemas import ContentType, SensitivityLevel
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kuwait RAG System API",
    description="Specialized RAG system for Kuwaiti legal, religious, and cultural content",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QA systems
qa_systems = {
    "general": EnhancedQASystem(),
    "legal": LegalQASystem(),
    "religious": ReligiousQASystem(),
    "cultural": CulturalQASystem()
}

vector_store = EnhancedVectorStoreManager()

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str = Field(..., description="السؤال المطلوب الإجابة عليه", example="ما هي قوانين العمل في الكويت؟")
    query_type: str = Field("auto", description="نوع السؤال (legal, religious, historical, cultural, general, auto)")
    max_results: int = Field(5, description="عدد النتائج المراد استرجاعها", ge=1, le=20)
    include_sources: bool = Field(True, description="هل تتضمن النتائج المصادر؟")

class QuestionResponse(BaseModel):
    success: bool
    question: str
    answer: str
    query_type: str
    processing_time: float
    sources: List[Dict[str, Any]] = []
    retrieved_documents: int
    confidence: float
    suggestions: List[str] = []

class SystemStatus(BaseModel):
    status: str
    vector_store_count: int
    models_loaded: bool
    api_version: str
    uptime: float

class ProcessingRequest(BaseModel):
    file_path: str
    content_type: str

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int = 0
    processing_time: float = 0.0

# System startup event
@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    logger.info("🚀 Starting Kuwait RAG System API...")
    
    # Check vector store status
    try:
        info = vector_store.get_collection_info()
        logger.info(f"✅ Vector store ready: {info.get('total_chunks', 0)} chunks")
    except Exception as e:
        logger.warning(f"⚠️ Vector store not ready: {str(e)}")
    
    logger.info("✅ Kuwait RAG System API started successfully")

# Health check endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "مرحبًا بكم في نظام الكويت للذكاء الاصطناعي",
        "version": "2.0.0",
        "system": "Kuwait RAG System",
        "language": "Arabic/English",
        "status": "operational"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Comprehensive system health check"""
    try:
        info = vector_store.get_collection_info()
        chunks_count = info.get('total_chunks', 0)
        
        return SystemStatus(
            status="healthy",
            vector_store_count=chunks_count,
            models_loaded=True,
            api_version="2.0.0",
            uptime=0.0  # You can implement uptime tracking
        )
    except Exception as e:
        return SystemStatus(
            status="degraded",
            vector_store_count=0,
            models_loaded=False,
            api_version="2.0.0",
            uptime=0.0
        )

# Main QA endpoint
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the Kuwait RAG system"""
    start_time = datetime.now()
    
    try:
        # Validate query type
        valid_types = ["auto", "general", "legal", "religious", "historical", "cultural"]
        if request.query_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"نوع السؤال غير صالح. الاختيارات المتاحة: {valid_types}")
        
        # Get appropriate QA system
        qa_system = qa_systems.get(request.query_type, qa_systems["general"])
        
        # Get answer
        result = qa_system.answer_question(
            question=request.question,
            query_type=request.query_type,
            max_results=request.max_results
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response_data = {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "query_type": result["query_type"],
            "processing_time": processing_time,
            "sources": result.get("sources", []) if request.include_sources else [],
            "retrieved_documents": result.get("retrieved_documents", 0),
            "confidence": result.get("confidence_scores", [0])[0] if result.get("confidence_scores") else 0.0,
            "suggestions": result.get("suggestions", [])
        }
        
        return QuestionResponse(**response_data)
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error processing question: {str(e)}")
        
        return QuestionResponse(
            success=False,
            question=request.question,
            answer=f"عذرًا، حدث خطأ أثناء معالجة سؤالك: {str(e)}",
            query_type=request.query_type,
            processing_time=processing_time,
            sources=[],
            retrieved_documents=0,
            confidence=0.0,
            suggestions=["يرجى المحاولة مرة أخرى أو إعادة صياغة السؤال"]
        )

# Specialized endpoints
@app.post("/ask/legal", response_model=QuestionResponse)
async def ask_legal_question(request: QuestionRequest):
    """Ask a legal-specific question"""
    request.query_type = "legal"
    return await ask_question(request)

@app.post("/ask/religious", response_model=QuestionResponse)
async def ask_religious_question(request: QuestionRequest):
    """Ask a religious-specific question"""
    request.query_type = "religious"
    return await ask_question(request)

@app.post("/ask/cultural", response_model=QuestionResponse)
async def ask_cultural_question(request: QuestionRequest):
    """Ask a cultural-specific question"""
    request.query_type = "cultural"
    return await ask_question(request)

# Vector store management endpoints
@app.get("/vector-store/info")
async def get_vector_store_info():
    """Get information about the vector store"""
    try:
        info = vector_store.get_collection_info()
        return {
            "success": True,
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting vector store info: {str(e)}")

@app.post("/vector-store/populate")
async def populate_vector_store_endpoint(background_tasks: BackgroundTasks):
    """Populate vector store from processed chunks"""
    try:
        # Find the latest chunks file
        chunks_files = []
        for file in os.listdir(settings.PROCESSED_DIR):
            if file.startswith("chunks_") and file.endswith(".jsonl"):
                chunks_files.append(file)
        
        if not chunks_files:
            raise HTTPException(status_code=404, detail="No processed chunks found")
        
        latest_file = sorted(chunks_files)[-1]
        chunks_path = os.path.join(settings.PROCESSED_DIR, latest_file)
        
        def populate_task():
            populate_vector_store(chunks_path)
        
        background_tasks.add_task(populate_task)
        
        return {
            "success": True,
            "message": "Vector store population started in background",
            "file": latest_file
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error populating vector store: {str(e)}")

# Search endpoints
@app.get("/search")
async def search_documents(
    query: str = Query(..., description="بحث في الوثائق"),
    content_type: str = Query(None, description="تصفية حسب نوع المحتوى"),
    max_results: int = Query(5, description="عدد النتائج", ge=1, le=20)
):
    """Search documents directly"""
    try:
        filters = None
        if content_type:
            filters = {"source_type": content_type}
        
        results = vector_store.search(
            query=query,
            n_results=max_results,
            filters=filters
        )
        
        return {
            "success": True,
            "query": query,
            "results": results.get("documents", []),
            "metadata": results.get("metadatas", []),
            "scores": results.get("scores", []),
            "total_results": len(results.get("documents", []))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# System management endpoints
@app.get("/system/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        vector_info = vector_store.get_collection_info()
        
        # Count files by type (you would implement this)
        file_stats = {
            "total_files": 0,
            "by_type": {},
            "total_chunks": vector_info.get("total_chunks", 0)
        }
        
        return {
            "success": True,
            "vector_store": vector_info,
            "files": file_stats,
            "models": {
                "embedding": settings.EMBEDDING_MODEL,
                "llm": "local"  # You can enhance this
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "message": "حدث خطأ في المعالجة"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "حدث خطأ غير متوقع في النظام"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Set to False in production
    )
import os
from typing import Dict, Any

class Settings:
    # Paths
    DATA_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    VECTOR_STORE_PATH = "data/vector_store"
    LOGS_DIR = "data/logs"
    
    # Chunking settings - Enhanced for Arabic content
    CHUNK_SIZES = {
        "quran": {"min": 50, "max": 100, "overlap": 10},
        "tafsir": {"min": 300, "max": 500, "overlap": 50},
        "legal": {"min": 400, "max": 600, "overlap": 75},
        "historical": {"min": 600, "max": 900, "overlap": 100},
        "cultural": {"min": 500, "max": 800, "overlap": 75},
        "dialect": {"min": 200, "max": 400, "overlap": 25},
        "mixed": {"min": 400, "max": 700, "overlap": 50}
    }
    
    # Arabic-specific text processing
    ARABIC_PUNCTUATION = "۔.,،;؛!؟?()[]{}«»"
    ARABIC_STOP_WORDS = {"في", "من", "إلى", "على", "أن", "لا", "ما", "هذا", "هذه", "كان"}
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION = 384
    
    # Local LLM settings (update with your model path)
    LOCAL_LLM_PATH = "path/to/your/local/model"
    LOCAL_LLM_PARAMS = {
        "temperature": 0.1,
        "max_tokens": 2000,
        "top_p": 0.9
    }
    
    # API settings (for pre-processing)
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_TIMEOUT = 30
    
    # Processing limits
    MAX_FILE_SIZE_MB = 50
    PROCESSING_TIMEOUT = 300  # 5 minutes per file
    MAX_RETRIES = 3
    
    # Content types with Arabic labels
    CONTENT_TYPES = {
        "quran": "القرآن الكريم",
        "tafsir": "كتب التفسير",
        "islamic_history": "التاريخ الإسلامي",
        "hadith": "الحديث والكتب الإسلامية",
        "history": "كتب التاريخ",
        "culture": "كتب الثقافة",
        "kuwaiti_dialect": "اللهجة الكويتية",
        "legal": "القوانين الكويتية",
        "mixed": "محتوى مختلط"
    }
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf', '.docx', '.doc', '.txt', '.json', '.jsonl', 
        '.html', '.htm', '.epub'
    }

settings = Settings()
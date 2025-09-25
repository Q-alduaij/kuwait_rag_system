from typing import Dict, Any, List
from enum import Enum

class AdvancedRAGConfig:
    """Advanced RAG configuration settings"""
    
    # Hybrid retrieval settings
    HYBRID_RETRIEVAL = {
        "vector_weight": 0.6,
        "keyword_weight": 0.3,
        "temporal_weight": 0.05,
        "authority_weight": 0.05,
        "rerank_enabled": True,
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
    
    # Evaluation settings
    EVALUATION = {
        "metrics": ["faithfulness", "answer_relevance", "context_precision", "context_recall"],
        "thresholds": {
            "faithfulness": 0.8,
            "answer_relevance": 0.7,
            "context_precision": 0.6,
            "context_recall": 0.7
        }
    }
    
    # Query enhancement settings
    QUERY_ENHANCEMENT = {
        "hyde_enabled": True,
        "query_expansion_enabled": True,
        "decomposition_enabled": True,
        "max_expansion_terms": 5
    }
    
    # Advanced chunking settings
    ADAPTIVE_CHUNKING = {
        "quran": {"context_verses": 2, "max_length": 500},
        "legal": {"include_hierarchy": True, "max_length": 600},
        "historical": {"preserve_narrative": True, "max_length": 800},
        "dialogue": {"preserve_speakers": True, "max_length": 400}
    }
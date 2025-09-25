import logging
from typing import List, Dict, Any, Optional, Tuple
from models.schemas import DocumentChunk, SearchQuery, SearchResult
from rag.vector_store import EnhancedVectorStoreManager
from utils.helpers import ArabicTextProcessor

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Advanced retriever combining multiple retrieval strategies"""
    
    def __init__(self):
        self.vector_store = EnhancedVectorStoreManager()
        self.text_processor = ArabicTextProcessor()
        
        # Retrieval weights
        self.weights = {
            "vector_similarity": 0.7,
            "keyword_relevance": 0.2,
            "temporal_relevance": 0.05,
            "source_authority": 0.05
        }
    
    def retrieve(self, query: SearchQuery) -> SearchResult:
        """Perform hybrid retrieval with multiple strategies"""
        import time
        start_time = time.time()
        
        # Normalize and preprocess query
        processed_query = self._preprocess_query(query.query)
        
        # Get vector-based results
        vector_results = self._vector_retrieval(processed_query, query)
        
        # Get keyword-based results (would implement BM25 here)
        keyword_results = self._keyword_retrieval(processed_query, query)
        
        # Combine and re-rank results
        combined_results = self._combine_results(
            vector_results, keyword_results, query.max_results
        )
        
        # Apply filters and thresholds
        filtered_results = self._apply_filters(combined_results, query)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            query=query.query,
            documents=filtered_results["documents"],
            scores=filtered_results["scores"],
            total_results=len(filtered_results["documents"]),
            search_time=search_time
        )
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better retrieval"""
        # Normalize Arabic text
        query = self.text_processor.normalize_arabic_text(query)
        
        # Extract keywords for expansion
        keywords = self.text_processor.extract_arabic_keywords(query)
        
        # Add keywords to query for better recall
        if keywords:
            expanded_query = query + " " + " ".join(keywords[:3])
            return expanded_query
        
        return query
    
    def _vector_retrieval(self, query: str, search_query: SearchQuery) -> Dict[str, Any]:
        """Perform vector similarity search"""
        try:
            results = self.vector_store.search(
                query=query,
                n_results=search_query.max_results * 2,  # Get more for re-ranking
                filters=search_query.filters,
                content_types=[ct.value for ct in search_query.content_types] if search_query.content_types else None
            )
            
            return {
                "documents": self._convert_to_document_chunks(results),
                "scores": results.get("scores", []),
                "method": "vector"
            }
            
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            return {"documents": [], "scores": [], "method": "vector"}
    
    def _keyword_retrieval(self, query: str, search_query: SearchQuery) -> Dict[str, Any]:
        """Perform keyword-based retrieval (simplified version)"""
        # This would typically use BM25 or similar
        # For now, we'll use a simple keyword matching approach
        
        try:
            # Use vector store with keyword-like filtering
            keyword_filters = self._build_keyword_filters(query, search_query.filters)
            
            results = self.vector_store.search(
                query=query,
                n_results=search_query.max_results,
                filters=keyword_filters
            )
            
            # Simple keyword scoring (would be enhanced with proper BM25)
            keyword_scores = self._calculate_keyword_scores(results["documents"], query)
            
            return {
                "documents": self._convert_to_document_chunks(results),
                "scores": keyword_scores,
                "method": "keyword"
            }
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {str(e)}")
            return {"documents": [], "scores": [], "method": "keyword"}
    
    def _build_keyword_filters(self, query: str, base_filters: Dict) -> Dict:
        """Build filters for keyword-based retrieval"""
        keywords = self.text_processor.extract_arabic_keywords(query, max_keywords=5)
        
        filters = base_filters.copy() if base_filters else {}
        
        if keywords:
            # This is a simplified approach - real BM25 would be better
            filters["content_tags"] = {"$overlap": keywords}
        
        return filters
    
    def _calculate_keyword_scores(self, documents: List[str], query: str) -> List[float]:
        """Calculate keyword relevance scores"""
        query_keywords = set(self.text_processor.extract_arabic_keywords(query))
        scores = []
        
        for doc in documents:
            doc_keywords = set(self.text_processor.extract_arabic_keywords(doc))
            overlap = len(query_keywords.intersection(doc_keywords))
            score = overlap / max(len(query_keywords), 1)
            scores.append(score)
        
        return scores
    
    def _combine_results(self, vector_results: Dict, keyword_results: Dict, max_results: int) -> Dict[str, Any]:
        """Combine and re-rank results from different retrieval methods"""
        combined = []
        
        # Add vector results with their weights
        for doc, score in zip(vector_results["documents"], vector_results["scores"]):
            weighted_score = score * self.weights["vector_similarity"]
            combined.append((doc, weighted_score, "vector"))
        
        # Add keyword results with their weights
        for doc, score in zip(keyword_results["documents"], keyword_results["scores"]):
            weighted_score = score * self.weights["keyword_relevance"]
            combined.append((doc, weighted_score, "keyword"))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates based on content hash
        unique_results = self._remove_duplicates(combined)
        
        # Take top results
        top_results = unique_results[:max_results]
        
        return {
            "documents": [item[0] for item in top_results],
            "scores": [item[1] for item in top_results],
            "methods": [item[2] for item in top_results]
        }
    
    def _remove_duplicates(self, results: List[Tuple]) -> List[Tuple]:
        """Remove duplicate documents based on content hash"""
        seen_hashes = set()
        unique_results = []
        
        for doc, score, method in results:
            doc_hash = doc.metadata.sha256_hash
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_results.append((doc, score, method))
        
        return unique_results
    
    def _apply_filters(self, results: Dict[str, Any], query: SearchQuery) -> Dict[str, Any]:
        """Apply similarity threshold and other filters"""
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(results["documents"], results["scores"]):
            if score >= query.similarity_threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        return {
            "documents": filtered_docs,
            "scores": filtered_scores
        }
    
    def _convert_to_document_chunks(self, results: Dict) -> List[DocumentChunk]:
        """Convert raw results to DocumentChunk objects"""
        documents = []
        
        for doc_content, metadata in zip(results.get("documents", []), results.get("metadatas", [])):
            try:
                # Create DocumentChunk from the data
                document = DocumentChunk(
                    content=doc_content,
                    metadata=DocumentMetadata(**metadata)
                )
                documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to convert result to DocumentChunk: {str(e)}")
                continue
        
        return documents

class ContextualRetriever(HybridRetriever):
    """Context-aware retriever for multi-turn conversations"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
    
    def retrieve_with_context(self, query: str, conversation_history: List[Dict]) -> SearchResult:
        """Retrieve with conversation context"""
        # Enhance query with conversation context
        enhanced_query = self._enhance_query_with_context(query, conversation_history)
        
        search_query = SearchQuery(
            query=enhanced_query,
            max_results=10
        )
        
        return self.retrieve(search_query)
    
    def _enhance_query_with_context(self, query: str, history: List[Dict]) -> str:
        """Enhance query with conversation context"""
        if not history:
            return query
        
        # Extract key terms from recent conversation
        recent_queries = [turn.get("question", "") for turn in history[-3:]]  # Last 3 turns
        context_terms = set()
        
        for q in recent_queries:
            terms = self.text_processor.extract_arabic_keywords(q)
            context_terms.update(terms)
        
        # Add context terms to current query
        if context_terms:
            enhanced_query = query + " " + " ".join(list(context_terms)[:5])
            return enhanced_query
        
        return query
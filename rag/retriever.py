import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from models.schemas import DocumentChunk, SearchQuery, SearchResult
from rag.vector_store import EnhancedVectorStoreManager
from utils.helpers import ArabicTextProcessor

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Advanced hybrid retriever combining vector similarity, BM25, and cross-encoder re-ranking
    Implements multiple retrieval strategies for optimal performance :cite[1]:cite[6]
    """
    
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_store = EnhancedVectorStoreManager()
        self.text_processor = ArabicTextProcessor()
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Initialize BM25 corpus
        self.bm25_corpus = []
        self.bm25_doc_map = {}
        self.bm25_initialized = False
        
        # Retrieval weights (optimized for Arabic content)
        self.weights = {
            "vector_similarity": 0.6,
            "keyword_relevance": 0.3,
            "temporal_relevance": 0.05,
            "source_authority": 0.05
        }
        
        # Search type configuration :cite[3]
        self.search_types = {
            "similarity": {"k": 10},
            "mmr": {"k": 10, "fetch_k": 50, "lambda_mult": 0.7},
            "similarity_threshold": {"k": 10, "score_threshold": 0.8}
        }

    def initialize_bm25(self, documents: List[DocumentChunk]):
        """Initialize BM25 with document corpus"""
        try:
            self.bm25_corpus = []
            self.bm25_doc_map = {}
            
            for i, doc in enumerate(documents):
                # Tokenize Arabic text for BM25
                tokens = self._tokenize_arabic(doc.content)
                self.bm25_corpus.append(tokens)
                self.bm25_doc_map[i] = doc
            
            if self.bm25_corpus:
                self.bm25 = BM25Okapi(self.bm25_corpus)
                self.bm25_initialized = True
                logger.info(f"✅ BM25 initialized with {len(documents)} documents")
                
        except Exception as e:
            logger.error(f"❌ BM25 initialization failed: {str(e)}")
            self.bm25_initialized = False

    def retrieve(self, query: SearchQuery) -> SearchResult:
        """
        Perform hybrid retrieval with multiple strategies :cite[1]:cite[6]
        """
        import time
        start_time = time.time()
        
        # Preprocess and enhance query
        processed_query = self._preprocess_query(query.query)
        
        # Parallel retrieval strategies
        vector_results = self._vector_retrieval(processed_query, query)
        keyword_results = self._keyword_retrieval(processed_query, query) if self.bm25_initialized else {"documents": [], "scores": []}
        
        # Combine and re-rank results
        combined_results = self._combine_and_rerank(
            vector_results, keyword_results, processed_query, query.max_results
        )
        
        # Apply final filters and thresholds
        final_results = self._apply_filters(combined_results, query)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            query=query.query,
            documents=final_results["documents"],
            scores=final_results["scores"],
            total_results=len(final_results["documents"]),
            search_time=search_time
        )

    def _vector_retrieval(self, query: str, search_query: SearchQuery) -> Dict[str, Any]:
        """Enhanced vector retrieval with multiple search types :cite[3]"""
        try:
            # Try different search strategies
            all_results = []
            
            for search_type, params in self.search_types.items():
                try:
                    # Merge with query-specific parameters
                    search_params = params.copy()
                    search_params["k"] = search_query.max_results * 2
                    
                    results = self.vector_store.search(
                        query=query,
                        n_results=search_params["k"],
                        filters=search_query.filters,
                        content_types=[ct.value for ct in search_query.content_types] if search_query.content_types else None
                    )
                    
                    if results.get("documents"):
                        # Convert to DocumentChunk objects
                        documents = self._convert_to_document_chunks(results)
                        all_results.extend(list(zip(documents, results.get("scores", []), [search_type] * len(documents))))
                        
                except Exception as e:
                    logger.warning(f"Search type {search_type} failed: {str(e)}")
                    continue
            
            # Sort by score and take top results
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_results = all_results[:search_query.max_results * 2]
            
            if top_results:
                documents, scores, methods = zip(*top_results)
                return {
                    "documents": list(documents),
                    "scores": list(scores),
                    "methods": list(methods)
                }
            else:
                return {"documents": [], "scores": [], "methods": []}
                
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            return {"documents": [], "scores": [], "methods": []}

    def _keyword_retrieval(self, query: str, search_query: SearchQuery) -> Dict[str, Any]:
        """BM25 keyword-based retrieval for Arabic content"""
        try:
            # Tokenize query for BM25
            query_tokens = self._tokenize_arabic(query)
            
            # Get BM25 scores
            doc_scores = self.bm25.get_scores(query_tokens)
            
            # Get top documents
            top_indices = np.argsort(doc_scores)[::-1][:search_query.max_results * 2]
            
            documents = []
            scores = []
            
            for idx in top_indices:
                if doc_scores[idx] > 0:  # Only include relevant documents
                    documents.append(self.bm25_doc_map[idx])
                    scores.append(float(doc_scores[idx]))
            
            return {
                "documents": documents,
                "scores": scores,
                "method": "bm25"
            }
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {str(e)}")
            return {"documents": [], "scores": [], "method": "bm25"}

    def _combine_and_rerank(self, vector_results: Dict, keyword_results: Dict, 
                           query: str, max_results: int) -> Dict[str, Any]:
        """Combine results and re-rank using cross-encoder :cite[1]"""
        try:
            # Combine all results
            all_documents = []
            all_scores = []
            
            # Add vector results with weighting
            for doc, score, method in zip(vector_results.get("documents", []), 
                                         vector_results.get("scores", []), 
                                         vector_results.get("methods", [])):
                weighted_score = score * self.weights["vector_similarity"]
                all_documents.append(doc)
                all_scores.append(weighted_score)
            
            # Add keyword results with weighting
            for doc, score in zip(keyword_results.get("documents", []), 
                                 keyword_results.get("scores", [])):
                weighted_score = score * self.weights["keyword_relevance"]
                all_documents.append(doc)
                all_scores.append(weighted_score)
            
            # Remove duplicates based on content hash
            unique_docs = []
            unique_scores = []
            seen_hashes = set()
            
            for doc, score in zip(all_documents, all_scores):
                doc_hash = doc.metadata.sha256_hash
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    unique_docs.append(doc)
                    unique_scores.append(score)
            
            # Cross-encoder re-ranking for top candidates :cite[1]
            if len(unique_docs) > 5:
                rerank_candidates = unique_docs[:50]  # Limit for performance
                
                # Prepare pairs for cross-encoder
                pairs = [(query, doc.content) for doc in rerank_candidates]
                
                # Get cross-encoder scores
                rerank_scores = self.cross_encoder.predict(pairs)
                
                # Combine scores
                combined_scores = []
                for i, doc in enumerate(rerank_candidates):
                    original_idx = unique_docs.index(doc)
                    combined_score = (0.7 * rerank_scores[i]) + (0.3 * unique_scores[original_idx])
                    combined_scores.append(combined_score)
                
                # Sort by combined scores
                reranked = sorted(zip(rerank_candidates, combined_scores), 
                                key=lambda x: x[1], reverse=True)
                
                # Update results
                unique_docs = [doc for doc, _ in reranked] + unique_docs[len(rerank_candidates):]
                unique_scores = [score for _, score in reranked] + unique_scores[len(rerank_candidates):]
            
            # Take top results
            final_results = list(zip(unique_docs, unique_scores))
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_results = final_results[:max_results]
            
            if final_results:
                documents, scores = zip(*final_results)
                return {
                    "documents": list(documents),
                    "scores": list(scores)
                }
            else:
                return {"documents": [], "scores": []}
                
        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            # Fallback to simple combination
            combined = list(zip(all_documents, all_scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            combined = combined[:max_results]
            
            if combined:
                documents, scores = zip(*combined)
                return {
                    "documents": list(documents),
                    "scores": list(scores)
                }
            else:
                return {"documents": [], "scores": []}

    def _tokenize_arabic(self, text: str) -> List[str]:
        """Tokenize Arabic text for BM25"""
        # Simple Arabic-aware tokenization
        tokens = re.findall(r'[\u0600-\u06FF]+|[A-Za-z0-9]+', text)
        return [token.lower() for token in tokens if token.strip()]

    def _preprocess_query(self, query: str) -> str:
        """Enhanced Arabic query preprocessing"""
        # Normalize Arabic text
        query = self.text_processor.normalize_arabic_text(query)
        
        # Query expansion with Arabic synonyms
        expanded_terms = self._expand_arabic_query(query)
        if expanded_terms:
            query += " " + " ".join(expanded_terms)
        
        return query

    def _expand_arabic_query(self, query: str) -> List[str]:
        """Expand query with Arabic synonyms and related terms"""
        # Basic Arabic query expansion
        expansion_terms = []
        
        # Add common Arabic synonyms (this would be enhanced with a proper dictionary)
        synonym_map = {
            "قانون": ["تشريع", "نظام", "مادة"],
            "قرآن": ["قران", "مصحف", "كتاب الله"],
            "حديث": ["سنة", "رواية", "أثر"]
        }
        
        for term, synonyms in synonym_map.items():
            if term in query:
                expansion_terms.extend(synonyms)
        
        return expansion_terms[:3]  # Limit expansion terms

    def _apply_filters(self, results: Dict[str, Any], query: SearchQuery) -> Dict[str, Any]:
        """Apply similarity threshold and content type filters"""
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(results["documents"], results["scores"]):
            # Apply similarity threshold
            if score >= query.similarity_threshold:
                # Apply content type filters
                if query.content_types:
                    if doc.metadata.source_type in query.content_types:
                        filtered_docs.append(doc)
                        filtered_scores.append(score)
                else:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)
        
        return {
            "documents": filtered_docs,
            "scores": filtered_scores
        }

    def _convert_to_document_chunks(self, results: Dict) -> List[DocumentChunk]:
        """Convert raw results to DocumentChunk objects"""
        documents = []
        
        for doc_content, metadata in zip(results.get("documents", []), 
                                        results.get("metadatas", [])):
            try:
                document = DocumentChunk(
                    content=doc_content,
                    metadata=DocumentMetadata(**metadata)
                )
                documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to convert result to DocumentChunk: {str(e)}")
                continue
        
        return documents

class EnsembleRetriever:
    """
    Ensemble retriever combining multiple retrieval strategies :cite[6]
    """
    
    def __init__(self, retrievers: List[HybridRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights if weights else [1.0/len(retrievers)] * len(retrievers)
        
    def retrieve(self, query: SearchQuery) -> SearchResult:
        """Retrieve using ensemble of retrievers"""
        all_results = []
        
        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                result = retriever.retrieve(query)
                # Apply weights to scores
                weighted_docs = [(doc, score * weight) for doc, score 
                               in zip(result.documents, result.scores)]
                all_results.extend(weighted_docs)
            except Exception as e:
                logger.error(f"Ensemble retriever failed: {str(e)}")
                continue
        
        # Sort by weighted scores
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates
        unique_results = []
        seen_hashes = set()
        
        for doc, score in all_results:
            if doc.metadata.sha256_hash not in seen_hashes:
                seen_hashes.add(doc.metadata.sha256_hash)
                unique_results.append((doc, score))
        
        # Take top results
        final_results = unique_results[:query.max_results]
        
        if final_results:
            documents, scores = zip(*final_results)
            return SearchResult(
                query=query.query,
                documents=list(documents),
                scores=list(scores),
                total_results=len(documents),
                search_time=0.0  # Would need to be calculated
            )
        else:
            return SearchResult(
                query=query.query,
                documents=[],
                scores=[],
                total_results=0,
                search_time=0.0
            )
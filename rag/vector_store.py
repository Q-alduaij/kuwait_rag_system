import chromadb
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)

class EnhancedVectorStoreManager:
    def __init__(self, collection_name: str = "kuwait_rag"):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=settings.VECTOR_STORE_PATH)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        logger.info(f"✅ Vector store initialized: {settings.VECTOR_STORE_PATH}")
        logger.info(f"✅ Using embedding model: {settings.EMBEDDING_MODEL}")

    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"📁 Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Kuwait RAG System - Arabic Documents"}
            )
            logger.info(f"📁 Created new collection: {self.collection_name}")
            return collection

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """Add chunks to vector store in batches"""
        if not chunks:
            logger.warning("⚠️ No chunks to add to vector store")
            return
        
        total_chunks = len(chunks)
        logger.info(f"📤 Adding {total_chunks} chunks to vector store...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            self._add_batch(batch, i, total_chunks)
        
        logger.info("✅ All chunks added to vector store")

    def _add_batch(self, batch: List[Dict[str, Any]], start_idx: int, total_chunks: int):
        """Add a single batch of chunks"""
        documents = []
        metadatas = []
        ids = []
        
        for j, chunk in enumerate(batch):
            chunk_id = f"chunk_{start_idx + j}_{chunk['metadata']['sha256_hash'][:8]}"
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk_id)
        
        try:
            # Generate embeddings for the batch
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✅ Added batch {start_idx//len(batch) + 1}: {len(batch)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Failed to add batch starting at index {start_idx}: {str(e)}")
            # Try adding chunks individually
            self._add_chunks_individually(batch, start_idx)

    def _add_chunks_individually(self, chunks: List[Dict[str, Any]], start_idx: int):
        """Add chunks individually if batch fails"""
        for j, chunk in enumerate(chunks):
            try:
                chunk_id = f"chunk_{start_idx + j}_{chunk['metadata']['sha256_hash'][:8]}"
                embedding = self.embedding_model.encode([chunk["content"]]).tolist()
                
                self.collection.add(
                    embeddings=embedding,
                    documents=[chunk["content"]],
                    metadatas=[chunk["metadata"]],
                    ids=[chunk_id]
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to add individual chunk {start_idx + j}: {str(e)}")

    def search(self, query: str, n_results: int = 5, filters: Dict = None, 
               content_types: List[str] = None) -> Dict[str, Any]:
        """Enhanced search with filtering and content type support"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Build where clause for filtering
            where_clause = self._build_where_clause(filters, content_types)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            # Format results
            formatted_results = self._format_search_results(results, query)
            
            logger.info(f"🔍 Search completed: {len(formatted_results['documents'])} results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": [], "error": str(e)}

    def _build_where_clause(self, filters: Dict, content_types: List[str]) -> Dict:
        """Build where clause for ChromaDB query"""
        where_clause = {}
        
        if filters:
            where_clause.update(filters)
        
        if content_types:
            where_clause["source_type"] = {"$in": content_types}
        
        return where_clause if where_clause else None

    def _format_search_results(self, results: Dict, query: str) -> Dict[str, Any]:
        """Format search results for easier consumption"""
        if not results["documents"]:
            return {"documents": [], "metadatas": [], "distances": [], "query": query}
        
        formatted = {
            "query": query,
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "total_results": len(results["documents"][0])
        }
        
        # Add relevance scores (convert distances to similarities)
        formatted["scores"] = [1 - distance for distance in results["distances"][0]]
        
        return formatted

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "embedding_model": settings.EMBEDDING_MODEL,
                "storage_path": settings.VECTOR_STORE_PATH
            }
        except Exception as e:
            logger.error(f"❌ Failed to get collection info: {str(e)}")
            return {"error": str(e)}

    def delete_collection(self):
        """Delete the current collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.warning(f"🗑️ Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to delete collection: {str(e)}")

# Utility function to populate vector store after processing
def populate_vector_store(chunks_file: str):
    """Convenience function to populate vector store from chunks file"""
    import json
    
    # Load chunks
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Initialize vector store
    vector_store = EnhancedVectorStoreManager()
    
    # Add chunks
    vector_store.add_chunks(chunks)
    
    # Print collection info
    info = vector_store.get_collection_info()
    print("📊 Vector Store Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return vector_store
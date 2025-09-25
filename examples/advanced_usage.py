from rag.retriever import HybridRetriever
from rag.evaluation import RagasEvaluator
from rag.advanced_techniques import QueryEnhancementEngine

# Initialize components
hybrid_retriever = HybridRetriever()
evaluator = RagasEvaluator()
query_enhancer = QueryEnhancementEngine()

# Example usage
def advanced_rag_pipeline(question: str, content_type: str):
    # Enhance query
    enhanced_query = query_enhancer.enhance_query(question, technique="hyde")
    
    # Perform hybrid retrieval
    search_query = SearchQuery(
        query=enhanced_query,
        content_types=[content_type],
        max_results=5
    )
    
    results = hybrid_retriever.retrieve(search_query)
    
    # Evaluate results
    evaluation_results = evaluator.evaluate_qa_pair(
        question=question,
        answer=results.documents[0].content if results.documents else "",
        contexts=[doc.content for doc in results.documents]
    )
    
    return results, evaluation_results

# Kuwait-specific example
kuwait_question = "ما هي المواد القانونية المتعلقة بالتعليم في الكويت؟"
results, evaluation = advanced_rag_pipeline(kuwait_question, "legal")
print(f"Retrieved {len(results.documents)} documents")
print(f"Faithfulness score: {evaluation.get('faithfulness', 0):.2f}")
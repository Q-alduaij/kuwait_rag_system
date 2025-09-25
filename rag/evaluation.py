import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset
from models.schemas import QAResponse, DocumentChunk

logger = logging.getLogger(__name__)

class RagasEvaluator:
    """
    Comprehensive RAG evaluation using RAGAS metrics :cite[2]
    """
    
    def __init__(self):
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevance': answer_relevance,
            'context_recall': context_recall,
            'context_precision': context_precision,
            'answer_correctness': answer_correctness,
            'answer_similarity': answer_similarity
        }
    
    def evaluate_qa_pair(self, question: str, answer: str, contexts: List[str], 
                        reference_answer: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a single QA pair using RAGAS metrics
        """
        try:
            # Prepare dataset for RAGAS
            data = {
                'question': [question],
                'answer': [answer],
                'contexts': [contexts]
            }
            
            if reference_answer:
                data['ground_truth'] = [reference_answer]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on availability of reference answer
            if reference_answer:
                metrics_to_use = [faithfulness, answer_relevance, context_recall, 
                                context_precision, answer_correctness]
            else:
                metrics_to_use = [faithfulness, answer_relevance, context_recall, context_precision]
            
            # Run evaluation
            results = evaluate(dataset, metrics=metrics_to_use)
            results_dict = {metric: score for metric, score in zip(results.keys(), results.values())}
            
            logger.info(f"✅ Evaluation completed for question: {question[:50]}...")
            return results_dict
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {str(e)}")
            return {}

    def evaluate_batch(self, qa_pairs: List[Dict]) -> pd.DataFrame:
        """
        Evaluate multiple QA pairs and return comprehensive results
        """
        all_results = []
        
        for i, qa_pair in enumerate(qa_pairs):
            try:
                results = self.evaluate_qa_pair(
                    question=qa_pair['question'],
                    answer=qa_pair['answer'],
                    contexts=qa_pair.get('contexts', []),
                    reference_answer=qa_pair.get('reference_answer')
                )
                
                results['question'] = qa_pair['question']
                results['timestamp'] = datetime.now()
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Failed to evaluate QA pair {i}: {str(e)}")
                continue
        
        return pd.DataFrame(all_results)

    def generate_evaluation_report(self, evaluation_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        """
        if evaluation_df.empty:
            return {}
        
        report = {
            'summary': {
                'total_questions': len(evaluation_df),
                'evaluation_date': datetime.now().isoformat(),
                'average_scores': {}
            },
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # Calculate average scores for each metric
        metric_columns = [col for col in evaluation_df.columns if col not in ['question', 'timestamp']]
        
        for metric in metric_columns:
            if metric in evaluation_df.columns:
                report['summary']['average_scores'][metric] = {
                    'mean': evaluation_df[metric].mean(),
                    'median': evaluation_df[metric].median(),
                    'std': evaluation_df[metric].std(),
                    'min': evaluation_df[metric].min(),
                    'max': evaluation_df[metric].max()
                }
        
        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(report['summary']['average_scores'])
        report['recommendations'] = recommendations
        
        return report

    def _generate_recommendations(self, scores: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on evaluation scores"""
        recommendations = []
        
        for metric, stats in scores.items():
            mean_score = stats['mean']
            
            if metric == 'faithfulness' and mean_score < 0.8:
                recommendations.append("Improve faithfulness by ensuring answers are grounded in retrieved contexts")
            elif metric == 'answer_relevance' and mean_score < 0.7:
                recommendations.append("Enhance answer relevance by improving retrieval quality")
            elif metric == 'context_precision' and mean_score < 0.6:
                recommendations.append("Boost context precision through better chunking and retrieval strategies")
            elif metric == 'context_recall' and mean_score < 0.7:
                recommendations.append("Increase context recall by expanding retrieval scope or improving query understanding")
        
        return recommendations

class KuwaitSpecificEvaluator:
    """
    Domain-specific evaluator for Kuwaiti content
    """
    
    def evaluate_arabic_quality(self, answer: str, question: str) -> Dict[str, float]:
        """Evaluate Arabic language quality and cultural appropriateness"""
        scores = {
            'arabic_fluency': self._assess_arabic_fluency(answer),
            'cultural_relevance': self._assess_cultural_relevance(answer, question),
            'domain_accuracy': self._assess_domain_accuracy(answer, question)
        }
        
        return scores
    
    def _assess_arabic_fluency(self, text: str) -> float:
        """Assess Arabic language fluency"""
        # Basic Arabic language assessment
        arabic_char_ratio = sum(1 for char in text if '\u0600' <= char <= '\u06FF') / len(text) if text else 0
        sentence_coherence = self._check_sentence_structure(text)
        
        return (arabic_char_ratio + sentence_coherence) / 2
    
    def _assess_cultural_relevance(self, answer: str, question: str) -> float:
        """Assess cultural relevance for Kuwaiti context"""
        kuwaiti_indicators = ['الكويت', 'كويتي', 'دولة الكويت', 'الخليج', 'العربية']
        score = 0.0
        
        for indicator in kuwaiti_indicators:
            if indicator in answer:
                score += 0.2
        
        return min(score, 1.0)
    
    def _assess_domain_accuracy(self, answer: str, question: str) -> float:
        """Assess domain accuracy for legal/religious content"""
        # This would be enhanced with domain-specific validation
        accuracy_indicators = ['المادة', 'القانون', 'الآية', 'السورة', 'التفسير']
        score = 0.0
        
        for indicator in accuracy_indicators:
            if indicator in answer and indicator in question:
                score += 0.3
        
        return min(score, 1.0)
    
    def _check_sentence_structure(self, text: str) -> float:
        """Check Arabic sentence structure quality"""
        sentences = text.split('۔')  # Arabic period
        if len(sentences) < 2:
            return 0.5
        
        # Basic structure assessment
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        
        if 5 <= avg_sentence_length <= 20:
            return 0.8
        elif 3 <= avg_sentence_length <= 25:
            return 0.6
        else:
            return 0.4
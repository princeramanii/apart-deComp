# Create the confidence scorer implementation
confidence_scorer_code = '''"""
Multi-factor confidence scoring system for task decomposition.
Implements the 7-factor confidence assessment framework.
"""

import numpy as np
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ConfidenceAnalysis:
    confidence: float
    breakdown: Dict[str, float]
    suggestions: List[str]
    uncertainty: float


class ConfidenceScorer:
    """
    Multi-factor confidence scoring system combining:
    - Description Clarity (20%)
    - Complexity Penalty (15%) 
    - Historical Success (15%)
    - Causal Trace Strength (20%)
    - Attention Coherence (10%)
    - Factual Consistency (15%)
    - Uncertainty Estimation (5%)
    """
    
    def __init__(self, rome_tracer=None, history_db=None):
        self.rome = rome_tracer
        self.history = history_db
        
        # Confidence factor weights
        self.weights = {
            'clarity': 0.20,
            'complexity': 0.15,
            'history': 0.15,
            'rome_strength': 0.20,
            'attention': 0.10,
            'factual': 0.15,
            'uncertainty': 0.05
        }
        
        self.max_depth = 8  # Maximum decomposition depth
        
    def calculate(self, task_prompt: str, context: str = "", depth: int = 0) -> ConfidenceAnalysis:
        """Calculate multi-factor confidence score for a task."""
        
        factors = {
            'clarity': self._clarity_analysis(task_prompt),
            'complexity': 1 - (depth / (self.max_depth + 0.1)),
            'history': self._historical_success(task_prompt),
            'rome_strength': self._rome_analysis(context),
            'attention': self._attention_coherence(task_prompt),
            'factual': self._factual_consistency(task_prompt, context),
            'uncertainty': self._uncertainty_estimation(task_prompt)
        }
        
        # Calculate weighted confidence
        confidence = sum(self.weights[key] * factors[key] for key in factors.keys())
        
        # Generate suggestions
        suggestions = self._generate_suggestions(factors)
        
        return ConfidenceAnalysis(
            confidence=confidence,
            breakdown=factors,
            suggestions=suggestions,
            uncertainty=factors['uncertainty']
        )
    
    def _clarity_analysis(self, text: str) -> float:
        """Analyze description clarity using NLP metrics."""
        if not text:
            return 0.0
            
        # Keyword density analysis
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Simple keyword patterns for task clarity
        task_keywords = ['analyze', 'create', 'design', 'implement', 'evaluate', 
                        'research', 'develop', 'compare', 'optimize', 'solve']
        
        keyword_matches = sum(1 for word in words if word.lower() in task_keywords)
        keyword_density = min(keyword_matches / word_count * 100, 3.0) / 3.0
        
        # Structural score (length and complexity)
        optimal_length = 50  # Optimal word count
        length_score = 1.0 - abs(word_count - optimal_length) / optimal_length
        length_score = max(0.0, min(1.0, length_score))
        
        return (keyword_density + length_score) / 2
    
    def _historical_success(self, task_prompt: str) -> float:
        """Get historical success rate for similar tasks."""
        if not self.history:
            return 0.7  # Default moderate confidence
        
        # Simple similarity-based lookup (in practice, use embeddings)
        task_type = self._classify_task_type(task_prompt)
        return self.history.get(task_type, 0.7)
    
    def _rome_analysis(self, context: str) -> float:
        """Analyze causal trace strength using ROME-style analysis."""
        if not self.rome or not context:
            return 0.6  # Default moderate strength
        
        # Simulate ROME causal tracing analysis
        # In practice, this would use actual model activations
        factual_indicators = ['fact', 'data', 'evidence', 'research', 'study']
        context_words = context.lower().split()
        
        factual_density = sum(1 for word in context_words if word in factual_indicators)
        return min(factual_density / len(context_words) * 10, 1.0) if context_words else 0.5
    
    def _attention_coherence(self, task_prompt: str) -> float:
        """Measure attention pattern coherence."""
        # Simplified coherence based on sentence structure
        sentences = task_prompt.split('.')
        if len(sentences) <= 1:
            return 0.8
        
        # Check for consistent length and structure
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
            
        variance = np.var(lengths) if len(lengths) > 1 else 0
        coherence = 1.0 - min(variance / 100, 1.0)  # Normalize variance
        
        return coherence
    
    def _factual_consistency(self, task_prompt: str, context: str) -> float:
        """Assess factual consistency between prompt and context."""
        if not context:
            return 0.7  # Default when no context
        
        # Simple overlap analysis (in practice, use BERTScore + ROUGE-L)
        prompt_words = set(task_prompt.lower().split())
        context_words = set(context.lower().split())
        
        if not prompt_words:
            return 0.5
        
        overlap = len(prompt_words & context_words) / len(prompt_words)
        return min(overlap * 2, 1.0)  # Scale up overlap score
    
    def _uncertainty_estimation(self, task_prompt: str) -> float:
        """Estimate uncertainty using task complexity indicators."""
        uncertainty_markers = ['maybe', 'possibly', 'might', 'could', 'uncertain']
        words = task_prompt.lower().split()
        
        if not words:
            return 0.5
        
        uncertainty_count = sum(1 for word in words if word in uncertainty_markers)
        # Higher uncertainty markers = lower confidence
        return max(0.1, 1.0 - (uncertainty_count / len(words) * 10))
    
    def _classify_task_type(self, task_prompt: str) -> str:
        """Classify task type for historical lookup."""
        task_prompt_lower = task_prompt.lower()
        
        if any(word in task_prompt_lower for word in ['analyze', 'analysis']):
            return 'analysis'
        elif any(word in task_prompt_lower for word in ['create', 'generate', 'design']):
            return 'creative'
        elif any(word in task_prompt_lower for word in ['research', 'investigate']):
            return 'research'
        elif any(word in task_prompt_lower for word in ['solve', 'calculate', 'compute']):
            return 'computation'
        else:
            return 'general'
    
    def _generate_suggestions(self, factors: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on factor scores."""
        suggestions = []
        
        if factors['clarity'] < 0.6:
            suggestions.append("Consider adding more specific task keywords")
        
        if factors['complexity'] < 0.5:
            suggestions.append("Task may be too complex - consider breaking down further")
        
        if factors['rome_strength'] < 0.5:
            suggestions.append("Add more factual context to improve grounding")
        
        if factors['attention'] < 0.6:
            suggestions.append("Improve sentence structure for better coherence")
        
        return suggestions


# Example usage and testing
if __name__ == "__main__":
    scorer = ConfidenceScorer()
    
    # Test cases
    test_tasks = [
        "Analyze the impact of climate change on agricultural productivity",
        "Create a marketing plan for a new product",
        "Maybe solve this complex optimization problem",
        "Design"  # Very short task
    ]
    
    for task in test_tasks:
        analysis = scorer.calculate(task)
        print(f"\\nTask: {task}")
        print(f"Confidence: {analysis.confidence:.3f}")
        print(f"Breakdown: {analysis.breakdown}")
        if analysis.suggestions:
            print(f"Suggestions: {analysis.suggestions}")
'''

# Write the confidence scorer file
with open('decomp_router/confidence/scorer.py', 'w') as f:
    f.write(confidence_scorer_code)

print("✅ Created confidence scorer implementation")
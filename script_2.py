# Create the model router implementation
model_router_code = '''"""
Dynamic model routing system for intelligent task assignment.
Routes tasks to optimal models based on confidence scores and capabilities.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ModelCapability(Enum):
    """Model capability categories."""
    HIGH_REASONING = "high_reasoning"
    BALANCED = "balanced" 
    COST_EFFECTIVE = "cost_effective"
    SPECIALIZED = "specialized"


@dataclass
class ModelProfile:
    """Profile containing model capabilities and constraints."""
    name: str
    cost_per_call: float
    complexity_handling: float  # 0-1 scale
    reasoning_capability: float  # 0-1 scale
    safety_score: float  # 0-1 scale
    processing_speed: float  # 0-1 scale
    factual_accuracy: float  # 0-1 scale
    capability_type: ModelCapability


@dataclass
class RoutingDecision:
    """Result of routing decision with explanation."""
    selected_model: str
    confidence_threshold: float
    cost_estimate: float
    reasoning: str
    alternatives: List[str]
    safety_check_passed: bool


class ModelRouter:
    """
    Intelligent model router that selects optimal models based on:
    - Task confidence scores
    - Model capability profiles
    - Cost constraints
    - Safety requirements
    """
    
    def __init__(self, martian_client=None):
        self.client = martian_client
        self.model_profiles = self._initialize_model_profiles()
        
        # Routing thresholds
        self.confidence_thresholds = {
            'high': 0.85,
            'medium': 0.60,
            'low': 0.0
        }
        
        # Safety requirements
        self.min_safety_score = 0.8
        self.max_cost_per_task = 0.20
        
    def _initialize_model_profiles(self) -> Dict[str, ModelProfile]:
        """Initialize model capability profiles."""
        return {
            'gpt-4o': ModelProfile(
                name='gpt-4o',
                cost_per_call=0.03,
                complexity_handling=0.95,
                reasoning_capability=0.95,
                safety_score=0.98,
                processing_speed=0.70,
                factual_accuracy=0.92,
                capability_type=ModelCapability.HIGH_REASONING
            ),
            'claude-3.5-sonnet': ModelProfile(
                name='claude-3.5-sonnet',
                cost_per_call=0.015,
                complexity_handling=0.93,
                reasoning_capability=0.92,
                safety_score=0.95,
                processing_speed=0.75,
                factual_accuracy=0.90,
                capability_type=ModelCapability.BALANCED
            ),
            'gpt-4o-mini': ModelProfile(
                name='gpt-4o-mini',
                cost_per_call=0.005,
                complexity_handling=0.75,
                reasoning_capability=0.80,
                safety_score=0.90,
                processing_speed=0.90,
                factual_accuracy=0.85,
                capability_type=ModelCapability.COST_EFFECTIVE
            ),
            'gemini-1.5-pro': ModelProfile(
                name='gemini-1.5-pro',
                cost_per_call=0.012,
                complexity_handling=0.88,
                reasoning_capability=0.85,
                safety_score=0.88,
                processing_speed=0.95,
                factual_accuracy=0.87,
                capability_type=ModelCapability.BALANCED
            )
        }
    
    def route_task(self, task_prompt: str, confidence_score: float, 
                   context: str = "", constraints: Dict = None) -> RoutingDecision:
        """
        Route a task to the optimal model based on confidence and constraints.
        
        Args:
            task_prompt: The task to be executed
            confidence_score: Confidence score from ConfidenceScorer
            context: Additional context for the task
            constraints: Optional routing constraints (cost, speed, etc.)
            
        Returns:
            RoutingDecision with selected model and reasoning
        """
        constraints = constraints or {}
        
        # Filter models based on constraints
        eligible_models = self._filter_eligible_models(constraints)
        
        # Score models based on task requirements
        model_scores = self._score_models(
            confidence_score, task_prompt, eligible_models, constraints
        )
        
        # Select best model
        best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            best_model, confidence_score, model_scores
        )
        
        # Get alternatives
        alternatives = sorted(
            [m for m in model_scores.keys() if m != best_model],
            key=lambda k: model_scores[k], reverse=True
        )[:2]
        
        return RoutingDecision(
            selected_model=best_model,
            confidence_threshold=confidence_score,
            cost_estimate=self.model_profiles[best_model].cost_per_call,
            reasoning=reasoning,
            alternatives=alternatives,
            safety_check_passed=self._safety_check(best_model, task_prompt)
        )
    
    def _filter_eligible_models(self, constraints: Dict) -> List[str]:
        """Filter models based on hard constraints."""
        eligible = []
        
        for model_name, profile in self.model_profiles.items():
            # Cost constraint
            if constraints.get('max_cost', self.max_cost_per_task) < profile.cost_per_call:
                continue
            
            # Safety constraint  
            if constraints.get('min_safety', self.min_safety_score) > profile.safety_score:
                continue
            
            # Speed constraint
            if constraints.get('min_speed', 0) > profile.processing_speed:
                continue
            
            eligible.append(model_name)
        
        return eligible if eligible else list(self.model_profiles.keys())
    
    def _score_models(self, confidence_score: float, task_prompt: str, 
                     eligible_models: List[str], constraints: Dict) -> Dict[str, float]:
        """Score eligible models based on task requirements."""
        scores = {}
        
        # Determine task complexity
        task_complexity = self._estimate_task_complexity(task_prompt)
        
        for model_name in eligible_models:
            profile = self.model_profiles[model_name]
            score = 0.0
            
            # Confidence-based weighting
            if confidence_score > self.confidence_thresholds['high']:
                # High confidence - prioritize cost and speed
                score += 0.4 * (1 - profile.cost_per_call / 0.05)  # Normalize cost
                score += 0.3 * profile.processing_speed
                score += 0.2 * profile.reasoning_capability
                score += 0.1 * profile.safety_score
                
            elif confidence_score > self.confidence_thresholds['medium']:
                # Medium confidence - balanced approach
                score += 0.3 * profile.reasoning_capability
                score += 0.3 * profile.complexity_handling
                score += 0.2 * (1 - profile.cost_per_call / 0.05)
                score += 0.2 * profile.safety_score
                
            else:
                # Low confidence - prioritize capability and safety
                score += 0.4 * profile.reasoning_capability
                score += 0.3 * profile.complexity_handling
                score += 0.2 * profile.safety_score
                score += 0.1 * profile.factual_accuracy
            
            # Task complexity adjustment
            if task_complexity > 0.7:
                score *= (0.5 + 0.5 * profile.complexity_handling)
            
            scores[model_name] = score
        
        return scores
    
    def _estimate_task_complexity(self, task_prompt: str) -> float:
        """Estimate task complexity from prompt analysis."""
        complexity_indicators = [
            'analyze', 'evaluate', 'compare', 'optimize', 'design',
            'multi-step', 'complex', 'comprehensive', 'detailed'
        ]
        
        words = task_prompt.lower().split()
        if not words:
            return 0.5
        
        complexity_count = sum(1 for word in words if word in complexity_indicators)
        
        # Length-based complexity
        length_complexity = min(len(words) / 100, 1.0)
        
        # Keyword-based complexity
        keyword_complexity = min(complexity_count / len(words) * 10, 1.0)
        
        return (length_complexity + keyword_complexity) / 2
    
    def _generate_routing_reasoning(self, selected_model: str, 
                                  confidence_score: float, 
                                  model_scores: Dict[str, float]) -> str:
        """Generate human-readable reasoning for routing decision."""
        profile = self.model_profiles[selected_model]
        
        reasoning_parts = []
        
        # Confidence-based reasoning
        if confidence_score > self.confidence_thresholds['high']:
            reasoning_parts.append(f"High confidence ({confidence_score:.2f}) allows cost-effective routing")
        elif confidence_score > self.confidence_thresholds['medium']:
            reasoning_parts.append(f"Medium confidence ({confidence_score:.2f}) requires balanced approach")
        else:
            reasoning_parts.append(f"Low confidence ({confidence_score:.2f}) prioritizes capability")
        
        # Model selection reasoning
        reasoning_parts.append(
            f"Selected {selected_model} for "
            f"{profile.capability_type.value.replace('_', ' ')} "
            f"(score: {model_scores[selected_model]:.3f})"
        )
        
        # Cost and performance
        reasoning_parts.append(
            f"Cost: ${profile.cost_per_call:.3f}, "
            f"Reasoning: {profile.reasoning_capability:.2f}, "
            f"Safety: {profile.safety_score:.2f}"
        )
        
        return ". ".join(reasoning_parts)
    
    def _safety_check(self, model_name: str, task_prompt: str) -> bool:
        """Perform safety validation for model-task combination."""
        profile = self.model_profiles[model_name]
        
        # Basic safety score check
        if profile.safety_score < self.min_safety_score:
            return False
        
        # Content safety check (simplified)
        unsafe_patterns = ['hack', 'exploit', 'illegal', 'harmful']
        task_lower = task_prompt.lower()
        
        if any(pattern in task_lower for pattern in unsafe_patterns):
            return False
        
        return True
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and model performance metrics."""
        stats = {
            'total_models': len(self.model_profiles),
            'avg_cost': np.mean([p.cost_per_call for p in self.model_profiles.values()]),
            'avg_safety': np.mean([p.safety_score for p in self.model_profiles.values()]),
            'model_distribution': {
                cap_type.value: len([p for p in self.model_profiles.values() 
                                   if p.capability_type == cap_type])
                for cap_type in ModelCapability
            }
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    router = ModelRouter()
    
    # Test routing decisions
    test_cases = [
        ("Simple data analysis task", 0.9),
        ("Complex multi-step reasoning problem", 0.4),
        ("Generate creative content", 0.7),
        ("Analyze financial risks", 0.3)
    ]
    
    print("Model Router Test Results:")
    print("=" * 50)
    
    for task, confidence in test_cases:
        decision = router.route_task(task, confidence)
        print(f"\\nTask: {task}")
        print(f"Confidence: {confidence}")
        print(f"Selected: {decision.selected_model}")
        print(f"Cost: ${decision.cost_estimate:.3f}")
        print(f"Reasoning: {decision.reasoning}")
        print(f"Alternatives: {decision.alternatives}")
    
    # Show model statistics
    print("\\n" + "=" * 50)
    print("Model Statistics:")
    stats = router.get_model_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
'''

# Write the model router file
with open('decomp_router/routing/router.py', 'w') as f:
    f.write(model_router_code)

print("✅ Created model router implementation")
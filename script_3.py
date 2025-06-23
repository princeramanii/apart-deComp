# Create the main decomposition controller
decomp_controller_code = '''"""
Main decomposition controller that orchestrates the enhanced ADaPT framework.
Integrates confidence scoring, model routing, and safety monitoring.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import json
from enum import Enum

from ..confidence.scorer import ConfidenceScorer, ConfidenceAnalysis
from ..routing.router import ModelRouter, RoutingDecision
from ..safety.monitor import SafetyMonitor
from ..interpretability.rome_tracer import ROMETracer


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DECOMPOSED = "decomposed"


@dataclass
class SubTask:
    """Represents a decomposed subtask."""
    id: str
    prompt: str
    context: str = ""
    depth: int = 0
    parent_id: Optional[str] = None
    confidence: Optional[float] = None
    assigned_model: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    cost: float = 0.0
    execution_time: float = 0.0
    safety_passed: bool = True
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    success: bool
    total_cost: float
    execution_time: float
    tasks_completed: int
    tasks_failed: int
    confidence_avg: float
    safety_violations: int
    decomposition_tree: Dict[str, Any]
    performance_metrics: Dict[str, float]


class EnhancedADaPTController:
    """
    Enhanced ADaPT controller with mechanistic interpretability.
    
    Orchestrates task decomposition using:
    - Confidence-based scoring
    - Dynamic model routing  
    - Safety monitoring
    - ROME-style causal tracing
    """
    
    def __init__(self, martian_api_key: str = None, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.rome_tracer = ROMETracer()
        self.confidence_scorer = ConfidenceScorer(rome_tracer=self.rome_tracer)
        self.model_router = ModelRouter()
        self.safety_monitor = SafetyMonitor()
        
        # Execution state
        self.task_registry: Dict[str, SubTask] = {}
        self.execution_history: List[ExecutionResult] = []
        self.api_key = martian_api_key
        
        # Performance tracking
        self.total_api_calls = 0
        self.total_cost = 0.0
        self.success_rate = 0.0
        
    def _default_config(self) -> Dict:
        """Default configuration for the controller."""
        return {
            'max_depth': 6,
            'confidence_threshold': 0.7,
            'safety_threshold': 0.8,
            'max_cost_per_task': 0.20,
            'parallel_execution': True,
            'enable_safety_checks': True,
            'enable_rome_tracing': True,
            'decomposition_strategy': 'adaptive'
        }
    
    def execute_task(self, task_prompt: str, context: str = "") -> ExecutionResult:
        """
        Main entry point for task execution with decomposition.
        
        Args:
            task_prompt: The main task to execute
            context: Additional context for the task
            
        Returns:
            ExecutionResult with metrics and results
        """
        start_time = time.time()
        
        # Create root task
        root_task = SubTask(
            id="root",
            prompt=task_prompt,
            context=context,
            depth=0
        )
        
        self.task_registry[root_task.id] = root_task
        
        try:
            # Execute with decomposition
            result = self._execute_with_decomposition(root_task)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            metrics = self._calculate_performance_metrics(execution_time)
            
            return ExecutionResult(
                success=result is not None,
                total_cost=metrics['total_cost'],
                execution_time=execution_time,
                tasks_completed=metrics['completed_tasks'],
                tasks_failed=metrics['failed_tasks'],
                confidence_avg=metrics['avg_confidence'],
                safety_violations=metrics['safety_violations'],
                decomposition_tree=self._build_decomposition_tree(),
                performance_metrics=metrics
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                total_cost=self.total_cost,
                execution_time=time.time() - start_time,
                tasks_completed=0,
                tasks_failed=1,
                confidence_avg=0.0,
                safety_violations=0,
                decomposition_tree={},
                performance_metrics={'error': str(e)}
            )
    
    def _execute_with_decomposition(self, task: SubTask) -> Optional[str]:
        """Execute task with adaptive decomposition."""
        
        # Step 1: Confidence Assessment
        confidence_analysis = self.confidence_scorer.calculate(
            task.prompt, task.context, task.depth
        )
        task.confidence = confidence_analysis.confidence
        
        # Step 2: Safety Check
        if self.config['enable_safety_checks']:
            safety_result = self.safety_monitor.validate_task(task.prompt, task.context)
            if not safety_result.safe:
                task.status = TaskStatus.FAILED
                task.safety_passed = False
                return None
        
        # Step 3: Route to Model
        routing_decision = self.model_router.route_task(
            task.prompt, task.confidence, task.context
        )
        task.assigned_model = routing_decision.selected_model
        
        # Step 4: Execution Decision
        if self._should_decompose(task, confidence_analysis):
            return self._decompose_and_execute(task)
        else:
            return self._execute_directly(task, routing_decision)
    
    def _should_decompose(self, task: SubTask, analysis: ConfidenceAnalysis) -> bool:
        """Decide whether to decompose task based on confidence and complexity."""
        
        # Don't decompose if at max depth
        if task.depth >= self.config['max_depth']:
            return False
        
        # Decompose if confidence is below threshold
        if analysis.confidence < self.config['confidence_threshold']:
            return True
        
        # Decompose if uncertainty is high
        if analysis.uncertainty > 0.7:
            return True
        
        # Check if task shows complexity indicators
        complexity_indicators = ['multi-step', 'analyze', 'comprehensive', 'detailed']
        if any(indicator in task.prompt.lower() for indicator in complexity_indicators):
            return True
        
        return False
    
    def _decompose_and_execute(self, parent_task: SubTask) -> Optional[str]:
        """Decompose task into subtasks and execute."""
        
        parent_task.status = TaskStatus.DECOMPOSED
        
        # Generate subtasks using planner model
        subtasks = self._generate_subtasks(parent_task)
        
        if not subtasks:
            # Fallback to direct execution
            return self._execute_directly(parent_task, 
                self.model_router.route_task(parent_task.prompt, parent_task.confidence)
            )
        
        # Execute subtasks
        subtask_results = []
        for subtask in subtasks:
            self.task_registry[subtask.id] = subtask
            result = self._execute_with_decomposition(subtask)
            if result:
                subtask_results.append(result)
            else:
                # Handle subtask failure
                parent_task.status = TaskStatus.FAILED
                return None
        
        # Synthesize results
        if subtask_results:
            parent_task.status = TaskStatus.COMPLETED
            return self._synthesize_results(parent_task, subtask_results)
        
        parent_task.status = TaskStatus.FAILED
        return None
    
    def _generate_subtasks(self, parent_task: SubTask) -> List[SubTask]:
        """Generate subtasks for decomposition."""
        
        # Simple rule-based decomposition (in practice, use LLM)
        subtasks = []
        
        # Decomposition patterns based on task type
        task_lower = parent_task.prompt.lower()
        
        if 'analyze' in task_lower:
            steps = [
                "Gather relevant information and data",
                "Identify key patterns and relationships", 
                "Draw conclusions and insights"
            ]
        elif 'create' in task_lower or 'design' in task_lower:
            steps = [
                "Define requirements and constraints",
                "Generate initial concepts and ideas",
                "Refine and finalize the design"
            ]
        elif 'compare' in task_lower:
            steps = [
                "Identify comparison criteria",
                "Analyze each option against criteria",
                "Summarize findings and recommendations"
            ]
        else:
            # Generic decomposition
            steps = [
                f"Break down the task: {parent_task.prompt}",
                f"Execute the main components",
                f"Integrate and finalize results"
            ]
        
        # Create subtasks
        for i, step in enumerate(steps):
            subtask = SubTask(
                id=f"{parent_task.id}_sub_{i}",
                prompt=step,
                context=parent_task.context,
                depth=parent_task.depth + 1,
                parent_id=parent_task.id
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _execute_directly(self, task: SubTask, routing_decision: RoutingDecision) -> Optional[str]:
        """Execute task directly using selected model."""
        
        task.status = TaskStatus.IN_PROGRESS
        start_time = time.time()
        
        try:
            # Simulate model execution (in practice, call actual API)
            result = self._simulate_model_call(
                task.prompt, 
                routing_decision.selected_model, 
                task.context
            )
            
            task.result = result
            task.cost = routing_decision.cost_estimate
            task.execution_time = time.time() - start_time
            task.status = TaskStatus.COMPLETED
            
            self.total_cost += task.cost
            self.total_api_calls += 1
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.execution_time = time.time() - start_time
            return None
    
    def _simulate_model_call(self, prompt: str, model: str, context: str = "") -> str:
        """Simulate model API call (replace with actual API call)."""
        
        # Simulate processing time based on model
        processing_times = {
            'gpt-4o': 2.0,
            'claude-3.5-sonnet': 1.5,
            'gpt-4o-mini': 0.8,
            'gemini-1.5-pro': 1.2
        }
        
        time.sleep(processing_times.get(model, 1.0) * 0.1)  # Reduced for demo
        
        # Generate mock response
        return f"[{model}] Response to: {prompt[:50]}... (simulated)"
    
    def _synthesize_results(self, parent_task: SubTask, subtask_results: List[str]) -> str:
        """Synthesize subtask results into final answer."""
        
        synthesis_prompt = f"""
        Original task: {parent_task.prompt}
        
        Subtask results:
        {chr(10).join(f"- {result}" for result in subtask_results)}
        
        Please synthesize these results into a comprehensive response.
        """
        
        # Route synthesis to high-capability model
        routing_decision = self.model_router.route_task(
            synthesis_prompt, 0.9  # High confidence for synthesis
        )
        
        return self._simulate_model_call(synthesis_prompt, routing_decision.selected_model)
    
    def _calculate_performance_metrics(self, execution_time: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        completed_tasks = len([t for t in self.task_registry.values() 
                              if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.task_registry.values() 
                           if t.status == TaskStatus.FAILED])
        
        confidences = [t.confidence for t in self.task_registry.values() 
                      if t.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        safety_violations = len([t for t in self.task_registry.values() 
                               if not t.safety_passed])
        
        return {
            'total_cost': self.total_cost,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'total_tasks': len(self.task_registry),
            'success_rate': completed_tasks / len(self.task_registry) if self.task_registry else 0,
            'avg_confidence': avg_confidence,
            'safety_violations': safety_violations,
            'api_calls': self.total_api_calls,
            'execution_time': execution_time,
            'avg_task_cost': self.total_cost / completed_tasks if completed_tasks > 0 else 0
        }
    
    def _build_decomposition_tree(self) -> Dict[str, Any]:
        """Build tree representation of task decomposition."""
        
        def build_node(task_id: str) -> Dict[str, Any]:
            task = self.task_registry[task_id]
            node = {
                'id': task_id,
                'prompt': task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt,
                'depth': task.depth,
                'confidence': task.confidence,
                'model': task.assigned_model,
                'status': task.status.value,
                'cost': task.cost,
                'execution_time': task.execution_time,
                'children': []
            }
            
            # Find children
            children = [t for t in self.task_registry.values() if t.parent_id == task_id]
            node['children'] = [build_node(child.id) for child in children]
            
            return node
        
        return build_node('root') if 'root' in self.task_registry else {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'total_executions': len(self.execution_history),
            'avg_success_rate': sum(r.success for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            'total_cost': sum(r.total_cost for r in self.execution_history),
            'model_router_stats': self.model_router.get_model_statistics(),
            'confidence_scorer_stats': {
                'total_assessments': len(self.task_registry),
                'avg_confidence': sum(t.confidence or 0 for t in self.task_registry.values()) / len(self.task_registry) if self.task_registry else 0
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize controller
    controller = EnhancedADaPTController()
    
    # Test task execution
    test_task = "Analyze the potential impacts of artificial intelligence on job markets in the next decade"
    
    print("Executing complex task with Enhanced ADaPT...")
    print("=" * 60)
    
    result = controller.execute_task(test_task)
    
    print(f"Success: {result.success}")
    print(f"Total Cost: ${result.total_cost:.3f}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Tasks Completed: {result.tasks_completed}")
    print(f"Average Confidence: {result.confidence_avg:.3f}")
    print(f"Safety Violations: {result.safety_violations}")
    
    # Show decomposition tree
    print("\\nDecomposition Tree:")
    print(json.dumps(result.decomposition_tree, indent=2))
'''

# Write the decomposition controller file
with open('decomp_router/core/controller.py', 'w') as f:
    f.write(decomp_controller_code)

print("✅ Created main decomposition controller")
# Update the main package __init__.py
main_init_code = '''"""
DecompRouter: Mechanistic Interpretability for Dynamic Task Decomposition

A novel framework for intelligent task decomposition using mechanistic interpretability
and dynamic model routing.
"""

from .core.controller import EnhancedADaPTController, ExecutionResult, SubTask
from .confidence.scorer import ConfidenceScorer, ConfidenceAnalysis
from .routing.router import ModelRouter, RoutingDecision, ModelProfile
from .safety.monitor import SafetyMonitor, SafetyResult
from .interpretability.rome_tracer import ROMETracer, CausalTraceResult

__version__ = "0.1.0"
__author__ = "Yash Ramani, Prince Ramani, Hisham Hanif"
__email__ = "team@decomprouter.dev"

# Main class for easy access
DecompRouter = EnhancedADaPTController

# Key exports
__all__ = [
    "DecompRouter",
    "EnhancedADaPTController",
    "ExecutionResult", 
    "SubTask",
    "ConfidenceScorer",
    "ConfidenceAnalysis",
    "ModelRouter",
    "RoutingDecision",
    "ModelProfile", 
    "SafetyMonitor",
    "SafetyResult",
    "ROMETracer",
    "CausalTraceResult"
]
'''

with open('decomp_router/__init__.py', 'w') as f:
    f.write(main_init_code)

print("✅ Updated main package __init__.py")

# Update confidence package __init__.py
confidence_init = '''"""Confidence scoring module for task assessment."""

from .scorer import ConfidenceScorer, ConfidenceAnalysis

__all__ = ["ConfidenceScorer", "ConfidenceAnalysis"]
'''

with open('decomp_router/confidence/__init__.py', 'w') as f:
    f.write(confidence_init)

# Update routing package __init__.py
routing_init = '''"""Model routing module for intelligent task assignment."""

from .router import ModelRouter, RoutingDecision, ModelProfile, ModelCapability

__all__ = ["ModelRouter", "RoutingDecision", "ModelProfile", "ModelCapability"]
'''

with open('decomp_router/routing/__init__.py', 'w') as f:
    f.write(routing_init)

# Update safety package __init__.py
safety_init = '''"""Safety monitoring module for secure task execution."""

from .monitor import SafetyMonitor, SafetyResult, SafetyViolation, SafetyLevel

__all__ = ["SafetyMonitor", "SafetyResult", "SafetyViolation", "SafetyLevel"]
'''

with open('decomp_router/safety/__init__.py', 'w') as f:
    f.write(safety_init)

# Update interpretability package __init__.py
interp_init = '''"""Mechanistic interpretability module for causal analysis."""

from .rome_tracer import ROMETracer, CausalTraceResult, ActivationPattern

__all__ = ["ROMETracer", "CausalTraceResult", "ActivationPattern"]
'''

with open('decomp_router/interpretability/__init__.py', 'w') as f:
    f.write(interp_init)

# Update core package __init__.py
core_init = '''"""Core decomposition logic and orchestration."""

from .controller import EnhancedADaPTController, ExecutionResult, SubTask, TaskStatus

__all__ = ["EnhancedADaPTController", "ExecutionResult", "SubTask", "TaskStatus"]
'''

with open('decomp_router/core/__init__.py', 'w') as f:
    f.write(core_init)

print("✅ Updated all package __init__.py files")
"""
Data Analysis Pipeline Orchestration.

This module contains pipeline orchestrators and workflow managers
for coordinating data analysis tasks across multiple AutoGen agents.
"""

from .data_pipeline import DataAnalysisPipeline
from .pipeline_orchestrator import PipelineOrchestrator
from .workflow_manager import WorkflowManager

__all__ = [
    "DataAnalysisPipeline",
    "PipelineOrchestrator", 
    "WorkflowManager"
]
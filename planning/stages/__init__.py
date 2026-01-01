"""Pipeline stages for action planning.

The action planning pipeline consists of three stages:
1. BasicAnalyzeStage: Detect touch/take/place attributes via LLM voting
2. PoseReasoningStage: Determine position and facing direction
3. PointReasoningStage: Locate specific contact points

Each stage can run independently, enabling pipeline parallelism where
multiple actions can be processed concurrently at different stages.
"""

from planner.planning.stages.base import (
    PipelineStage,
    StageInput,
    StageOutput,
    ActionContext,
)
from planner.planning.stages.basic_analyze import BasicAnalyzeStage
from planner.planning.stages.pose_reasoning import PoseReasoningStage
from planner.planning.stages.point_reasoning import PointReasoningStage

__all__ = [
    'PipelineStage',
    'StageInput',
    'StageOutput',
    'ActionContext',
    'BasicAnalyzeStage',
    'PoseReasoningStage',
    'PointReasoningStage',
]

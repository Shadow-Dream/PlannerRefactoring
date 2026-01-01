"""Planning backend package.

The planning system uses a three-stage pipeline architecture:
1. BasicAnalyzeStage: Detect touch/take/place attributes
2. PoseReasoningStage: Determine position and facing
3. PointReasoningStage: Locate contact points

This architecture enables future pipeline parallelism where multiple
actions can be processed concurrently at different stages.
"""
from planner.planning.planning_backend import PlanningBackend
from planner.planning.action_parser import ActionParser
from planner.planning.position_handler import PositionHandler
from planner.planning.contact_locator import ContactPointLocator
from planner.planning.stages import (
    PipelineStage,
    StageInput,
    StageOutput,
    ActionContext,
    BasicAnalyzeStage,
    PoseReasoningStage,
    PointReasoningStage,
)

__all__ = [
    'PlanningBackend',
    'ActionParser',
    'PositionHandler',
    'ContactPointLocator',
    'PipelineStage',
    'StageInput',
    'StageOutput',
    'ActionContext',
    'BasicAnalyzeStage',
    'PoseReasoningStage',
    'PointReasoningStage',
]

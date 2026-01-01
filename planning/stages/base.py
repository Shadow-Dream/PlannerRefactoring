"""Base classes and data structures for pipeline stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import copy


class StageType(Enum):
    """Pipeline stage types."""
    BASIC_ANALYZE = "basic_analyze"
    POSE_REASONING = "pose_reasoning"
    POINT_REASONING = "point_reasoning"


@dataclass
class ActionContext:
    """Context for action processing across pipeline stages.

    This class holds all the data needed to process an action through
    the pipeline stages. It's designed to be passed between stages
    and can be serialized for queue-based communication.
    """
    # Action identification
    action_id: str
    action_string: str

    # Parsed action attributes (set by BasicAnalyzeStage)
    target: Optional[str] = None
    at: Optional[List[str]] = None
    by: Optional[List[str]] = None
    touch: Optional[bool] = None
    take: Optional[bool] = None
    place: Optional[bool] = None
    long_range: Optional[bool] = None
    contact_points: Optional[List[str]] = None

    # Position attributes (set by PoseReasoningStage)
    position: Optional[List[float]] = None
    glb_position: Optional[List[float]] = None
    facing: Optional[float] = None
    glb_facing: Optional[float] = None
    position_tag: Optional[int] = None
    tag_direction: Optional[str] = None

    # Contact targets (set by PointReasoningStage)
    contact_targets: Optional[List[List[float]]] = None
    glb_contact_targets: Optional[List[List[float]]] = None

    # Processing state
    current_stage: StageType = StageType.BASIC_ANALYZE
    completed_stages: List[StageType] = field(default_factory=list)
    error: Optional[str] = None

    # Flags for special handling
    pseudo_take: bool = False
    pseudo_place: bool = False
    skip_remaining: bool = False

    def clone(self) -> 'ActionContext':
        """Create a deep copy of this context."""
        return copy.deepcopy(self)

    def mark_stage_complete(self, stage: StageType):
        """Mark a stage as completed."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)

    def is_stage_complete(self, stage: StageType) -> bool:
        """Check if a stage is completed."""
        return stage in self.completed_stages


@dataclass
class StageInput:
    """Input data for a pipeline stage."""
    # Action context being processed
    context: ActionContext

    # Environment state (shared, read-only during stage processing)
    objects: Dict[str, Any] = field(default_factory=dict)
    parents: Dict[str, str] = field(default_factory=dict)
    object_dict: Dict[str, List[str]] = field(default_factory=dict)

    # Agent state
    state: Dict[str, Any] = field(default_factory=dict)
    last_state_position: List[float] = field(default_factory=lambda: [0, 0])

    # Current actions being performed (for concurrent action handling)
    action_dict: Dict[str, Any] = field(default_factory=dict)

    # Additional stage-specific data
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageOutput:
    """Output data from a pipeline stage."""
    # Updated action context
    context: ActionContext

    # Whether processing should continue to next stage
    continue_pipeline: bool = True

    # Capture request (if stage needs capture data)
    capture_request: Optional[List[str]] = None

    # Result to send (if stage produces final output)
    result: Optional[Dict[str, Any]] = None

    # Error information
    error: Optional[str] = None


class PipelineStage(ABC):
    """Abstract base class for pipeline stages.

    Each stage processes an ActionContext and produces an updated context.
    Stages are designed to be:
    1. Stateless - all state is in the context
    2. Independent - can run in separate processes
    3. Resumable - can continue from capture results
    """

    def __init__(self, env_id: int, vlm_client, logger):
        """Initialize pipeline stage.

        Args:
            env_id: Environment identifier
            vlm_client: VLM client for LLM queries
            logger: Logger for recording conversations
        """
        self.env_id = env_id
        self.vlm_client = vlm_client
        self.logger = logger

    @property
    @abstractmethod
    def stage_type(self) -> StageType:
        """Return the type of this stage."""
        pass

    @abstractmethod
    def process(self, stage_input: StageInput) -> StageOutput:
        """Process input and return output.

        Args:
            stage_input: Input data for this stage

        Returns:
            StageOutput with updated context and processing results
        """
        pass

    def can_process(self, context: ActionContext) -> bool:
        """Check if this stage can process the given context.

        Args:
            context: Action context to check

        Returns:
            True if this stage can process the context
        """
        # Check if previous stages are completed
        stage_order = [
            StageType.BASIC_ANALYZE,
            StageType.POSE_REASONING,
            StageType.POINT_REASONING,
        ]
        current_index = stage_order.index(self.stage_type)

        # All previous stages must be completed
        for i in range(current_index):
            if not context.is_stage_complete(stage_order[i]):
                return False

        # This stage must not be completed yet
        return not context.is_stage_complete(self.stage_type)


class PipelineCoordinator:
    """Coordinates pipeline stage execution.

    This class manages the flow of ActionContexts through pipeline stages.
    It can be extended to support parallel processing where multiple
    contexts are processed concurrently at different stages.
    """

    def __init__(self, env_id: int, vlm_client, logger):
        """Initialize pipeline coordinator.

        Args:
            env_id: Environment identifier
            vlm_client: VLM client for LLM queries
            logger: Logger for recording conversations
        """
        self.env_id = env_id
        self.vlm_client = vlm_client
        self.logger = logger

        # Initialize stages (lazy loading to avoid circular imports)
        self._stages: Dict[StageType, PipelineStage] = {}

    def _get_stage(self, stage_type: StageType) -> PipelineStage:
        """Get or create a pipeline stage."""
        if stage_type not in self._stages:
            from planner.planning.stages.basic_analyze import BasicAnalyzeStage
            from planner.planning.stages.pose_reasoning import PoseReasoningStage
            from planner.planning.stages.point_reasoning import PointReasoningStage

            stage_classes = {
                StageType.BASIC_ANALYZE: BasicAnalyzeStage,
                StageType.POSE_REASONING: PoseReasoningStage,
                StageType.POINT_REASONING: PointReasoningStage,
            }
            self._stages[stage_type] = stage_classes[stage_type](
                self.env_id, self.vlm_client, self.logger
            )
        return self._stages[stage_type]

    def process_action(self, stage_input: StageInput) -> StageOutput:
        """Process an action through all applicable pipeline stages.

        Args:
            stage_input: Input data containing action context

        Returns:
            Final StageOutput after all stages complete
        """
        context = stage_input.context
        current_output = None

        # Process through each stage in order
        stage_order = [
            StageType.BASIC_ANALYZE,
            StageType.POSE_REASONING,
            StageType.POINT_REASONING,
        ]

        for stage_type in stage_order:
            stage = self._get_stage(stage_type)

            if not stage.can_process(context):
                continue

            # Update context in input
            stage_input.context = context

            # Process stage
            current_output = stage.process(stage_input)
            context = current_output.context

            # Check for early termination
            if not current_output.continue_pipeline:
                break

            # Check for capture request (need to wait)
            if current_output.capture_request:
                return current_output

        return current_output or StageOutput(context=context)

    def process_single_stage(self, stage_type: StageType,
                            stage_input: StageInput) -> StageOutput:
        """Process a single pipeline stage.

        This method is useful for pipeline parallelism where stages
        run in separate processes.

        Args:
            stage_type: Type of stage to process
            stage_input: Input data for the stage

        Returns:
            StageOutput from the stage
        """
        stage = self._get_stage(stage_type)
        return stage.process(stage_input)

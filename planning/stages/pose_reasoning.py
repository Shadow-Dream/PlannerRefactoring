"""Stage 2: Pose Reasoning - Determine position and facing direction.

This stage:
1. Determines the position where the agent should stand
2. Computes the facing direction
3. Handles both long-range and close-range interactions

This stage depends on BasicAnalyzeStage for touch/long_range attributes.
"""

import re
import copy
import numpy as np
from typing import Dict, Any, List, Optional

from planner.planning.stages.base import (
    PipelineStage, StageType, StageInput, StageOutput, ActionContext
)
from planner.planning.position_handler import PositionHandler
from planner.core import quaternion
from planner.prompts import SURFACE_TO_JOINT
from planner.utils.geometry_utils import get_instance_distance, get_instance_position_distance
from planner.utils.format_utils import format_coordinate_string


class PoseReasoningStage(PipelineStage):
    """Stage 2: Position and facing reasoning.

    Responsibilities:
    - Determine position for long-range interactions
    - Determine position for close-range interactions
    - Compute facing direction
    - Request capture data when needed
    """

    def __init__(self, env_id: int, vlm_client, logger):
        """Initialize pose reasoning stage."""
        super().__init__(env_id, vlm_client, logger)
        self.position_handler = PositionHandler(env_id, vlm_client, logger)

    @property
    def stage_type(self) -> StageType:
        return StageType.POSE_REASONING

    def process(self, stage_input: StageInput) -> StageOutput:
        """Process pose reasoning.

        Args:
            stage_input: Input containing action context and environment state

        Returns:
            StageOutput with updated context
        """
        context = stage_input.context
        objects = stage_input.objects
        parents = stage_input.parents
        state = stage_input.state
        action_dict = stage_input.action_dict
        object_dict = stage_input.object_dict
        last_state_position = stage_input.last_state_position

        # Skip if no target or already marked to skip
        if context.target is None or context.skip_remaining:
            context.mark_stage_complete(self.stage_type)
            return StageOutput(context=context, continue_pipeline=False)

        # Determine target for positioning
        if context.place and context.at and len(context.at) > 0:
            target = context.at[0]
        else:
            target = context.target

        parent = parents.get(target, target)

        # Determine anchor (position reference)
        anchor, is_previous_anchor = self._determine_anchor(
            parent, action_dict, parents
        )

        # Check if we need capture data
        capture_targets = [target, anchor] if target != anchor else [target]

        # Check if capture data is available
        if 'capture_data' not in stage_input.extra:
            # Request capture and wait
            context.current_stage = StageType.POSE_REASONING
            return StageOutput(
                context=context,
                continue_pipeline=True,
                capture_request=capture_targets
            )

        # Update objects with capture data
        capture_data = stage_input.extra['capture_data']
        if len(capture_targets) == 1:
            objects[capture_targets[0]].update(capture_data[0])
        else:
            objects[capture_targets[0]].update(capture_data[0])
            objects[anchor].update(capture_data[1])

        # Build merged action string
        merged_action_strings = list(action_dict.keys()) + [context.action_string]
        if len(merged_action_strings) == 1:
            merged_action_type = merged_action_strings[0]
        elif len(merged_action_strings) == 2:
            merged_action_type = f"{merged_action_strings[0]} and {merged_action_strings[1]}"
        else:
            merged_action_type = ", ".join(merged_action_strings[:-1]) + f", and {merged_action_strings[-1]}"
        merged_action_type = re.sub(r'\d', '', merged_action_type)

        # Handle position based on interaction type
        if context.long_range:
            context = self._handle_long_range(
                context, target, objects, object_dict, action_dict, state
            )
        else:
            context = self._handle_close_range(
                context, target, anchor, objects, parents,
                action_dict, merged_action_type, last_state_position
            )

        # Transform to relative coordinates
        context = self._transform_to_relative(context, objects, parents)

        # Mark stage complete
        context.mark_stage_complete(self.stage_type)
        context.current_stage = StageType.POINT_REASONING

        # Determine if we should continue to point reasoning
        should_continue = (
            context.touch or context.place
        ) and not context.long_range

        return StageOutput(
            context=context,
            continue_pipeline=should_continue
        )

    def _determine_anchor(self, parent: str, action_dict: Dict[str, Any],
                         parents: Dict[str, str]) -> tuple:
        """Determine the anchor object for positioning."""
        is_previous_anchor = False
        anchor = parent

        for current_action in action_dict.values():
            if current_action.place and current_action.at and len(current_action.at) > 0:
                anchor = current_action.at[0]
                anchor = parents.get(anchor, anchor)
                is_previous_anchor = True
            elif current_action.target is not None and not current_action.long_range:
                anchor = parents.get(current_action.target, current_action.target)
                is_previous_anchor = True

        return anchor, is_previous_anchor

    def _handle_long_range(self, context: ActionContext, target: str,
                          objects: Dict[str, Any], object_dict: Dict[str, List[str]],
                          action_dict: Dict[str, Any], state: Dict[str, Any]) -> ActionContext:
        """Handle position specification for long-range interactions."""
        # Create a temporary action object for position handler
        from planner.core.action import HumanAction
        action = HumanAction(context.action_string, objects)
        action.target = context.target
        action.at = context.at
        action.by = context.by
        action.touch = context.touch
        action.take = context.take
        action.place = context.place
        action.long_range = context.long_range
        action.contact_points = context.contact_points

        # Use position handler for long-range
        action = self.position_handler.handle_long_range_position(
            action, target, objects, object_dict, action_dict, state
        )

        # Update context with results
        context.position = action.position
        context.facing = action.facing

        return context

    def _handle_close_range(self, context: ActionContext, target: str, anchor: str,
                           objects: Dict[str, Any], parents: Dict[str, str],
                           action_dict: Dict[str, Any], merged_action_type: str,
                           last_state_position: List[float]) -> ActionContext:
        """Handle position specification for close-range interactions."""
        # Create a temporary action object for position handler
        from planner.core.action import HumanAction
        action = HumanAction(context.action_string, objects)
        action.target = context.target
        action.at = context.at
        action.by = context.by
        action.touch = context.touch
        action.take = context.take
        action.place = context.place
        action.long_range = context.long_range
        action.contact_points = context.contact_points

        # Use position handler for close-range
        action, target_directions, anchor_directions, anchor_position_tag = (
            self.position_handler.handle_close_range_position(
                action, target, anchor, objects, parents,
                action_dict, merged_action_type, last_state_position
            )
        )

        # Compute facing direction
        self._compute_facing(action, objects, anchor, action_dict)

        # Update context with results
        context.position = action.position
        context.position_tag = action.position_tag
        context.tag_direction = action.tag_direction
        context.facing = action.facing

        return context

    def _compute_facing(self, action, objects: Dict[str, Any],
                       anchor: str, action_dict: Dict[str, Any]):
        """Compute facing direction for action."""
        anchor_contact_points = []
        for current_action in action_dict.values():
            if current_action.contact_points is not None:
                anchor_contact_points.extend(current_action.contact_points)

        contact_points = action.contact_points or []
        if "pelvis" in [SURFACE_TO_JOINT.get(j, j) for j in contact_points]:
            action.facing = [
                None, 0, np.pi/4, np.pi/2, np.pi*3/4,
                np.pi, -np.pi*3/4, -np.pi/2, -np.pi/4
            ][action.position_tag]
        elif "pelvis" in [SURFACE_TO_JOINT.get(j, j) for j in anchor_contact_points]:
            anchor_instance = objects.get(anchor, {})
            if anchor_instance and action.position:
                act_position = np.array(action.position)[:2]
                anc_position = (
                    np.array(anchor_instance["position"])[:2] +
                    np.array(anchor_instance.get("bbox", [[0, 0, 0]] * 8)).mean(0)[:2]
                )
                anc_direction = act_position - anc_position
                action.facing = np.arctan2(anc_direction[1], anc_direction[0])

    def _transform_to_relative(self, context: ActionContext,
                              objects: Dict[str, Any],
                              parents: Dict[str, str]) -> ActionContext:
        """Transform action coordinates to relative space."""
        if context.place and context.at and len(context.at) > 0:
            reposition_target = context.at[0]
        elif context.target is not None:
            reposition_target = context.target
        else:
            reposition_target = None

        if reposition_target is not None and context.position is not None:
            # Save global position
            context.glb_position = copy.deepcopy(context.position)

            act_position = np.array(context.position).tolist()
            if len(act_position) == 2:
                act_position += [0]
            act_position = np.array(act_position)[:3]

            tar_instance = objects[reposition_target]
            tar_position = np.array(tar_instance["position"])[:3]
            tar_rotation = np.array(tar_instance["rotation"])
            tar_rotation = tar_rotation[None]
            tar_rotation = quaternion.qinv_np(tar_rotation)

            rel_position = act_position - tar_position
            rel_position = rel_position[None]
            rel_position = quaternion.qrot_np(tar_rotation, rel_position)
            rel_position = rel_position[0][:2].tolist()
            context.position = rel_position

            # Transform facing
            if context.facing is not None:
                context.glb_facing = copy.deepcopy(context.facing)
                fac_direction = [np.cos(context.facing), np.sin(context.facing), 0]
                fac_direction = np.array(fac_direction)[None]
                fac_direction = quaternion.qrot_np(tar_rotation, fac_direction)
                fac_direction = fac_direction[0, :2]
                context.facing = float(np.arctan2(fac_direction[1], fac_direction[0]))

        return context

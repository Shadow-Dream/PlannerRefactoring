"""Stage 3: Point Reasoning - Locate specific contact points.

This stage:
1. Locates specific contact points on the target object
2. Uses grid-based VLM selection for efficient localization
3. Transforms contact targets to relative coordinates

This stage depends on both BasicAnalyzeStage (for contact_points)
and PoseReasoningStage (for position_tag).
"""

import re
import copy
import numpy as np
from typing import Dict, Any, List, Optional

from planner.planning.stages.base import (
    PipelineStage, StageType, StageInput, StageOutput, ActionContext
)
from planner.planning.contact_locator import ContactPointLocator
from planner.planning.position_handler import PositionHandler
from planner.core import quaternion
from planner.utils.format_utils import get_volume


class PointReasoningStage(PipelineStage):
    """Stage 3: Contact point localization.

    Responsibilities:
    - Locate specific contact points on target object
    - Handle both detailed and simple localization
    - Transform contact targets to relative coordinates
    """

    def __init__(self, env_id: int, vlm_client, logger):
        """Initialize point reasoning stage."""
        super().__init__(env_id, vlm_client, logger)
        self.position_handler = PositionHandler(env_id, vlm_client, logger)
        self.contact_locator = ContactPointLocator(
            env_id, vlm_client, logger, self.position_handler
        )

    @property
    def stage_type(self) -> StageType:
        return StageType.POINT_REASONING

    def process(self, stage_input: StageInput) -> StageOutput:
        """Process contact point reasoning.

        Args:
            stage_input: Input containing action context and environment state

        Returns:
            StageOutput with updated context
        """
        context = stage_input.context
        objects = stage_input.objects
        parents = stage_input.parents
        action_dict = stage_input.action_dict

        # Skip if no target or not touch/place action
        if context.target is None or context.skip_remaining:
            context.mark_stage_complete(self.stage_type)
            return StageOutput(context=context, continue_pipeline=False)

        # Skip if long-range action
        if context.long_range:
            context.mark_stage_complete(self.stage_type)
            return StageOutput(context=context, continue_pipeline=False)

        # Skip if no contact points
        if not context.contact_points or len(context.contact_points) == 0:
            context.mark_stage_complete(self.stage_type)
            return StageOutput(context=context, continue_pipeline=False)

        # Determine target for contact localization
        if context.place and context.at and len(context.at) > 0:
            target = context.at[0]
        else:
            target = context.target

        parent = parents.get(target, target)
        anchor = self._determine_anchor(parent, action_dict, parents)

        # Get target instance
        target_instance = objects.get(target, {})

        # Build merged action type
        merged_action_strings = list(action_dict.keys()) + [context.action_string]
        if len(merged_action_strings) == 1:
            merged_action_type = merged_action_strings[0]
        elif len(merged_action_strings) == 2:
            merged_action_type = f"{merged_action_strings[0]} and {merged_action_strings[1]}"
        else:
            merged_action_type = ", ".join(merged_action_strings[:-1]) + f", and {merged_action_strings[-1]}"
        merged_action_type = re.sub(r'\d', '', merged_action_type)

        # Locate contact points
        if context.touch or context.place:
            # Check volume for detailed vs simple localization
            volume = get_volume(target_instance) if "bbox" in target_instance else 0

            if volume > 0.125 and not context.take:
                # Detailed localization via VLM
                context = self._locate_detailed_contacts(
                    context, target, anchor, objects,
                    re.sub(r'\d', '', anchor), merged_action_type
                )
            else:
                # Simple localization using object center
                context = self._locate_simple_contacts(context, target_instance)

        # Transform contact targets to relative coordinates
        context = self._transform_contacts_to_relative(context, objects)

        # Mark stage complete
        context.mark_stage_complete(self.stage_type)

        return StageOutput(
            context=context,
            continue_pipeline=False
        )

    def _determine_anchor(self, parent: str, action_dict: Dict[str, Any],
                         parents: Dict[str, str]) -> str:
        """Determine the anchor object for positioning."""
        anchor = parent

        for current_action in action_dict.values():
            if current_action.place and current_action.at and len(current_action.at) > 0:
                anchor = current_action.at[0]
                anchor = parents.get(anchor, anchor)
            elif current_action.target is not None and not current_action.long_range:
                anchor = parents.get(current_action.target, current_action.target)

        return anchor

    def _locate_detailed_contacts(self, context: ActionContext, target: str,
                                  anchor: str, objects: Dict[str, Any],
                                  anchor_type: str, merged_action_type: str) -> ActionContext:
        """Locate contact points using detailed VLM-based localization."""
        # Create a temporary action object for contact locator
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
        action.position = context.position
        action.position_tag = context.position_tag
        action.tag_direction = context.tag_direction

        # Use contact locator
        contact_targets = self.contact_locator.locate_contact_points(
            action, target, anchor, objects,
            context.tag_direction, anchor_type,
            merged_action_type, context.position_tag
        )

        context.contact_targets = contact_targets
        return context

    def _locate_simple_contacts(self, context: ActionContext,
                               target_instance: Dict[str, Any]) -> ActionContext:
        """Locate contact targets using simple position approximation."""
        if not target_instance or "position" not in target_instance:
            context.contact_targets = []
            return context

        position = np.array(target_instance["position"])
        bbox = target_instance.get("bbox", [[0, 0, 0]] * 8)
        position = position + np.array(bbox).mean(0)

        context.contact_targets = [
            position.tolist() for _ in (context.contact_points or [])
        ]
        return context

    def _transform_contacts_to_relative(self, context: ActionContext,
                                        objects: Dict[str, Any]) -> ActionContext:
        """Transform contact targets to relative coordinates."""
        if context.place and context.at and len(context.at) > 0:
            reposition_target = context.at[0]
        elif context.target is not None:
            reposition_target = context.target
        else:
            reposition_target = None

        if (reposition_target is not None and
            context.contact_targets is not None and
            len(context.contact_targets) > 0):

            # Save global contact targets
            context.glb_contact_targets = copy.deepcopy(context.contact_targets)

            tar_instance = objects[reposition_target]
            tar_position = np.array(tar_instance["position"])[:3]
            tar_rotation = np.array(tar_instance["rotation"])
            tar_rotation = tar_rotation[None]
            tar_rotation = quaternion.qinv_np(tar_rotation)

            con_position = np.array(context.contact_targets)
            con_position = con_position - tar_position
            tar_rota_con = np.broadcast_to(tar_rotation, (con_position.shape[0], 4))
            con_position = quaternion.qrot_np(tar_rota_con, con_position)
            context.contact_targets = con_position.tolist()

        return context

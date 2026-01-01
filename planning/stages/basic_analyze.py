"""Stage 1: Basic Analyze - Detect touch/take/place attributes.

This stage:
1. Parses action targets from the action string
2. Detects touch/take/place/long_range attributes via LLM voting
3. Determines contact points

This is the first stage in the pipeline and has no dependencies
on other stages.
"""

import re
import numpy as np
from collections import Counter
from concurrent import futures as ftr
from typing import Dict, Any, List, Optional

from planner.planning.stages.base import (
    PipelineStage, StageType, StageInput, StageOutput, ActionContext
)
from planner.core.action import HumanAction
from planner.prompts import (
    SYSTEM_SPECIFY_TOUCH, USER1_SPECIFY_TOUCH, ASSISTANT1_SPECIFY_TOUCH,
    USER2_SPECIFY_TOUCH, USER_SPECIFY_TARGET, SURFACE_TO_JOINT
)


class BasicAnalyzeStage(PipelineStage):
    """Stage 1: Basic action attribute analysis.

    Responsibilities:
    - Parse targets from action string
    - Detect touch/take/place attributes via LLM voting
    - Determine contact points
    - Compute relative direction for ongoing actions
    """

    @property
    def stage_type(self) -> StageType:
        return StageType.BASIC_ANALYZE

    def process(self, stage_input: StageInput) -> StageOutput:
        """Process basic action analysis.

        Args:
            stage_input: Input containing action context and environment state

        Returns:
            StageOutput with updated context
        """
        context = stage_input.context
        objects = stage_input.objects
        state = stage_input.state
        action_dict = stage_input.action_dict

        # Create HumanAction for parsing
        action = HumanAction(context.action_string, objects)

        # Step 1: Parse targets
        action = self._parse_targets(action)

        # Update context with parsed targets
        context.target = action.target
        context.at = action.at
        context.by = action.by

        # Check if target exists
        if context.target and context.target not in objects:
            context.target = None
            context.skip_remaining = True
            context.mark_stage_complete(self.stage_type)
            return StageOutput(
                context=context,
                continue_pipeline=False
            )

        # Step 2: Detect touch attributes if target exists
        if context.target:
            # Build interaction and state strings
            interaction_string = self._build_interaction_string(context)
            state_string = self._build_state_string(state)

            # Compute relative direction for ongoing actions
            relative_string = self._compute_relative_direction(
                context, objects, action_dict
            )

            # Detect touch attributes via voting
            touch_result = self._detect_touch_attributes(
                interaction_string, state_string, relative_string, state
            )

            context.touch = touch_result['touch']
            context.take = touch_result['take']
            context.place = touch_result['place']
            context.contact_points = touch_result['contact_points']

            # Handle place action contact points
            if context.place and (not context.contact_points or len(context.contact_points) == 0):
                contact_points = []
                if context.target == state.get("left_slot"):
                    contact_points.append("left_wrist")
                if context.target == state.get("right_slot"):
                    contact_points.append("right_wrist")
                context.contact_points = contact_points

            # Set long_range based on touch
            context.long_range = not context.touch

            # Handle non-takable objects
            context = self._handle_non_takable(context)

        # Mark stage complete
        context.mark_stage_complete(self.stage_type)
        context.current_stage = StageType.POSE_REASONING

        return StageOutput(
            context=context,
            continue_pipeline=context.target is not None and not context.skip_remaining
        )

    def _parse_targets(self, action: HumanAction) -> HumanAction:
        """Parse and identify targets from action with multiple objects."""
        if len(action.targets) <= 1:
            if len(action.targets) == 1:
                action.target = action.targets[0]
            return action

        user1 = USER_SPECIFY_TARGET.replace("sentence_placeholder", action.marked_action)
        self.logger.print_role("User")
        self.logger.write(user1)
        self.logger.write()

        response = self.vlm_client.chat([{"role": "user", "content": user1}])
        self.logger.print_role("Assistant")
        self.logger.write(response)
        self.logger.write()

        lines = response.split("\n")
        pattern = re.compile(r"<(.*?)>")
        for line in lines:
            targets = pattern.findall(line)
            if "target" in line:
                action.target = targets[0]
            elif "by" in line:
                action.by = targets
            elif "at" in line:
                action.at = targets

        return action

    def _build_interaction_string(self, context: ActionContext) -> str:
        """Build interaction string for touch detection."""
        interaction = []
        marked_action = context.action_string.replace(
            context.target, f"<{context.target}>"
        ) if context.target else context.action_string

        interaction.append("Action: " + marked_action)
        interaction.append("Target: " + (context.target or ""))
        if context.at:
            interaction.append("At: " + ", ".join(context.at))
        if context.by:
            interaction.append("By: " + ", ".join(context.by))
        return "\n".join(interaction)

    def _build_state_string(self, state: Dict[str, Any]) -> str:
        """Build state string for touch detection."""
        left = state.get("left_slot")
        right = state.get("right_slot")
        if left and right:
            return f"You are holding {left} in your left hand and {right} in your right hand."
        elif left:
            return f"You are holding {left} in your left hand. Your right hand is available."
        elif right:
            return f"You are holding {right} in your right hand. Your left hand is available."
        else:
            return "You are currently holding nothing in your hands."

    def _compute_relative_direction(self, context: ActionContext,
                                   objects: Dict[str, Any],
                                   action_dict: Dict[str, Any]) -> Optional[str]:
        """Compute relative direction string based on current action state."""
        current_action_position = None
        current_action_facing = None

        for current_action_string, current_action in action_dict.items():
            current_contact_points = [
                SURFACE_TO_JOINT.get(p, p) for p in (current_action.contact_points or [])
            ]

            if len(current_contact_points) > 0 and "pelvis" in current_contact_points:
                glb_targets = current_action.glb_contact_targets
                if glb_targets:
                    current_action_position = glb_targets[
                        current_contact_points.index("pelvis")
                    ][:2]
            elif current_action.glb_position is not None:
                current_action_position = current_action.glb_position

            if current_action.glb_facing is not None:
                current_action_facing = current_action.glb_facing

        if current_action_facing is None or current_action_position is None:
            return None

        if context.target not in objects:
            return None

        target_instance = objects[context.target]
        target_position = (
            np.array(target_instance["position"], dtype=np.float32) +
            np.array(target_instance["bbox"], dtype=np.float32).mean(0)
        )
        target_position = target_position[:2]
        current_position = np.array(current_action_position, dtype=np.float32)
        target_direction = target_position - current_position

        current_front = np.array([np.cos(current_action_facing), np.sin(current_action_facing)])
        current_right = np.array([np.sin(current_action_facing), -np.cos(current_action_facing)])
        current_front = np.dot(current_front, target_direction)
        current_right = np.dot(current_right, target_direction)

        direction_map = np.array([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=np.float32)
        direction_map = direction_map / np.linalg.norm(direction_map, axis=-1, keepdims=True)
        direction_index = current_front * direction_map[:, 0] + current_right * direction_map[:, 1]
        direction_index = direction_index.argmax()

        direction_strings = [
            "front", "right-hand side", "right-hand side", "right-hand side",
            "back", "left-hand side", "left-hand side", "left-hand side"
        ]
        return f"The {context.target} is in the {direction_strings[direction_index]} of you."

    def _detect_touch_attributes(self, interaction_string: str,
                                state_string: str,
                                relative_string: Optional[str],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect touch, take, place attributes using voting."""
        if relative_string is not None:
            state_string = state_string + "\n" + relative_string

        user2 = (USER2_SPECIFY_TOUCH
                .replace("interaction_placeholder", interaction_string)
                .replace("state_placeholder", state_string))

        self.logger.print_role("System")
        self.logger.write(SYSTEM_SPECIFY_TOUCH)
        self.logger.write()
        self.logger.print_role("User")
        self.logger.write(user2)
        self.logger.write()

        touch_buffer = []
        take_buffer = []
        range_buffer = []
        place_buffer = []
        contact_point_buffer = []

        def vote_model(model_id):
            """Single vote from model."""
            system = SYSTEM_SPECIFY_TOUCH.replace("{id}", model_id)
            response = self.vlm_client.chat_with_lock([
                {"role": "system", "content": system},
                {"role": "user", "content": USER1_SPECIFY_TOUCH},
                {"role": "assistant", "content": ASSISTANT1_SPECIFY_TOUCH},
                {"role": "user", "content": user2}
            ])

            touch = take = long_range = place = False
            contact_points = []

            lines = response.split("\n")
            for line in lines:
                line = line.lower()
                if "touch" in line:
                    touch = ("true" in line)
                elif "take" in line:
                    take = ("true" in line)
                elif "range" in line:
                    long_range = ("true" in line)
                elif "place" in line:
                    place = ("true" in line)
                elif "contact point" in line:
                    contact_points = line.split("[")[1].split("]")[0].split(",")
                    contact_points = [point.replace("'", "").strip() for point in contact_points]
                    contact_points = [
                        ("left_wrist" if state.get("left_slot") == joint
                         else "right_wrist" if state.get("right_slot") == joint
                         else joint)
                        for joint in contact_points
                    ]
            return touch, take, long_range, place, contact_points

        with ftr.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(vote_model, model_id)
                for model_id in ["Alpha", "Epsilon", "Sigma", "Omega", "Zeta"]
            ]
            for future in ftr.as_completed(futures):
                touch, take, long_range, place, contact_points = future.result()
                touch_buffer.append(touch)
                take_buffer.append(take)
                range_buffer.append(long_range)
                place_buffer.append(place)
                contact_point_buffer.append(contact_points)

        self.logger.print_role("Assistant")
        self.logger.write("The model votes:")
        self.logger.write(f"Touch: {touch_buffer}")
        self.logger.write(f"Take: {take_buffer}")
        self.logger.write(f"Long Range: {range_buffer}")
        self.logger.write(f"Place: {place_buffer}")
        self.logger.write(f"Contact Points: {contact_point_buffer}")

        touch = touch_buffer.count(True) >= 3
        take = take_buffer.count(True) >= 3
        place = place_buffer.count(True) >= 3

        if not touch:
            contact_points = []
        else:
            filtered_points = [
                cp for idx, cp in enumerate(contact_point_buffer)
                if idx < len(touch_buffer) and touch_buffer[idx] and cp
            ]
            if filtered_points:
                counter = Counter(tuple(sublist) for sublist in filtered_points)
                most_common = counter.most_common(1)[0][0]
                contact_points = list(most_common)
            else:
                contact_points = []

        return {
            'touch': touch,
            'take': take,
            'place': place,
            'contact_points': contact_points
        }

    def _handle_non_takable(self, context: ActionContext) -> ActionContext:
        """Handle non-takable objects."""
        takable_objects = ["plant container", "nature shelf trinkets", "box"]
        is_takable = any(obj in context.target for obj in takable_objects)

        if not is_takable:
            if context.take:
                context.take = False
                context.pseudo_take = True
                if context.contact_points and len(context.contact_points) > 1:
                    context.contact_points = context.contact_points[:1]

            if context.place:
                context.place = False
                context.pseudo_place = True
                if context.contact_points and len(context.contact_points) > 1:
                    context.contact_points = context.contact_points[:1]

        return context

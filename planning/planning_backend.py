"""Main planning backend for LLM-based behavior planning.

The planning backend orchestrates the LLM-based planning process,
handling command parsing and action execution through a three-stage pipeline:

1. BasicAnalyzeStage: Detect touch/take/place attributes
2. PoseReasoningStage: Determine position and facing
3. PointReasoningStage: Locate contact points

This architecture enables future pipeline parallelism where multiple
actions can be processed concurrently at different stages.
"""

import os
import re
import time
import json
import copy
import hashlib
import traceback
import numpy as np
import cv2 as cv
import torch

from planner.core import quaternion
from planner.core.action import HumanAction
from planner.core.vlm_client import VLMClient, Logger
from planner.utils.geometry_utils import get_convex_hull, get_instance_distance, get_instance_position_distance
from planner.utils.format_utils import (
    format_objects, format_objects_labels, format_state,
    format_coordinate, format_coordinate_string, get_volume
)
from planner.utils.image_utils import get_base64
from planner.prompts import (
    STEP1_SYSTEM1, STEP1_USER1, STEP1_ASSISTANT1, STEP1_USER2,
    SYSTEM1, USER1, ASSISTANT1, SURFACE_TO_JOINT,
    USER_FAILED_OBJECT, USER_FAILED_MOVE, USER_FAILED_NOT_CLOSE_ENOUGH
)
from planner.planning.stages.base import (
    ActionContext, StageInput, StageOutput, StageType, PipelineCoordinator
)


class PlanningBackend:
    """Backend process for LLM-based behavior planning.

    This class manages the main planning loop and coordinates
    the three-stage action planning pipeline.
    """

    def __init__(self, env_id, request_queue, result_queue):
        """Initialize planning backend.

        Args:
            env_id: Environment identifier
            request_queue: Queue for receiving requests
            result_queue: Queue for sending results
        """
        self.env_id = env_id
        self.request_queue = request_queue
        self.result_queue = result_queue

        self.vlm_client = VLMClient(env_id)
        self.logger = Logger(env_id)

        # Pipeline coordinator for action processing
        self.pipeline = PipelineCoordinator(env_id, self.vlm_client, self.logger)

        # Action ID counter for unique identification
        self._action_id_counter = 0

    def run(self):
        """Main planning loop entry point."""
        try:
            self._main_loop()
        except Exception as e:
            print("发生异常：", e)
            traceback.print_exc()
            self.result_queue.put({"type": "fail"})

    def _main_loop(self):
        """Main planning loop implementation."""
        requests = []

        # Wait for init request
        objects, parents, origin_prompt = self._wait_for_init(requests)

        # Initialize hash and buffer
        hash_code = self._compute_hash(origin_prompt, objects)
        self.vlm_client.initialize_buffer(hash_code)

        # Filter deprecated objects
        objects, parents = self._filter_objects(objects, parents)

        # Build object dict
        main_objects = list(set(parents.values()))
        object_dict = {name: [] for name in main_objects}
        for name, parent in parents.items():
            if name != parent:
                object_dict[parent].append(name)

        # Load labels
        image_dir = "planner/image"
        parent_labels = np.load(os.path.join(image_dir, "labels.npy"), allow_pickle=True)[None][0]
        object_string = format_objects_labels(objects, object_dict, parent_labels)

        # Add volume to instances
        for instance in objects.values():
            if "bbox" in instance:
                instance["volume"] = get_volume(instance)

        # Refine prompt
        prompt = self._refine_prompt(origin_prompt, object_string, image_dir)

        # Initialize state
        state = {
            "left_slot": None,
            "right_slot": None,
            "position": [0, 0],
            "position_name": "origin"
        }
        action_dict = {}
        last_state_position = [0, 0]

        # Build system message
        system1 = (SYSTEM1
                   .replace("origin_prompt_placeholder", origin_prompt)
                   .replace("object_placeholder", object_string)
                   .replace("prompt_placeholder", prompt))

        messages = [
            {"role": "system", "content": system1},
            {"role": "user", "content": USER1},
            {"role": "assistant", "content": ASSISTANT1},
        ]

        self.logger.print_role("System")
        self.logger.write(system1)
        self.logger.write()
        self.logger.print_role("User")
        self.logger.write(USER1)
        self.logger.write()
        self.logger.print_role("Assistant")
        self.logger.write(ASSISTANT1)
        self.logger.write()

        message = "Command execution successful. Current state: "
        pseudo_take = False
        pseudo_place = False

        # Main command loop
        while True:
            state_string = format_state(state, action_dict.keys())
            messages.append({"role": "user", "content": f"{message}\n{state_string}"})

            self.logger.print_role("User")
            self.logger.write(f"{message}\n{state_string}")
            self.logger.write()

            response = self.vlm_client.chat(messages)

            self.logger.print_role("Assistant")
            self.logger.write(response)
            self.logger.write()

            messages.append({"role": "assistant", "content": response})
            command = response

            # Execute command
            if "start" in command:
                message, pseudo_take, pseudo_place = self._handle_start(
                    command, objects, parents, state, action_dict,
                    object_dict, last_state_position
                )
                if action_dict:
                    last_state_position = state["position"]
            elif "stop" in command:
                message = self._handle_stop(
                    command, requests, action_dict, state,
                    pseudo_take, pseudo_place
                )
            elif "end" in command:
                self._handle_end(requests, action_dict)
                return
            else:
                if "(" in command and ")" in command:
                    command = command[:command.index("(")]
                message = f"Invalid command: '{command}', please follow the format requirements and provide a valid command (start, stop or end)"

    # =========================================================================
    # Initialization methods
    # =========================================================================

    def _wait_for_init(self, requests):
        """Wait for initialization request."""
        while True:
            time.sleep(0.1)
            if len(requests) == 0:
                if self.request_queue.empty():
                    continue
                request = self.request_queue.get()
                requests.append(request)

            while not self.request_queue.empty():
                request = self.request_queue.get()
                requests.append(request)

            for request_index, request in enumerate(requests):
                if request["type"] == "init":
                    origin_prompt, objects, parents = request["content"]
                    requests.pop(request_index)
                    return objects, parents, origin_prompt

    def _compute_hash(self, origin_prompt, objects):
        """Compute environment hash code."""
        code = ""
        code += origin_prompt
        start_point = torch.load(f"planner/capture/start_point.pt", weights_only=False)
        code += str(start_point.tolist())
        object_names = sorted(objects.keys())
        code += str(object_names)
        code = hashlib.sha256(code.encode()).hexdigest()
        return str(code)

    def _filter_objects(self, objects, parents):
        """Filter deprecated objects based on height and visibility."""
        deprecated_objects = set()
        for name, instance in objects.items():
            object_position = (
                np.array(instance["position"], dtype=np.float32) +
                np.array(instance["bbox"], dtype=np.float32).mean(0)
            )
            object_height = object_position[2]
            if parents[name] == name:
                continue

            if object_height < 0.2 or object_height > 1.8:
                deprecated_objects.add(name)
                continue

            if "cabinet" not in parents[name]:
                continue

            target_instance = objects[name]
            parent_instance = objects[parents[name]]
            target_bbox = (
                np.array(target_instance["bbox"], dtype=np.float32) +
                np.array(target_instance["position"], dtype=np.float32)
            )
            parent_bbox = (
                np.array(parent_instance["bbox"], dtype=np.float32) +
                np.array(parent_instance["position"], dtype=np.float32)
            )
            target_bbox_2d = get_convex_hull(target_bbox[..., :2])
            if target_bbox_2d is None:
                continue
            target_bbox_2d = target_bbox_2d.buffer(0)
            parent_bbox_2d = get_convex_hull(parent_bbox[..., :2])
            if parent_bbox_2d is None:
                continue
            parent_bbox_2d = parent_bbox_2d.buffer(0)
            target_height = target_bbox[..., 2].mean()
            parent_max_height = parent_bbox[..., 2].max()
            overlap = target_bbox_2d.intersection(parent_bbox_2d)
            if overlap.is_empty or not hasattr(overlap, "area") or overlap.area == 0:
                continue
            overlap = overlap.area / min(target_bbox_2d.area, parent_bbox_2d.area)
            if overlap > 0.75 and target_height < parent_max_height:
                deprecated_objects.add(name)

        for name in deprecated_objects:
            del objects[name]
            del parents[name]

        parents["origin"] = "origin"
        objects["origin"] = {"position": [0, 0, 0]}

        return objects, parents

    def _refine_prompt(self, origin_prompt, object_string, image_dir):
        """Refine user prompt using LLM."""
        images = os.listdir(image_dir)
        images = [os.path.join(image_dir, image) for image in images if image.endswith(".png")]
        images = [cv.imread(i) for i in images]

        height = images[0].shape[0]
        black_gap = np.zeros((height, 16, 3), dtype=np.uint8)

        padded_images = []
        for idx, img in enumerate(images):
            padded_images.append(img)
            if idx < len(images) - 1:
                padded_images.append(black_gap)

        image = np.hstack(padded_images)
        h, w = image.shape[:2]
        image = cv.resize(image, (int(w * 512 / h), 512))

        step1_user2 = (STEP1_USER2
                       .replace("prompt_placeholder", origin_prompt)
                       .replace("object_placeholder", object_string))

        self.logger.print_role("System")
        self.logger.write(STEP1_SYSTEM1)
        self.logger.write()
        self.logger.print_role("User")
        self.logger.write(STEP1_USER1)
        self.logger.write()
        self.logger.print_role("Assistant")
        self.logger.write(STEP1_ASSISTANT1)
        self.logger.write()
        self.logger.print_role("User")
        self.logger.write(step1_user2)
        self.logger.write()

        response = self.vlm_client.chat([
            {"role": "system", "content": STEP1_SYSTEM1},
            {"role": "user", "content": STEP1_USER1},
            {"role": "assistant", "content": STEP1_ASSISTANT1},
            {"role": "user", "content": step1_user2}
        ])

        self.logger.print_role("Assistant")
        self.logger.write(response)
        self.logger.write()

        return response

    # =========================================================================
    # Command handlers
    # =========================================================================

    def _handle_start(self, command, objects, parents, state, action_dict,
                     object_dict, last_state_position):
        """Handle start command using the three-stage pipeline with implicit movement.

        For close-range actions, this sends a preliminary action to start navigation
        immediately, then continues with full reasoning and sends an update.
        """
        action_string = command[command.index("(") + 1:command.index(")")]
        action_string = action_string.replace('"', '')

        # Create action context for pipeline
        self._action_id_counter += 1
        context = ActionContext(
            action_id=f"action_{self._action_id_counter}",
            action_string=action_string
        )

        # =====================================================================
        # QUICK TARGET EXTRACTION - Send preliminary action BEFORE VLM calls
        # =====================================================================
        # Use HumanAction to quickly extract targets using regex (no VLM)
        quick_action = HumanAction(action_string, objects)
        quick_target = None
        if len(quick_action.targets) == 1:
            quick_target = quick_action.targets[0]
        elif len(quick_action.targets) > 1:
            # Multiple targets - use first one for preliminary navigation
            # Stage 1 will determine the actual target later
            quick_target = quick_action.targets[0]

        # Send preliminary navigation action immediately if we have a target
        sent_preliminary = False
        if quick_target and quick_target in objects:
            preliminary_action = self._create_quick_preliminary_action(
                action_string, quick_target
            )
            self.result_queue.put(preliminary_action)
            sent_preliminary = True

        # =====================================================================
        # STAGE 1: Full Basic Analysis (includes VLM voting)
        # =====================================================================
        # Create stage input
        stage_input = StageInput(
            context=context,
            objects=objects,
            parents=parents,
            object_dict=object_dict,
            state=state,
            last_state_position=last_state_position,
            action_dict=action_dict
        )

        # Process Stage 1: Basic Analyze
        from planner.planning.stages.basic_analyze import BasicAnalyzeStage
        stage1 = BasicAnalyzeStage(self.env_id, self.vlm_client, self.logger)
        output1 = stage1.process(stage_input)
        context = output1.context

        pseudo_take = context.pseudo_take
        pseudo_place = context.pseudo_place

        # Handle non-interactive behavior (no target)
        if context.target is None or context.target not in objects:
            if context.skip_remaining or (not context.place or not context.at or len(context.at) == 0):
                action = self._context_to_action(context, objects)
                action_dict[context.action_string] = action
                # If we sent a preliminary action for a non-interactive behavior,
                # send update to correct it
                if sent_preliminary:
                    update_dict = action.serialize()
                    update_dict["type"] = "update"
                    self.result_queue.put(update_dict)
                else:
                    self.result_queue.put(action.serialize())
                return "Command execution successful. Current state: ", pseudo_take, pseudo_place

        # Determine navigation target for implicit movement
        # For place actions: navigate to "at" object, not "target"
        if context.place and context.at and len(context.at) > 0:
            nav_target = context.at[0]
        else:
            nav_target = context.target

        # Update state position to navigation target
        parent = parents.get(nav_target, nav_target)
        parent_instance = objects.get(parent, {})
        if parent_instance:
            state["position_name"] = nav_target
            state["position"] = format_coordinate(parent_instance["position"])

        # Validate action (check if current actions allow this new action)
        message = self._validate_action(context, objects, parents, state, action_dict, parent_instance)
        if message:
            return message, False, False

        # Determine target and anchor for capture
        target = nav_target
        parent = parents.get(target, target)
        anchor = self._determine_anchor(parent, action_dict, parents)

        # Wait for capture
        capture_targets = [target, anchor] if target != anchor else [target]
        capture_data = self._wait_for_capture(capture_targets, objects, anchor)
        stage_input.extra['capture_data'] = capture_data

        # Process Stage 2: Pose Reasoning
        stage_input.context = context
        from planner.planning.stages.pose_reasoning import PoseReasoningStage
        stage2 = PoseReasoningStage(self.env_id, self.vlm_client, self.logger)
        output2 = stage2.process(stage_input)
        context = output2.context

        # Process Stage 3: Point Reasoning (if needed)
        if output2.continue_pipeline:
            stage_input.context = context
            from planner.planning.stages.point_reasoning import PointReasoningStage
            stage3 = PointReasoningStage(self.env_id, self.vlm_client, self.logger)
            output3 = stage3.process(stage_input)
            context = output3.context

        # Convert context to action and store
        action = self._context_to_action(context, objects)
        action_dict[context.action_string] = action

        # Send update if we sent a preliminary action, otherwise send full action
        if sent_preliminary:
            update_dict = action.serialize()
            update_dict["type"] = "update"
            self.result_queue.put(update_dict)
        else:
            self.result_queue.put(action.serialize())

        return "Command execution successful. Current state: ", pseudo_take, pseudo_place

    def _create_preliminary_action(self, context, nav_target):
        """Create a preliminary action for immediate navigation.

        This action has position=None which tells update_handler to
        navigate to the object's bbox center.
        """
        return {
            "type": "action",
            "action": context.action_string,
            "target": nav_target,
            "at": context.at,
            "by": context.by,
            "touch": context.touch,
            "long_range": context.long_range,
            "facing": None,
            "take": context.take,
            "place": context.place,
            "position": None,  # None triggers navigation to object bbox
            "contact_points": context.contact_points or [],
            "contact_targets": []
        }

    def _create_quick_preliminary_action(self, action_string, target):
        """Create a quick preliminary action for immediate navigation.

        This is called BEFORE VLM processing, so we don't know touch/take/place yet.
        We assume it's an interactive action and let the update correct it if needed.
        """
        return {
            "type": "action",
            "action": action_string,
            "target": target,
            "at": None,
            "by": None,
            "touch": True,  # Assume interactive - will be corrected by update
            "long_range": False,
            "facing": None,
            "take": False,
            "place": False,
            "position": None,  # None triggers navigation to object bbox
            "contact_points": [],
            "contact_targets": []
        }

    def _handle_stop(self, command, requests, action_dict, state, pseudo_take, pseudo_place):
        """Handle stop command."""
        action_string = command[command.index("(") + 1:command.index(")")]
        action_string = action_string.replace('"', '')
        action_failed = False

        # Check if this action is tracked
        is_tracked_action = action_string in action_dict

        while True:
            time.sleep(0.1)
            if len(requests) == 0:
                if self.request_queue.empty():
                    continue
                request = self.request_queue.get()
                requests.append(request)

            while not self.request_queue.empty():
                request = self.request_queue.get()
                requests.append(request)

            has_done_request = False
            for request_index, request in enumerate(requests):
                if request["type"] == "done" and request["content"] == action_string:
                    has_done_request = True
                    if is_tracked_action:
                        action = action_dict[action_string]
                        if action.touch and "exceed" in request:
                            action_failed = True
                    break

            if not has_done_request:
                continue

            self.result_queue.put({"type": "stop", "action": action_string})
            break

        if not is_tracked_action:
            print(f"Stopping untracked action: {action_string}")
            return "Command execution successful. Current state: "

        action = action_dict.pop(action_string)

        if action_failed:
            return "Action stopped, but the objective of the action do not achieved. You can retry the action or select another target, or just skip this action."

        if action.take:
            if "right_hand" in action.contact_points:
                state["right_slot"] = action.target
            if "left_hand" in action.contact_points:
                state["left_slot"] = action.target

        if pseudo_take and "book" in action.target:
            if "right_hand" in action.contact_points:
                state["right_slot"] = "book"
            if "left_hand" in action.contact_points:
                state["left_slot"] = "book"

        if action.place:
            if state["right_slot"] == action.target:
                state["right_slot"] = None
            if state["left_slot"] == action.target:
                state["left_slot"] = None

        if pseudo_place and "book" in action.target:
            if state["right_slot"] == action.target:
                state["right_slot"] = None
            if state["left_slot"] == action.target:
                state["left_slot"] = None

        print("Stoping Action:", action_string)
        return "Action stoped and completed successfully."

    def _handle_end(self, requests, action_dict):
        """Handle end command."""
        while len(action_dict) > 0:
            while True:
                time.sleep(0.1)
                if len(requests) == 0:
                    if self.request_queue.empty():
                        continue
                    request = self.request_queue.get()
                    requests.append(request)

                while not self.request_queue.empty():
                    request = self.request_queue.get()
                    requests.append(request)

                has_done_request = None
                for request_index, request in enumerate(requests):
                    if request["type"] == "done" and request["content"] in action_dict:
                        has_done_request = request["content"]
                        break

                if has_done_request is None:
                    continue

                self.result_queue.put({"type": "stop", "action": has_done_request})
                break

            del action_dict[has_done_request]
            print("Stoping Action:", has_done_request)

        self.result_queue.put({"type": "complete"})

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _determine_anchor(self, parent, action_dict, parents):
        """Determine anchor for positioning."""
        anchor = parent
        for current_action in action_dict.values():
            if current_action.place and current_action.at and len(current_action.at) > 0:
                anchor = current_action.at[0]
                anchor = parents.get(anchor, anchor)
            elif current_action.target is not None and not current_action.long_range:
                anchor = parents.get(current_action.target, current_action.target)
        return anchor

    def _wait_for_capture(self, capture_targets, objects, anchor):
        """Wait for capture result."""
        self.result_queue.put({"type": "capture", "target": capture_targets})

        requests = []
        while True:
            time.sleep(0.1)
            if len(requests) == 0:
                if self.request_queue.empty():
                    continue
                request = self.request_queue.get()
                requests.append(request)

            while not self.request_queue.empty():
                request = self.request_queue.get()
                requests.append(request)

            for request_index, request in enumerate(requests):
                if request["type"] == "capture":
                    capture_data = request["content"]
                    requests.pop(request_index)
                    return capture_data

    def _validate_action(self, context, objects, parents, state, action_dict, parent_instance):
        """Validate action constraints."""
        close_objects = []
        close_positions = []

        for conducting_action in action_dict.values():
            if conducting_action.long_range:
                if conducting_action.glb_position is None and conducting_action.position is None:
                    continue
                conducting_position = conducting_action.position if conducting_action.glb_position is None else conducting_action.glb_position
                conducting_position = np.array(conducting_position, dtype=np.float32)
                close_positions.append(conducting_position)
            if conducting_action.target in objects:
                close_objects.append(conducting_action.target)
            if conducting_action.at:
                close_objects += conducting_action.at
            if conducting_action.by:
                close_objects += conducting_action.by

        if not context.long_range:
            if context.target in objects:
                close_objects.append(context.target)
            if context.at:
                close_objects += context.at
            if context.by:
                close_objects += [b for b in context.by if b in objects]

        for close_object in close_objects:
            close_instance = objects.get(close_object, {})
            if not close_instance:
                continue
            distance = get_instance_distance(close_instance, parent_instance)
            if distance > 0.5:
                return (USER_FAILED_NOT_CLOSE_ENOUGH
                       .replace("{self.position}", format_coordinate_string(state["position"]))
                       .replace("{close_object}", close_object)
                       .replace("{target_position}", format_coordinate_string(close_instance["position"]))
                       .replace("{action_string}", context.action_string))

        for close_position in close_positions:
            distance = get_instance_position_distance(parent_instance, close_position)
            if distance > 0.5:
                return (USER_FAILED_NOT_CLOSE_ENOUGH
                       .replace("{self.position}", format_coordinate_string(state["position"]))
                       .replace("{close_object}", "action position")
                       .replace("{target_position}", format_coordinate_string(close_position))
                       .replace("{action_string}", context.action_string))

        return None

    def _context_to_action(self, context: ActionContext, objects) -> HumanAction:
        """Convert ActionContext to HumanAction."""
        action = HumanAction(context.action_string, objects)
        action.target = context.target
        action.at = context.at
        action.by = context.by
        action.touch = context.touch
        action.take = context.take
        action.place = context.place
        action.long_range = context.long_range
        action.contact_points = context.contact_points or []
        action.position = context.position
        action.glb_position = context.glb_position
        action.facing = context.facing
        action.glb_facing = context.glb_facing
        action.position_tag = context.position_tag
        action.tag_direction = context.tag_direction
        action.contact_targets = context.contact_targets or []
        action.glb_contact_targets = context.glb_contact_targets
        return action

    def _is_holding(self, state, target):
        """Check if target is being held."""
        return target == state["left_slot"] or target == state["right_slot"]

"""Main planning backend for LLM-based behavior planning."""
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
from planner.planning.action_parser import ActionParser
from planner.planning.position_handler import PositionHandler
from planner.planning.contact_locator import ContactPointLocator


class PlanningBackend:
    """Backend process for LLM-based behavior planning."""

    def __init__(self, env_id, request_queue, result_queue):
        """Initialize planning backend."""
        self.env_id = env_id
        self.request_queue = request_queue
        self.result_queue = result_queue

        self.vlm_client = VLMClient(env_id)
        self.logger = Logger(env_id)
        self.position_handler = PositionHandler(env_id, self.vlm_client, self.logger)
        self.contact_locator = ContactPointLocator(
            env_id, self.vlm_client, self.logger, self.position_handler
        )

    def run(self):
        """Main planning loop."""
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

        # Initialize action parser
        action_parser = ActionParser(self.vlm_client, self.logger, state)

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
            if "move_to" in command:
                message, pseudo_take, pseudo_place = self._handle_move_to(
                    command, objects, parents, state, action_dict
                )
            elif "start" in command:
                message, pseudo_take, pseudo_place = self._handle_start(
                    command, objects, parents, state, action_dict,
                    action_parser, object_dict, last_state_position
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

    def _handle_move_to(self, command, objects, parents, state, action_dict):
        """Handle move_to command."""
        target = command[command.index("(") + 1:command.rindex(")")]
        target = target.replace('"', '').replace("_", " ")

        if target not in objects or target == "origin":
            message = USER_FAILED_OBJECT.replace("{object_placeholder}", target)
            return message, False, False

        parent = parents[target]
        parent_instance = objects[parent]
        target_instance = objects[target]

        # Check if can move to
        for action_string, action in action_dict.items():
            if self._is_holding(state, action.target):
                continue

            if action.long_range:
                if action.glb_position is None and action.position is None:
                    continue
                action_position = action.position if action.glb_position is None else action.glb_position
                action_position = np.array(action_position, dtype=np.float32)
                distance = get_instance_position_distance(parent_instance, action_position)
            else:
                action_instance = objects[action.target]
                distance = get_instance_distance(parent_instance, action_instance)

            if distance > 0.5:
                message = (USER_FAILED_MOVE
                          .replace("{action_string}", action_string)
                          .replace("{target}", target))
                return message, False, False

        # Update held object positions
        if state["left_slot"] and state["left_slot"] in objects:
            objects[state["left_slot"]]["position"] = parent_instance["position"]
        if state["right_slot"] and state["right_slot"] in objects:
            objects[state["right_slot"]]["position"] = parent_instance["position"]

        state["position_name"] = target
        state["position"] = format_coordinate(parent_instance["position"])

        action = {
            "type": "action",
            "action": f"Walking to {target}",
            "target": target,
            "at": None,
            "by": None,
            "touch": None,
            "long_range": None,
            "facing": None,
            "take": None,
            "place": None,
            "position": None,
            "contact_points": [],
            "contact_targets": []
        }
        self.result_queue.put(action)

        return "Command execution successful. Current state: ", False, False

    def _handle_start(self, command, objects, parents, state, action_dict,
                     action_parser, object_dict, last_state_position):
        """Handle start command."""
        action_string = command[command.index("(") + 1:command.index(")")]
        action_string = action_string.replace('"', '')
        action = HumanAction(action_string, objects)

        # Parse targets
        action = action_parser.parse_targets(action)

        pseudo_take = False
        pseudo_place = False

        if action.target:
            parent = parents.get(action.target, action.target)
            parent_instance = objects.get(parent, {})

            # Build interaction string and state string
            interaction_string = action_parser.build_interaction_string(action)
            state_string = action_parser.build_state_string(state)

            # Compute relative direction
            current_relative_string = action_parser.compute_relative_direction(
                action, objects, action_dict
            )

            # Detect touch attributes
            action = action_parser.detect_touch_attributes(
                action, interaction_string, state_string, current_relative_string
            )

            # Handle place action contact points
            if action.place and (action.contact_points is None or len(action.contact_points) == 0):
                contact_points = []
                if action.target == state["left_slot"]:
                    contact_points.append("left_wrist")
                if action.target == state["right_slot"]:
                    contact_points.append("right_wrist")
                action.contact_points = contact_points

            action.long_range = not action.touch

            # Handle non-takable objects
            if not any([o in action.target for o in ["plant container", "nature shelf trinkets", "box"]]):
                if action.take:
                    action.take = False
                    pseudo_take = True
                    if len(action.contact_points) > 1:
                        action.contact_points = action.contact_points[:1]
                if action.place:
                    action.place = False
                    pseudo_place = True
                    if len(action.contact_points) > 1:
                        action.contact_points = action.contact_points[:1]

            # Handle non-interactive behavior
            if action.target not in objects:
                action.target = None
                if not action.place or not action.at or len(action.at) == 0:
                    action_dict[action.action] = action
                    self.result_queue.put(action.serialize())
                    return "Command execution successful. Current state: ", pseudo_take, pseudo_place

            # Validate action
            message = self._validate_action(action, objects, parents, state, action_dict, parent_instance)
            if message:
                return message, False, False

            # Process action position and contacts
            message = self._process_action_position(
                action, objects, parents, state, action_dict,
                object_dict, last_state_position
            )

        action_dict[action.action] = action

        # Transform to relative coordinates
        self._transform_to_relative(action, objects)

        self.result_queue.put(action.serialize())
        return "Command execution successful. Current state: ", pseudo_take, pseudo_place

    def _validate_action(self, action, objects, parents, state, action_dict, parent_instance):
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

        if not action.long_range:
            if action.target in objects:
                close_objects.append(action.target)
            if action.at:
                close_objects += action.at
            if action.by:
                close_objects += action.by
                remove_by = []
                for by in action.by:
                    if by not in objects:
                        remove_by.append(by)
                for by in remove_by:
                    action.by.remove(by)

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
                       .replace("{action_string}", action.action))

        for close_position in close_positions:
            distance = get_instance_position_distance(parent_instance, close_position)
            if distance > 0.5:
                return (USER_FAILED_NOT_CLOSE_ENOUGH
                       .replace("{self.position}", format_coordinate_string(state["position"]))
                       .replace("{close_object}", "action position")
                       .replace("{target_position}", format_coordinate_string(close_position))
                       .replace("{action_string}", action.action))

        return None

    def _process_action_position(self, action, objects, parents, state, action_dict,
                                 object_dict, last_state_position):
        """Process action position and contact points."""
        if action.place and action.at is not None and len(action.at) > 0:
            target = action.at[0]
        else:
            target = action.target

        parent = parents.get(target, target)

        # Determine anchor
        is_previous_anchor = False
        anchor = parent
        for current_action in action_dict.values():
            if current_action.place and current_action.at is not None and len(current_action.at) > 0:
                anchor = current_action.at[0]
                anchor = parents.get(anchor, anchor)
                is_previous_anchor = True
            elif current_action.target is not None and not current_action.long_range:
                anchor = parents.get(current_action.target, current_action.target)
                is_previous_anchor = True

        # Wait for capture
        capture_targets = [target, anchor] if target != anchor else [target]
        self._wait_for_capture(capture_targets, objects, anchor)

        # Build merged action string
        merged_action_strings = list(action_dict.keys()) + [action.action]
        if len(merged_action_strings) == 1:
            merged_action_string = merged_action_strings[0]
        elif len(merged_action_strings) == 2:
            merged_action_string = f"{merged_action_strings[0]} and {merged_action_strings[1]}"
        else:
            merged_action_string = ", ".join(merged_action_strings[:-1]) + f", and {merged_action_strings[-1]}"
        merged_action_type = re.sub(r'\d', '', merged_action_string)

        # Handle position specification
        if action.long_range:
            action = self.position_handler.handle_long_range_position(
                action, target, objects, object_dict, action_dict, state
            )
        else:
            action, target_directions, anchor_directions, anchor_position_tag = (
                self.position_handler.handle_close_range_position(
                    action, target, anchor, objects, parents,
                    action_dict, merged_action_type, last_state_position
                )
            )

            # Handle contact points
            if action.touch or action.place:
                target_instance = objects[target]
                if target_instance.get("volume", 0) > 0.125 and not action.take:
                    action.contact_targets = self.contact_locator.locate_contact_points(
                        action, target, anchor, objects, action.tag_direction,
                        re.sub(r'\d', '', anchor), merged_action_type, action.position_tag
                    )
                else:
                    action.contact_targets = self.contact_locator.locate_simple_contact_targets(
                        action, target_instance
                    )

            # Compute facing
            self._compute_facing(action, objects, anchor, action_dict)

        return "Command execution successful. Current state: "

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
                    if len(capture_targets) == 1:
                        target_instance_update = request["content"][0]
                    else:
                        target_instance_update, parent_instance_update = request["content"]

                    objects[capture_targets[0]].update(target_instance_update)
                    if len(capture_targets) > 1:
                        objects[anchor].update(parent_instance_update)

                    requests.pop(request_index)
                    return

    def _compute_facing(self, action, objects, anchor, action_dict):
        """Compute facing direction for action."""
        anchor_contact_points = []
        for current_action in action_dict.values():
            if current_action.contact_points is not None:
                anchor_contact_points.extend(current_action.contact_points)

        if "pelvis" in [SURFACE_TO_JOINT.get(j, j) for j in action.contact_points]:
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

    def _transform_to_relative(self, action, objects):
        """Transform action coordinates to relative space."""
        if action.place and action.at is not None and len(action.at) > 0:
            reposition_target = action.at[0]
        elif action.target is not None:
            reposition_target = action.target
        else:
            reposition_target = None

        if reposition_target is not None and action.position is not None:
            action.glb_position = copy.deepcopy(action.position)
            act_position = np.array(action.position).tolist()
            # Only add z coordinate if position is 2D
            if len(act_position) == 2:
                act_position += [0]
            act_position = np.array(act_position)[:3]  # Ensure only 3 elements

            tar_instance = objects[reposition_target]
            tar_position = np.array(tar_instance["position"])[:3]  # Ensure only 3 elements

            tar_rotation = np.array(tar_instance["rotation"])
            tar_rotation = tar_rotation[None]
            tar_rotation = quaternion.qinv_np(tar_rotation)

            rel_position = act_position - tar_position
            rel_position = rel_position[None]
            rel_position = quaternion.qrot_np(tar_rotation, rel_position)
            rel_position = rel_position[0][:2].tolist()
            action.position = rel_position

            if action.contact_targets is not None and len(action.contact_targets) > 0:
                action.glb_contact_targets = copy.deepcopy(action.contact_targets)
                con_position = np.array(action.contact_targets)
                con_position = con_position - tar_position
                tar_rota_con = np.broadcast_to(tar_rotation, (con_position.shape[0], 4))
                con_position = quaternion.qrot_np(tar_rota_con, con_position)
                con_position = con_position.tolist()
                action.contact_targets = con_position

            if action.facing is not None:
                action.glb_facing = copy.deepcopy(action.facing)
                fac_direction = [np.cos(action.facing), np.sin(action.facing), 0]
                fac_direction = np.array(fac_direction)[None]
                fac_direction = quaternion.qrot_np(tar_rotation, fac_direction)
                fac_direction = fac_direction[0, :2]
                fac_direction = float(np.arctan2(fac_direction[1], fac_direction[0]))
                action.facing = fac_direction

    def _handle_stop(self, command, requests, action_dict, state, pseudo_take, pseudo_place):
        """Handle stop command."""
        action_string = command[command.index("(") + 1:command.index(")")]
        action_string = action_string.replace('"', '')
        action_failed = False

        # Check if this action is in our tracking dict
        # Initial state actions like "Standing on the origin." may not be tracked
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

        # If action wasn't tracked (like initial state), just return success
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

    def _is_holding(self, state, target):
        """Check if target is being held."""
        return target == state["left_slot"] or target == state["right_slot"]

"""Action parsing and touch detection handlers."""
import re
import numpy as np
from collections import Counter
from concurrent import futures as ftr

from planner.prompts import (
    SYSTEM_SPECIFY_TOUCH, USER1_SPECIFY_TOUCH, ASSISTANT1_SPECIFY_TOUCH,
    USER2_SPECIFY_TOUCH, USER_SPECIFY_TARGET, SURFACE_TO_JOINT
)


class ActionParser:
    """Parser for action attributes like touch, take, place."""

    def __init__(self, vlm_client, logger, state):
        """Initialize action parser."""
        self.vlm_client = vlm_client
        self.logger = logger
        self.state = state

    def parse_targets(self, action):
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

    def detect_touch_attributes(self, action, interaction_string, state_string,
                               current_relative_string=None):
        """Detect touch, take, place attributes using voting."""
        if current_relative_string is not None:
            state_string = state_string + "\n" + current_relative_string

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
                        ("left_wrist" if self.state["left_slot"] == joint
                         else "right_wrist" if self.state["right_slot"] == joint
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

        action.touch = touch_buffer.count(True) >= 3
        action.take = take_buffer.count(True) >= 3
        action.long_range = range_buffer.count(True) >= 3
        action.place = place_buffer.count(True) >= 3

        if not action.touch:
            action.contact_points = []
        else:
            filtered_points = [
                cp for idx, cp in enumerate(contact_point_buffer)
                if idx < len(touch_buffer) and touch_buffer[idx] and cp
            ]
            counter = Counter(tuple(sublist) for sublist in filtered_points)
            most_common = counter.most_common(1)[0][0]
            action.contact_points = list(most_common)

        return action

    def build_interaction_string(self, action):
        """Build interaction string for touch detection."""
        interaction = []
        interaction.append("Action: " + action.marked_action)
        interaction.append("Target: " + action.target)
        if action.at:
            interaction.append("At: " + ", ".join(action.at))
        if action.by:
            interaction.append("By: " + ", ".join(action.by))
        return "\n".join(interaction)

    def build_state_string(self, state):
        """Build state string for touch detection."""
        left = state["left_slot"]
        right = state["right_slot"]
        if left and right:
            return f"You are holding {left} in your left hand and {right} in your right hand."
        elif left:
            return f"You are holding {left} in your left hand. Your right hand is available."
        elif right:
            return f"You are holding {right} in your right hand. Your left hand is available."
        else:
            return "You are currently holding nothing in your hands."

    def compute_relative_direction(self, action, objects, action_dict):
        """Compute relative direction string based on current action state."""
        current_action_position = None
        current_action_facing = None

        for current_action_string, current_action in action_dict.items():
            current_contact_points = [
                SURFACE_TO_JOINT[p] for p in current_action.contact_points
            ] if current_action.contact_points is not None else []

            if len(current_contact_points) > 0 and "pelvis" in current_contact_points:
                current_action_position = current_action.glb_contact_targets[
                    current_contact_points.index("pelvis")
                ][:2]
            elif current_action.glb_position is not None:
                current_action_position = current_action.glb_position

            if current_action.glb_facing is not None:
                current_action_facing = current_action.glb_facing

        if current_action_facing is None or current_action_position is None:
            return None

        target_instance = objects[action.target]
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
        return f"The {action.target} is in the {direction_strings[direction_index]} of you."

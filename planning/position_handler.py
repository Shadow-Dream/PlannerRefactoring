"""Position specification handlers for both long-range and close-range interactions."""
import os
import re
import time
import numpy as np
import cv2 as cv
import torch
import requests as req

from planner.utils.image_utils import (
    get_base64, resize_foreground, resize_foreground_coord_mask,
    get_marker, get_normal_map, get_height_map
)
from planner.prompts import (
    USER1_DIRECTION_LONGRANGE, USER1_SPECIFY_DISTANCE_LONGRANGE,
    USER1_SPECIFY_FACING_LONGRANGE, SYSTEM1_SPECIFY_POSITION,
    USER1_SPECIFY_POSITION_TARGET, USER1_SPECIFY_POSITION_MARKER,
    USER1_SPECIFY_POSITION_ACTION, USER_POSITIONING_INITIAL,
    USER_POSITIONING_INITIAL_PLACE, USER_POSITIONING_LOCATE,
    USER_POSITIONING_LOCATE_PLACE
)
from planner.utils.format_utils import format_objects


class PositionHandler:
    """Handler for position specification via VLM."""

    def __init__(self, env_id, vlm_client, logger):
        """Initialize position handler."""
        self.env_id = env_id
        self.vlm_client = vlm_client
        self.logger = logger
        self.capture_backend = "http://127.0.0.1:14000"

    def get_remote_file(self, path):
        """Download file from capture backend or use local file if available."""
        # Check if file already exists locally (for mock testing)
        # Handle paths with env_id prefix like "0/bed1_rgb.png"
        path_parts = path.split("/")
        if len(path_parts) > 1 and path_parts[0].isdigit():
            # Remove env_id prefix for local lookup
            filename = "/".join(path_parts[1:])
        else:
            filename = path

        local_paths = [
            os.path.join("planner/capture", filename),
            os.path.join("capture", filename),
            os.path.join("planner/capture", path),
            os.path.join("capture", path),
        ]
        for local_path in local_paths:
            if os.path.exists(local_path):
                # Copy to expected location if needed
                target_path = os.path.join("capture", path)
                if not os.path.exists(target_path):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    import shutil
                    shutil.copy(local_path, target_path)
                return

        # Try to download from remote server
        request_path = path.replace(" ", "<space>").replace("/", "<bar>")
        try:
            response = req.get(f"{self.capture_backend}/download/{request_path}")
            if response.status_code == 200:
                file_path = os.path.join("capture", path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            # If download fails, check if we can find file locally
            print(f"Warning: Could not download {path}: {e}")

    def handle_long_range_position(self, action, target, objects, object_dict,
                                   action_dict, state):
        """Handle position specification for long-range interactions."""
        target_instance = objects[target]
        bev_map_path = f"planner/capture/bev_map.png"
        bev_map_buffer_path = f"planner/capture/bev_map_buffer.png"

        while not os.path.exists(bev_map_buffer_path):
            time.sleep(0.1)

        bev_map = cv.imread(bev_map_path)
        bev_map_buffer = cv.imread(bev_map_buffer_path)[..., 0]
        bev_map = np.pad(bev_map, ((16, 16), (16, 16), (0, 0)), mode='constant', constant_values=0)
        bev_map_buffer = np.pad(bev_map_buffer, ((16, 16), (16, 16)), mode='constant', constant_values=0)

        min_x, min_y, max_axis, resolution = torch.load(f"planner/capture/bev_map_state.pt", weights_only=False)

        def points_to_pixel(points):
            points = (points - np.array([min_x, min_y], dtype=np.float32)) / max_axis * resolution
            return points.astype(np.int32) + 16

        def pixel_to_points(points):
            points = points.astype(np.float32)
            points = points / resolution * max_axis
            points = points + np.array([min_x, min_y], dtype=np.float32)
            return points

        def in_range(posi):
            x, y = points_to_pixel(posi)
            if x >= 0 and y >= 0 and y < bev_map.shape[0] and x < bev_map.shape[1]:
                return True
            return False

        tar_posi = (
            np.array(target_instance["position"]) +
            np.array(target_instance["bbox"]).mean(0)
        )[:2]

        m_object_string = format_objects(objects, object_dict, True)

        # Draw objects on map
        mi = 1
        for major_name in object_dict:
            if major_name == "origin":
                continue
            major_instance = objects[major_name]
            major_posi = (
                np.array(major_instance["position"]) +
                np.array(major_instance["bbox"]).mean(0)
            )[:2]
            x, y = points_to_pixel(major_posi)
            mi += 1
            cv.circle(bev_map, (x, y), 5, (255, 0, 0), -1)
            cv.putText(bev_map, str(mi), (x + 5, y), cv.FONT_HERSHEY_SIMPLEX,
                      0.5, (255, 0, 0), thickness=1, lineType=cv.LINE_AA)

        x, y = points_to_pixel(tar_posi)
        cv.circle(bev_map, (x, y), 10, (0, 0, 255), -1)

        bev_map_object_direction = bev_map + 0
        bev_map_distance = bev_map + 0
        bev_map_self_direction = bev_map + 0

        bev_map_object_direction, valid_labels = self._draw_directional_arrows(
            bev_map_object_direction, (x, y), 50
        )

        stas_instance = objects[state["position_name"]]
        stas_posi = (
            np.array(stas_instance["position"]) +
            np.array(stas_instance["bbox"]).mean(0)
        )[:2]

        action_current_string = ", ".join(action_dict)
        target_string = target

        # Build arrow string
        arrow_string = self._build_arrow_string(target_instance, valid_labels, target_string)

        x_pos, y_pos = stas_posi
        position_current_string = f"({x_pos:.1f}, {y_pos:.1f})"
        x_tar, y_tar = tar_posi
        position_target_string = f"({x_tar:.1f}, {y_tar:.1f})"

        if np.linalg.norm(tar_posi - stas_posi) < 0.5:
            position_current_string += ", which is close to the target object"
        else:
            x, y = points_to_pixel(stas_posi)
            cv.circle(bev_map_object_direction, (x, y), 7, (0, 255, 0), -1)

        cv.imwrite(f"logs/debug_bev_map_object_direction_{self.env_id}.png", bev_map_object_direction)

        if action_current_string == "":
            action_current_string = "doing nothing"

        # Get direction from LLM
        tar_dirc = self._get_direction_from_llm(
            target_string, m_object_string, arrow_string,
            action_current_string, action.action,
            position_target_string, position_current_string,
            bev_map_object_direction
        )

        if tar_dirc is None:
            action.position = stas_posi.tolist()
        else:
            action.position = self._get_position_from_llm(
                tar_dirc, tar_posi, bev_map_buffer, bev_map_distance,
                target_string, m_object_string, action.action,
                points_to_pixel, pixel_to_points, in_range
            )

            x, y = points_to_pixel(np.array(action.position))
            cv.circle(bev_map_self_direction, (x, y), 10, (0, 255, 0), -1)
            bev_map_self_direction, _ = self._draw_directional_arrows(
                bev_map_self_direction, (x, y), 50, (0, 255, 0)
            )
            cv.imwrite(f"logs/debug_bev_map_self_direction_{self.env_id}.png", bev_map_self_direction)

            # Get facing direction
            action.facing = self._get_facing_from_llm(
                action.position, tar_posi, target_string, m_object_string,
                action.action, bev_map_self_direction
            )

        return action

    def _draw_directional_arrows(self, image, center, length=50,
                                 color=(0, 0, 255), thickness=2, font_scale=0.5):
        """Draw eight directional arrows on image."""
        cx, cy = center
        h, w = image.shape[:2]

        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        valid_labels = []

        for idx, (dx, dy) in enumerate(directions):
            end_x = int(cx + dx * length)
            end_y = int(cy + dy * length)
            cv.arrowedLine(image, (cx, cy), (end_x, end_y), color, thickness, tipLength=0.2)

            label_x = int(end_x + dx * 10)
            label_y = int(end_y + dy * 10)

            if 16 <= label_x < w - 16 and 16 <= label_y < h - 16:
                cv.putText(image, str(idx + 1), (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX,
                          font_scale, color, thickness=1, lineType=cv.LINE_AA)
                valid_labels.append(idx + 1)

        return image, valid_labels

    def _build_arrow_string(self, target_instance, valid_labels, target_string):
        """Build arrow description string based on target orientation."""
        if target_instance["front"] is not None:
            front_index = (target_instance["front"] // 45) % 8
            target_directions = [
                "front", "front-left", "left", "back-left",
                "back", "back-right", "right", "front-right"
            ]
            target_directions = {
                ((front_index + i) % 8 + 1): direction
                for i, direction in enumerate(target_directions)
                if ((front_index + i) % 8 + 1) in valid_labels
            }
            return "\n".join([
                f"- {k}: {v} of {target_string}"
                for k, v in target_directions.items()
            ])
        elif target_instance["end"] is not None:
            end_index = (target_instance["end"] // 45) % 8
            target_directions = [
                "end", "corner", "side", "corner",
                "end", "corner", "side", "corner"
            ]
            target_directions = {
                ((end_index + i) % 8 + 1): direction
                for i, direction in enumerate(target_directions)
                if ((end_index + i) % 8 + 1) in valid_labels
            }
            return "\n".join([
                f"- {k}: {v} of {target_string}"
                for k, v in target_directions.items()
            ])
        else:
            return "Arrow labels: 1, 2, 3, 4, 5, 6, 7, 8"

    def _get_direction_from_llm(self, target, objects, arrows, action_current,
                                action, position_target, position_current, image):
        """Get direction choice from LLM."""
        user1 = (USER1_DIRECTION_LONGRANGE
                 .replace("{target}", target)
                 .replace("{objects}", objects)
                 .replace("{arrows}", arrows)
                 .replace("{action_current}", action_current)
                 .replace("{action}", action)
                 .replace("{position_target}", position_target)
                 .replace("{position_current}", position_current))

        self.logger.print_role("USER")
        self.logger.write(user1)

        response = self.vlm_client.chat([{"role": "user", "content": user1}])
        self.logger.print_role("Assistant")
        self.logger.write(response)

        try:
            response = response.split(">>>")[1].split("<<<")[0].strip()
            response = response.replace(target, "")
            response = [i for i in range(8) if str(i + 1) in response][0]
            directions = [
                (1, 0), (1, 1), (0, 1), (-1, 1),
                (-1, 0), (-1, -1), (0, -1), (1, -1)
            ]
            tar_dirc = np.array(directions[response]).astype(float)
            tar_dirc /= np.linalg.norm(tar_dirc)
            return tar_dirc
        except:
            return None

    def _get_position_from_llm(self, tar_dirc, tar_posi, bev_map_buffer,
                               bev_map_distance, target_string, m_object_string,
                               action_string, points_to_pixel, pixel_to_points, in_range):
        """Get specific position from LLM."""
        tar_dis_start = 1
        cur_act_posi = tar_posi + tar_dirc
        dis_label_positions = {}
        label_strings = ""

        while in_range(cur_act_posi):
            x, y = points_to_pixel(cur_act_posi)
            if bev_map_buffer[y, x] > 0:
                cv.circle(bev_map_distance, (x, y), 5, (0, 255, 0), -1)
                dis_label = len(dis_label_positions) + 1
                dis_label_positions[dis_label] = cur_act_posi + 0
                cv.putText(bev_map_distance, str(dis_label), (x + 5, y),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                label_strings += f"- {dis_label}: {tar_dis_start:.2f}\n"
            cur_act_posi = cur_act_posi + tar_dirc
            tar_dis_start += 1

        cv.imwrite(f"logs/debug_map_{self.env_id}.png", bev_map_distance)

        user1 = (USER1_SPECIFY_DISTANCE_LONGRANGE
                 .replace("{target}", target_string)
                 .replace("{objects}", m_object_string)
                 .replace("{distances}", label_strings)
                 .replace("{action}", action_string))

        self.logger.print_role("USER")
        self.logger.write(user1)

        for _ in range(3):
            response = self.vlm_client.chat([{"role": "user", "content": user1}])
            self.logger.print_role("Assistant")
            self.logger.write(response)
            try:
                response = response.split(">>>")[1].split("<<<")[0].strip()
                response_int = None
                for ri in dis_label_positions.keys():
                    if str(ri) in response:
                        response_int = ri
                        break
                return dis_label_positions[response_int].tolist()
            except:
                time.sleep(1)
                continue

        return tar_posi.tolist()

    def _get_facing_from_llm(self, act_position, tar_posi, target_string,
                             m_object_string, action_string, bev_map_self_direction):
        """Get facing direction from LLM."""
        act_posi = np.array(act_position)
        tar_dirc = tar_posi - act_posi[:2]

        directions = np.array([
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]).astype(float)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = (directions * tar_dirc).sum(-1)
        facing_target_string = str(int(directions.argmax() + 1))

        user1 = (USER1_SPECIFY_FACING_LONGRANGE
                 .replace("{target}", target_string)
                 .replace("{objects}", m_object_string)
                 .replace("{label}", facing_target_string)
                 .replace("{action}", action_string))

        self.logger.print_role("USER")
        self.logger.write(user1)

        for _ in range(3):
            response = self.vlm_client.chat([{"role": "user", "content": user1}])
            self.logger.print_role("Assistant")
            self.logger.write(response)
            try:
                response = response.split(">>>")[1].split("<<<")[0].strip()
                response_int = None
                for ri in range(1, 9):
                    if str(ri) in response:
                        response_int = ri
                        break
                return [
                    None, 0, np.pi/4, np.pi/2, np.pi*3/4,
                    np.pi, -np.pi*3/4, -np.pi/2, -np.pi/4
                ][response_int]
            except:
                time.sleep(1)
                continue
        return None

    def handle_close_range_position(self, action, target, anchor, objects, parents,
                                    action_dict, merged_action_type, last_state_position):
        """Handle position specification for close-range interactions."""
        self.get_remote_file(f"{self.env_id}/{target}_rgb.png")
        image_file = f"planner/capture/{target}_rgb.png"
        image = cv.imread(image_file, cv.IMREAD_UNCHANGED)
        h, w = image.shape[:2]
        image = cv.resize(image, (int(w * 0.25), int(h * 0.25)))

        target_type = re.sub(r'\d', '', target)
        target_string = target_type
        anchor_type = re.sub(r'\d', '', anchor)

        anchor_instance = objects[anchor]
        target_instance = objects[target]

        # Build target description
        target_string = self._build_target_description(
            target, anchor, target_string, anchor_type,
            anchor_instance, target_instance, objects, action_dict
        )

        # Build user prompt
        labels = target_instance["labels"]
        user1, target_directions, anchor_directions = self._build_position_prompt(
            target, anchor, target_type, anchor_type,
            target_instance, anchor_instance, labels,
            merged_action_type, last_state_position, objects
        )

        self.logger.print_role("System")
        self.logger.write(SYSTEM1_SPECIFY_POSITION)
        self.logger.write()
        self.logger.print_role("User")
        self.logger.write(user1)
        self.logger.write()

        # Initialize with default values in case all attempts fail
        anchor_position_tag = list(labels.keys())[0] if labels else 7
        action_position_tag = anchor_position_tag
        # Set default position and tag on action in case parsing fails
        if labels:
            action.position = labels[anchor_position_tag]
            action.position_tag = anchor_position_tag
            action.tag_direction = self._compute_tag_direction(
                target, parents[target], target_instance, anchor_position_tag, anchor_position_tag
            )

        # Get position from LLM
        for _ in range(3):
            response = self.vlm_client.chat([
                {"role": "system", "content": SYSTEM1_SPECIFY_POSITION},
                {"role": "user", "content": user1}
            ])
            self.logger.print_role("Assistant")
            self.logger.write(response)
            self.logger.write()

            try:
                response = response.split(">>>")[1].split("<<<")[0].strip()
                i = None
                for ri in labels:
                    if str(ri) in response:
                        i = ri
                        break
                action.position = labels[i]
                action.position_tag = i
                action_position_tag = i
                anchor_position_tag = i

                # Compute tag direction
                action.tag_direction = self._compute_tag_direction(
                    target, parents[target], target_instance, i, action_position_tag
                )
                break
            except:
                time.sleep(1)
                continue

        return action, target_directions, anchor_directions, anchor_position_tag

    def _build_target_description(self, target, anchor, target_string, anchor_type,
                                  anchor_instance, target_instance, objects, action_dict):
        """Build description string for target object."""
        is_previous_anchor = False
        for current_action in action_dict.values():
            if current_action.place and current_action.at is not None and len(current_action.at) > 0:
                is_previous_anchor = True
            elif current_action.target is not None and not current_action.long_range:
                is_previous_anchor = True

        if target != anchor and not is_previous_anchor:
            target_string += f", which is belong to {anchor_type}, and marked in red box in the image."
        elif target != anchor and is_previous_anchor:
            if objects[anchor]["front"] is not None or objects[anchor]["end"] is not None:
                anchor_position = (
                    np.array(anchor_instance["position"]) +
                    np.array(anchor_instance["bbox"]).mean(0)
                )
                target_position = (
                    np.array(target_instance["position"]) +
                    np.array(target_instance["bbox"]).mean(0)
                )
                anchor_position = anchor_position[:2]
                target_position = target_position[:2]

                target_direction = target_position - anchor_position
                anchor_facing = objects[anchor]["front"] if objects[anchor]["front"] is not None else objects[anchor]["end"]
                anchor_facing = np.deg2rad(anchor_facing)
                anchor_front = np.array([np.cos(anchor_facing), np.sin(anchor_facing)])
                anchor_right = np.array([np.sin(anchor_facing), -np.cos(anchor_facing)])
                target_front = np.dot(target_direction, anchor_front)
                target_right = np.dot(target_direction, anchor_right)

                direction_map = np.array([
                    [1, 0], [1, 1], [0, 1], [-1, 1],
                    [-1, 0], [-1, -1], [0, -1], [1, -1]
                ], dtype=np.float32)
                direction_map = direction_map / np.linalg.norm(direction_map, axis=-1, keepdims=True)
                direction_index = direction_map[:, 0] * target_front + direction_map[:, 1] * target_right
                direction_index = direction_index.argmax()

                if anchor_instance["front"] is not None:
                    direction_names = [
                        "front", "front-right", "right", "back-right",
                        "back", "back-left", "left", "front-left"
                    ]
                else:
                    direction_names = [
                        "end", "corner", "side", "corner",
                        "end", "corner", "side", "corner"
                    ]
                target_string += f", which is on the {direction_names[direction_index]} of {anchor_type}, and marked in red box in the image."
            else:
                target_string += f", which is beside the {anchor_type}, and marked in red box in the image."

        return target_string

    def _build_position_prompt(self, target, anchor, target_type, anchor_type,
                               target_instance, anchor_instance, labels,
                               merged_action_type, last_state_position, objects):
        """Build position specification prompt."""
        user1 = []
        target_string = target_type
        if target != anchor:
            target_string += ", which is marked in red box in the image."
        user1.append(USER1_SPECIFY_POSITION_TARGET.replace("object_placeholder", target_string))

        target_has_orientation = False
        anchor_has_orientation = False
        target_directions = {}
        anchor_directions = {}

        # Check target orientation
        if anchor_instance["front"] is not None:
            front_index = (anchor_instance["front"] // 45) % 8
            target_directions = {
                ((front_index + i) % 8 + 1): direction
                for i, direction in enumerate([
                    "front", "front-left", "left", "back-left",
                    "back", "back-right", "right", "front-right"
                ])
                if ((front_index + i) % 8 + 1) in labels
            }
            target_has_orientation = True
        elif anchor_instance["end"] is not None:
            end_index = (anchor_instance["end"] // 45) % 8
            target_directions = {
                ((end_index + i) % 8 + 1): direction
                for i, direction in enumerate([
                    "end", "corner", "side", "corner",
                    "end", "corner", "side", "corner"
                ])
                if ((end_index + i) % 8 + 1) in labels
            }
            target_has_orientation = True

        # Check anchor orientation
        if anchor_instance["front"] is not None or anchor_instance["end"] is not None:
            anchor_facing = objects[anchor]["front"] if objects[anchor]["front"] is not None else objects[anchor]["end"]
            anchor_facing = np.deg2rad(anchor_facing)
            direction_map = np.array([
                [1, 0], [1, 1], [0, 1], [-1, 1],
                [-1, 0], [-1, -1], [0, -1], [1, -1]
            ], dtype=np.float32)
            anchor_front = np.array([np.cos(anchor_facing), np.sin(anchor_facing)])
            anchor_left = np.array([-np.sin(anchor_facing), np.cos(anchor_facing)])
            direction_map = direction_map[:, 0:1] * anchor_front + direction_map[:, 1:2] * anchor_left

            if anchor_instance["front"] is not None:
                candidate_directions = [
                    "front", "front-left", "left", "back-left",
                    "back", "back-right", "right", "front-right"
                ]
            else:
                candidate_directions = [
                    "end", "corner", "side", "corner",
                    "end", "corner", "side", "corner"
                ]

            anchor_position = (
                np.array(anchor_instance["position"]) +
                np.array(anchor_instance["bbox"]).mean(0)
            )
            anchor_position = anchor_position[:2]
            anchor_directions = {
                i: (np.array(p)[:2] - anchor_position) for i, p in labels.items()
            }
            anchor_directions = {
                i: d / np.linalg.norm(d) for i, d in anchor_directions.items()
            }
            anchor_directions = {
                i: np.argmax((d[None] * direction_map).sum(axis=-1))
                for i, d in anchor_directions.items()
            }
            anchor_directions = {
                i: candidate_directions[d] for i, d in anchor_directions.items()
            }
            anchor_has_orientation = True

        # Build marker descriptions
        pposition = np.array(last_state_position, dtype=np.float32)
        markers = self._build_marker_descriptions(
            target, anchor, target_type, anchor_type,
            labels, target_directions, anchor_directions,
            target_has_orientation, anchor_has_orientation,
            pposition, objects
        )
        user1.append(USER1_SPECIFY_POSITION_MARKER.replace("marker_placeholder", markers))
        user1.append(USER1_SPECIFY_POSITION_ACTION.replace("action_placeholder", merged_action_type))
        user1 = "\n\n".join(user1)

        return user1, target_directions, anchor_directions

    def _build_marker_descriptions(self, target, anchor, target_type, anchor_type,
                                   labels, target_directions, anchor_directions,
                                   target_has_orientation, anchor_has_orientation,
                                   pposition, objects):
        """Build marker description strings."""
        label_indices = sorted(labels.keys())
        markers = []

        if anchor_has_orientation and target_has_orientation and target != anchor:
            for label_index in label_indices:
                target_direction = target_directions.get(label_index, "unknown")
                anchor_direction = anchor_directions.get(label_index, "unknown")
                label_position = np.array(labels[label_index]).astype(float)
                tlabe_position = (
                    np.array(objects[target]["position"], dtype=np.float32) +
                    np.array(objects[target]["bbox"], dtype=np.float32).mean(0)
                )
                tl_distance = np.linalg.norm(tlabe_position[:2] - label_position[:2])
                tp_distance = np.linalg.norm(pposition[:2] - label_position[:2])
                marker = (f"- Label {label_index} is in the {anchor_direction} of {anchor_type}, "
                         f"and in the {target_direction} of {target_type}. "
                         f"Its distance to {target_type} is {tl_distance:.2f}, "
                         f"and your distance to it is {tp_distance:.2f}")
                markers.append(marker)
        elif target_has_orientation:
            for label_index in label_indices:
                target_direction = target_directions.get(label_index, "unknown")
                marker = f"- Label {label_index} is in the {target_direction} of {target_type}"
                label_position = np.array(labels[label_index]).astype(float)
                tp_distance = np.linalg.norm(pposition[:2] - label_position[:2])
                if target != anchor:
                    tlabe_position = (
                        np.array(objects[target]["position"], dtype=np.float32) +
                        np.array(objects[target]["bbox"], dtype=np.float32).mean(0)
                    )
                    tl_distance = np.linalg.norm(tlabe_position[:2] - label_position[:2])
                    marker += f". Its distance to {target_type} is {tl_distance:.2f}, and your distance to it is {tp_distance:.2f}"
                else:
                    marker += f". Your distance to it is {tp_distance:.2f}"
                markers.append(marker)
        elif anchor_has_orientation:
            for label_index in label_indices:
                anchor_direction = anchor_directions.get(label_index, "unknown")
                label_position = np.array(labels[label_index]).astype(float)
                tp_distance = np.linalg.norm(pposition[:2] - label_position[:2])
                marker = f"- Label {label_index} is in the {anchor_direction} of {anchor_type}. Your distance to it is {tp_distance:.2f}"
                markers.append(marker)
        else:
            if target != anchor:
                for label_index in label_indices:
                    label_position = np.array(labels[label_index]).astype(float)
                    tlabe_position = (
                        np.array(objects[target]["position"], dtype=np.float32) +
                        np.array(objects[target]["bbox"], dtype=np.float32).mean(0)
                    )
                    tl_distance = np.linalg.norm(tlabe_position[:2] - label_position[:2])
                    tp_distance = np.linalg.norm(pposition[:2] - label_position[:2])
                    marker = f"- Label {label_index}, its distance to {target_type} is {tl_distance:.2f}, and your distance to it is {tp_distance:.2f}"
                    markers.append(marker)
            else:
                for label_index in label_indices:
                    label_position = np.array(labels[label_index]).astype(float)
                    tp_distance = np.linalg.norm(pposition[:2] - label_position[:2])
                    marker = f"- Label {label_index}. Your distance to it is {tp_distance:.2f}"
                    markers.append(marker)

        return "\n".join(markers)

    def _compute_tag_direction(self, target, parent, target_instance, i, action_position_tag):
        """Compute tag direction based on position tag."""
        if target != parent:
            return None

        if target_instance["front"] is not None:
            front_index = (target_instance["front"] // 45) % 8
            directions = [
                "front", "front-left", "left", "back-left",
                "back", "back-right", "right", "front-right"
            ]
            tag_direction_index = (i - 1 - front_index + 8) % 8

            # Adjust for corner cases
            if tag_direction_index == 1:
                tag_direction_index = 0
                action_position_tag = (action_position_tag + 7) % 8
            if tag_direction_index == 7:
                tag_direction_index = 0
                action_position_tag = (action_position_tag + 1) % 8
            if tag_direction_index == 3:
                tag_direction_index = 4
                action_position_tag = (action_position_tag + 1) % 8
            if tag_direction_index == 5:
                tag_direction_index = 4
                action_position_tag = (action_position_tag + 7) % 8

            return directions[tag_direction_index]

        elif target_instance["end"] is not None:
            end_index = (target_instance["end"] // 45) % 8
            directions = [
                "end", "corner", "side", "corner",
                "end", "corner", "side", "corner"
            ]
            tag_direction_index = (i - 1 - end_index + 8) % 8

            if tag_direction_index == 1:
                tag_direction_index = 0
                action_position_tag = (action_position_tag + 7) % 8
            if tag_direction_index == 7:
                tag_direction_index = 0
                action_position_tag = (action_position_tag + 1) % 8
            if tag_direction_index == 3:
                tag_direction_index = 4
                action_position_tag = (action_position_tag + 1) % 8
            if tag_direction_index == 5:
                tag_direction_index = 4
                action_position_tag = (action_position_tag + 7) % 8

            return directions[tag_direction_index]

        return None

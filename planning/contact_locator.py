"""Contact point localization via iterative VLM queries."""
import os
import re
import time
import numpy as np
import cv2 as cv

from planner.utils.image_utils import (
    get_base64, resize_foreground, resize_foreground_coord_mask,
    get_marker, get_normal_map, get_height_map
)
from planner.utils.text_utils import is_up, is_down, is_left, is_right
from planner.prompts import (
    USER_POSITIONING_INITIAL, USER_POSITIONING_INITIAL_PLACE,
    USER_POSITIONING_LOCATE, USER_POSITIONING_LOCATE_PLACE
)


class ContactPointLocator:
    """Locator for contact points via iterative VLM queries."""

    def __init__(self, env_id, vlm_client, logger, position_handler):
        """Initialize contact point locator."""
        self.env_id = env_id
        self.vlm_client = vlm_client
        self.logger = logger
        self.position_handler = position_handler

    def locate_contact_points(self, action, target, anchor, objects, direction,
                             anchor_type, merged_action_type, position_tag):
        """Locate contact points for an action."""
        tag = position_tag - 1

        self.position_handler.get_remote_file(f"{self.env_id}/{target}/{tag}_rgb.png")
        self.position_handler.get_remote_file(f"{self.env_id}/{target}/{tag}_mask.png")
        self.position_handler.get_remote_file(f"{self.env_id}/{target}/{tag}_rgb_depth.npy")

        image = cv.imread(f"planner/capture/{target}/{tag}_rgb.png")
        mask = cv.imread(f"planner/capture/{target}/{tag}_mask.png")
        point_grid = np.load(f"planner/capture/{target}/{tag}_rgb_depth.npy")

        normal = get_normal_map(point_grid, mask > 0)
        height = get_height_map(point_grid, mask > 0)

        mask = mask[:, :, :1]
        image = np.concatenate([image, mask], axis=-1)
        normal = np.concatenate([normal, mask], axis=-1)
        height = np.concatenate([height, mask], axis=-1)

        image = resize_foreground(image)
        normal = resize_foreground(normal)
        height = resize_foreground(height)
        mask, point_grid = resize_foreground_coord_mask(mask, point_grid)

        image = cv.resize(image, (1024, 1024))
        normal = cv.resize(normal, (1024, 1024))
        height = cv.resize(height, (1024, 1024))

        feed = image.copy()
        feed = cv.resize(feed, (512, 512))

        contact_targets = []
        action_string = merged_action_type
        target_string = re.sub(r'\d', '', target)

        left_direction, right_direction = self._get_adjacent_directions(direction)

        for joint_string in action.contact_points:
            view_string = self._build_view_string(
                direction, left_direction, right_direction,
                target_string, anchor_type
            )

            if action.place:
                place_target_string = re.sub(r'\d', '', action.target)
                prompt = (USER_POSITIONING_INITIAL_PLACE
                         .replace("{view}", view_string)
                         .replace("{target}", target_string)
                         .replace("{action}", action_string)
                         .replace("{place_target}", place_target_string))
            else:
                prompt = (USER_POSITIONING_INITIAL
                         .replace("{view}", view_string)
                         .replace("{target}", target_string)
                         .replace("{joint}", joint_string)
                         .replace("{action}", action_string))

            image_url = {"url": f"data:image/jpeg;base64,{get_base64(feed)}"}

            self.logger.print_role("User")
            self.logger.write(prompt)
            self.logger.write()

            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]

            part_string = self._get_initial_part(content)

            # Iterative refinement
            contact_point = self._iterative_locate(
                image, normal, height, mask, point_grid,
                view_string, target_string, joint_string,
                action_string, part_string, action.place,
                action.target if action.place else None,
                objects.get(action.target, {}) if action.target else {}
            )
            contact_targets.append(contact_point)

        return contact_targets

    def _get_adjacent_directions(self, direction):
        """Get left and right adjacent directions."""
        if direction is None:
            return None, None

        if direction in ["front", "front-left", "left", "back-left",
                        "back", "back-right", "right", "front-right"]:
            directions = ["front", "front-left", "left", "back-left",
                         "back", "back-right", "right", "front-right"]
            direction_index = directions.index(direction)
            left_direction = directions[(direction_index - 2) % 8]
            right_direction = directions[(direction_index + 2) % 8]
        elif direction in ["end", "corner", "side"]:
            directions = ["end", "corner", "side", "corner",
                         "end", "corner", "side", "corner"]
            direction_index = directions.index(direction)
            left_direction = directions[(direction_index - 2) % 8]
            right_direction = directions[(direction_index + 2) % 8]
        else:
            return None, None

        return left_direction, right_direction

    def _build_view_string(self, direction, left_direction, right_direction,
                          target_string, anchor_type):
        """Build view description string."""
        if direction:
            return (f"{direction} view photo of {target_string}, "
                   f"which means the left of the image is the {left_direction} "
                   f"of the {anchor_type}, while the right of the image is "
                   f"the {right_direction} of the {anchor_type}.")
        return "photo"

    def _get_initial_part(self, content):
        """Get initial part string from VLM."""
        for _ in range(3):
            response = self.vlm_client.chat([{"role": "user", "content": content}])
            self.logger.print_role("Assistant")
            self.logger.write(response)
            self.logger.write()
            try:
                return response.split(">>>")[1].split("<<<")[0]
            except:
                time.sleep(1)
                continue
        return ""

    def _iterative_locate(self, image, normal, height, mask, point_grid,
                         view_string, target_string, joint_string,
                         action_string, part_string, is_place,
                         place_target, target_instance):
        """Iteratively locate contact point via VLM queries."""
        last_point = (512, 512)
        delta = 0.125
        last_delta_dir = (0, 0)
        change_dir_time = 0

        alpha = np.where(image[..., 3] > 0)
        w = alpha[1].max() - alpha[1].min()
        h = alpha[0].max() - alpha[0].min()

        def cal_point_position(last_point):
            grid_size = mask.shape[0]
            y, x = last_point
            x = (x * grid_size) // 1024
            y = (y * grid_size) // 1024
            last_point = (x, y)
            mask_coordinates = np.stack(np.where(mask[:, :, 0] > 0)).T
            delta = mask_coordinates - np.array(last_point, dtype=np.float32)
            delta = np.linalg.norm(delta, axis=-1)
            min_distance_index = delta.argmin()
            last_point = mask_coordinates[min_distance_index]
            x, y = last_point
            return point_grid[x, y]

        for ask_time in range(6):
            point_image = get_marker(image, last_point)
            point_height = get_marker(height, last_point)
            point_normal = get_marker(normal, last_point)

            point_feed = point_image.copy()
            point_feed = cv.resize(point_feed, (512, 512))
            cv.imwrite(f"logs/point_feed_{self.env_id}_{joint_string}.png", point_feed)

            if ask_time == 5:
                break

            # Build direction strings for distance hints
            direction_strings = self._build_direction_hints(
                last_point, w, h, mask, point_grid,
                target_instance, target_string, cal_point_position
            )

            if is_place and place_target:
                place_target_string = re.sub(r'\d', '', place_target)
                prompt = (USER_POSITIONING_LOCATE_PLACE
                         .replace("{round}", str(ask_time))
                         .replace("{view}", view_string)
                         .replace("{target}", target_string)
                         .replace("{action}", action_string)
                         .replace("{part}", part_string)
                         .replace("{directions}", direction_strings)
                         .replace("{place_target}", place_target_string))
            else:
                prompt = (USER_POSITIONING_LOCATE
                         .replace("{round}", str(ask_time))
                         .replace("{view}", view_string)
                         .replace("{target}", target_string)
                         .replace("{joint}", joint_string)
                         .replace("{action}", action_string)
                         .replace("{directions}", direction_strings)
                         .replace("{part}", part_string))

            self.logger.print_role("User")
            self.logger.write(prompt)
            self.logger.write()

            image_url = {"url": f"data:image/jpeg;base64,{get_base64(point_feed)}"}
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]

            response = self.vlm_client.chat([{"role": "user", "content": content}])
            self.logger.print_role("Assistant")
            self.logger.write(response)
            self.logger.write()

            try:
                response = response.lower()
                response = response.split(">>>")[1].split("<<<")[0].strip()
                if "yes" in response:
                    break
                else:
                    response = response[response.index(",") + 1:].strip()
                    if " is " in response:
                        response = response.split(" is ")[1].strip()

                    x, y = last_point
                    if "left" in response:
                        dx = -w
                    elif "right" in response:
                        dx = w
                    else:
                        dx = 0

                    if is_up(response):
                        dy = -h
                    elif is_down(response):
                        dy = h
                    else:
                        dy = 0

                    current_delta_dir = (dx, dy)
                    if current_delta_dir != last_delta_dir:
                        change_dir_time += 1
                    else:
                        change_dir_time = max(1, change_dir_time - 1)
                    last_delta_dir = current_delta_dir

                    if is_left(response):
                        x -= int(delta * w / (change_dir_time ** 2))
                    elif is_right(response):
                        x += int(delta * w / (change_dir_time ** 2))

                    if is_up(response):
                        y -= int(delta * h / (change_dir_time ** 2))
                    elif is_down(response):
                        y += int(delta * h / (change_dir_time ** 2))

                    last_point = (x, y)
            except:
                time.sleep(0.1)
                continue

        # Final position calculation
        last_point = cal_point_position(last_point)
        return last_point.tolist()

    def _build_direction_hints(self, last_point, w, h, mask, point_grid,
                               target_instance, target_string, cal_point_position):
        """Build direction hint strings with distance info."""
        if not target_instance or "position" not in target_instance:
            return ""

        target_position = (
            np.array(target_instance["position"]) +
            np.array(target_instance.get("bbox", [[0, 0, 0]] * 8)).mean(0)
        )

        direction_strings = [f"After moveing, the point's distance to {target_string} is:"]
        for response in ["left", "right", "up", "down",
                        "left-up", "left-down", "right-up", "right-down"]:
            x, y = last_point
            if "left" in response:
                dx = -w
            elif "right" in response:
                dx = w
            else:
                dx = 0

            if "up" in response or "above" in response:
                dy = -h
            elif "down" in response or "low" in response:
                dy = h
            else:
                dy = 0

            if "left" in response:
                x -= int(0.125 * w)
            elif "right" in response:
                x += int(0.125 * w)

            if "up" in response or "above" in response:
                y -= int(0.125 * h)
            elif "down" in response or "low" in response:
                y += int(0.125 * h)

            try:
                point_position = cal_point_position((x, y))
                point_distance = np.linalg.norm(np.array(point_position) - target_position)
                direction_strings.append(f"- {response}: {point_distance:.2f}m")
            except:
                pass

        return "\n".join(direction_strings)

    def locate_simple_contact_targets(self, action, target_instance):
        """Locate contact targets using simple position approximation."""
        position = np.array(target_instance["position"])
        position += np.array(target_instance["bbox"]).mean(0)
        return [position.tolist() for _ in action.contact_points]

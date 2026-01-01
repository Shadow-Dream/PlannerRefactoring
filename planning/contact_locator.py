"""Contact point localization via grid-based VLM selection.

This module provides grid-based contact point localization, which is more
efficient than the iterative approach. It overlays an N*N grid of numbered
labels on the object image and asks the VLM to select the appropriate label.
"""

import re
import time
import numpy as np
import cv2 as cv

from planner.utils.image_utils import (
    get_base64, draw_grid_labels_on_object
)
from planner.prompts import (
    USER_GRID_CONTACT_POINT, USER_GRID_CONTACT_POINT_PLACE
)


class ContactPointLocator:
    """Locator for contact points via grid-based VLM selection.

    This class uses a grid-based approach where:
    1. An N*N grid of numbered labels is overlaid on the object image
    2. The VLM is asked to select the appropriate label number
    3. The selected label is mapped to 3D coordinates

    This is more efficient than the iterative marker-moving approach.
    """

    # Default grid size (N*N labels)
    DEFAULT_GRID_SIZE = 8

    def __init__(self, env_id, vlm_client, logger, position_handler):
        """Initialize contact point locator."""
        self.env_id = env_id
        self.vlm_client = vlm_client
        self.logger = logger
        self.position_handler = position_handler

    def locate_contact_points(self, action, target, anchor, objects, direction,
                             anchor_type, merged_action_type, position_tag):
        """Locate contact points for an action using grid-based selection.

        Args:
            action: Action object with contact_points list
            target: Target object name
            anchor: Anchor object name
            objects: Dict of all objects
            direction: View direction string
            anchor_type: Type of anchor object
            merged_action_type: Merged action description
            position_tag: Position tag (1-8)

        Returns:
            List of contact target 3D coordinates
        """
        tag = position_tag - 1

        # Load image, mask, and depth data
        self.position_handler.get_remote_file(f"{self.env_id}/{target}/{tag}_rgb.png")
        self.position_handler.get_remote_file(f"{self.env_id}/{target}/{tag}_mask.png")
        self.position_handler.get_remote_file(f"{self.env_id}/{target}/{tag}_rgb_depth.npy")

        image = cv.imread(f"planner/capture/{target}/{tag}_rgb.png")
        mask = cv.imread(f"planner/capture/{target}/{tag}_mask.png")
        point_grid = np.load(f"planner/capture/{target}/{tag}_rgb_depth.npy")

        # Prepare mask
        mask_for_grid = mask[:, :, 0] if len(mask.shape) == 3 else mask

        # Prepare image with alpha channel
        if image.shape[2] == 3:
            alpha = (mask_for_grid > 0).astype(np.uint8) * 255
            image_rgba = np.concatenate([image, alpha[:, :, np.newaxis]], axis=-1)
        else:
            image_rgba = image

        # Build simple view string (e.g., "front view")
        target_string = re.sub(r'\d', '', target)
        view_string = f"{direction} view" if direction else "view"

        action_string = merged_action_type
        contact_targets = []

        for joint_string in action.contact_points:
            # Generate grid-labeled image and coordinate mapping
            n_grid = self.DEFAULT_GRID_SIZE
            max_label = n_grid * n_grid

            debug_path = f"logs/grid_contact_{self.env_id}_{target}_{joint_string}.png"
            labeled_image, label_to_coord = draw_grid_labels_on_object(
                image_rgba, mask_for_grid, point_grid,
                n_grid=n_grid, output_path=debug_path
            )

            # Resize for VLM input
            feed_image = cv.resize(labeled_image, (512, 512))

            # Build prompt
            if action.place:
                place_target_string = re.sub(r'\d', '', action.target)
                prompt = (USER_GRID_CONTACT_POINT_PLACE
                         .replace("{view}", view_string)
                         .replace("{target}", target_string)
                         .replace("{action}", action_string)
                         .replace("{place_target}", place_target_string)
                         .replace("{max_label}", str(max_label)))
            else:
                prompt = (USER_GRID_CONTACT_POINT
                         .replace("{view}", view_string)
                         .replace("{target}", target_string)
                         .replace("{joint}", joint_string)
                         .replace("{action}", action_string)
                         .replace("{max_label}", str(max_label)))

            # Log the prompt
            self.logger.print_role("User")
            self.logger.write(prompt)
            self.logger.write()

            # Prepare image for VLM
            image_url = {"url": f"data:image/png;base64,{get_base64(feed_image)}"}
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url}
            ]

            # Get VLM response with retry
            selected_label = self._get_grid_selection(content, max_label)

            # Get 3D coordinate for selected label
            if selected_label in label_to_coord:
                contact_point = label_to_coord[selected_label]
            else:
                # Fallback: use center label
                center_label = (n_grid // 2) * n_grid + (n_grid // 2) + 1
                contact_point = label_to_coord.get(center_label, [0, 0, 0])
                self.logger.write(f"Warning: Invalid label {selected_label}, using center {center_label}")

            contact_targets.append(contact_point)

        return contact_targets

    def _get_grid_selection(self, content, max_label, max_retries=3):
        """Get grid label selection from VLM with retry.

        Args:
            content: VLM message content
            max_label: Maximum valid label number
            max_retries: Number of retry attempts

        Returns:
            Selected label number (1 to max_label)
        """
        for attempt in range(max_retries):
            response = self.vlm_client.chat([{"role": "user", "content": content}])
            self.logger.print_role("Assistant")
            self.logger.write(response)
            self.logger.write()

            try:
                # Extract number from >>> ... <<<
                selected = response.split(">>>")[1].split("<<<")[0].strip()

                # Parse the number
                # Handle cases like "28" or "Label 28" or ">>>28<<<"
                numbers = re.findall(r'\d+', selected)
                if numbers:
                    label_num = int(numbers[0])
                    if 1 <= label_num <= max_label:
                        return label_num

                # If no valid number found, try to find any number in range
                all_numbers = re.findall(r'\d+', response)
                for num_str in all_numbers:
                    num = int(num_str)
                    if 1 <= num <= max_label:
                        return num

            except (IndexError, ValueError) as e:
                self.logger.write(f"Parse error (attempt {attempt + 1}): {e}")
                time.sleep(0.5)
                continue

        # Fallback: return center label
        n_grid = int(np.sqrt(max_label))
        center = (n_grid // 2) * n_grid + (n_grid // 2) + 1
        self.logger.write(f"Using fallback center label: {center}")
        return center

    def locate_simple_contact_targets(self, action, target_instance):
        """Locate contact targets using simple position approximation."""
        position = np.array(target_instance["position"])
        position += np.array(target_instance["bbox"]).mean(0)
        return [position.tolist() for _ in action.contact_points]

"""
LLM-based Planner for Humanoid Agent Decision Making.

This module provides the main planner classes for controlling humanoid agents
using LLM-based decision making and behavior generation.

Classes:
    LLMPlanner: Main planner with LLM-based behavior generation
"""

import torch
import os
import numpy as np
import copy
import re
import multiprocessing
import shapely as sl
from shapely import ops

from planner.core import quaternion
from planner.prompts import SURFACE_TO_JOINT
from planner.utils.action_logger import ActionLogger

# Import utility functions for backward compatibility
from planner.utils.text_utils import is_up, is_down, is_left, is_right
from planner.utils.geometry_utils import get_convex_hull, polygon_to_list, linestring_to_list

# Import action classes
from planner.core.action import HumanAction, merge_action, action_local_to_world

# Import backends
from planner.navigation import NavigationBackend
from planner.planning import PlanningBackend


__all__ = [
    'LLMPlanner', 'HumanAction',
    'merge_action', 'action_local_to_world',
    'is_up', 'is_down', 'is_left', 'is_right',
    'get_convex_hull', 'polygon_to_list', 'linestring_to_list'
]


class LLMPlanner:
    """
    LLM-based planner for humanoid agent behavior generation.

    This planner uses a vision-language model to generate behaviors based on
    natural language instructions and visual observations of the environment.
    """

    def __init__(self, env, env_id):
        """
        Initialize the LLM planner.

        Args:
            env: Environment instance
            env_id: Environment identifier
        """
        self.env_id = env_id
        self.planner_request_queue = multiprocessing.Queue()
        self.planner_result_queue = multiprocessing.Queue()
        self.navigator_request_queue = multiprocessing.Queue()
        self.navigator_result_queue = multiprocessing.Queue()

        self.planning_backend = multiprocessing.Process(target=self.planning)
        self.planning_backend.daemon = True
        self.navigation_backend = multiprocessing.Process(target=self.navigation)
        self.navigation_backend.daemon = True

        self.planning_backend.start()
        self.navigation_backend.start()

        self.state = {
            "left_slot": None,
            "right_slot": None,
            "action": [],
            "position": None,
            "position_name": None
        }

        self.stage = "init"
        self.capture_counter = 0

        self.path = None
        self.dones = []
        self.todos = []
        self.navigating = False

        self.target_state_dict = {
            "position": {"traj": torch.tensor([0, 0, 0], dtype=torch.float32, device=env.device)},
            "velocity": None,
            "heading": None
        }
        self.flush_action = False
        self.prompt = "A person is standing still."
        self.failed = False
        self.completed = False  # Task completed successfully
        self.enable_lefthand_ik = False
        self.enable_righthand_ik = False
        self.smooth_reference = False
        self.grab_position_left = None
        self.grab_position_right = None
        self.left_take_target = None
        self.right_take_target = None
        self.left_drop_target = None
        self.right_drop_target = None

        self.change_state_next = False
        self.has_action_done_next = False

        self.two_hand_grab_target = None
        self.two_hand_grab_position = None

        self.pending_stop = []
        self.need_pending = False

        self.last_action = ""
        self.last_position = None

        # Action logger for detailed state tracking
        self.action_logger = ActionLogger(env_id)

    def exit(self):
        """Clean up and exit the planner."""
        # Close action logger
        self.action_logger.close()

        self.planner_request_queue.close()
        self.planner_result_queue.close()
        self.navigator_request_queue.close()
        self.navigator_result_queue.close()

        self.planning_backend.terminate()
        self.planning_backend.join()

        self.navigation_backend.terminate()
        self.navigation_backend.join()

    def update(self, env):
        """
        Update the planner state.

        Args:
            env: Environment instance
        """
        self._update(env)

        # Handle two-hand grabbing IK control
        if self.two_hand_grab_target is not None:
            self._handle_two_hand_grab(env)

        # Ensure position exists
        if len(self.target_state_dict["position"]) == 0:
            self._set_default_position(env)

    def _handle_two_hand_grab(self, env):
        """Handle two-hand grabbing IK control."""
        agent_position = env.get_joint_info(self.env_id, "pelvis")[0]
        agent_position = torch.tensor(agent_position).cuda().float()
        agent_rotation = env.current_facing_angle[self.env_id]
        agent_forward = torch.tensor([np.cos(agent_rotation), np.sin(agent_rotation)]).cuda().float()
        agent_right = torch.tensor([np.sin(agent_rotation), -np.cos(agent_rotation)]).cuda().float()
        pos = torch.tensor(self.two_hand_grab_position).cuda().float()
        pos[:2] = pos[0] * agent_forward
        pos += agent_position

        self.enable_lefthand_ik = True
        self.enable_righthand_ik = True
        self.grab_position_left = pos
        self.grab_position_right = pos

    def _set_default_position(self, env):
        """Set default target position."""
        position = env.get_agent_position(self.env_id)
        target_position = position.tolist() + [0.9]
        target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
        self.target_state_dict["position"] = {"pelvis": target_position}

    def _update(self, env):
        """Internal update implementation."""
        # Acquire observation from environment
        if env.is_ready[self.env_id]:
            env.is_ready[self.env_id] = False
            self.planner_request_queue.put({
                "type": "capture",
                "content": env.capture_results[self.env_id]
            })
            env.capture_results[self.env_id] = []

        objects = env.objects[self.env_id]
        parents = env.parents[self.env_id]
        position = env.get_agent_position(self.env_id)

        # Receive behavior from planner
        if not self.planner_result_queue.empty():
            response = self.planner_result_queue.get()
            if response["type"] == "action":
                self._handle_action_response(response, parents)
            elif response["type"] == "update":
                self._handle_update_response(response, parents)
            elif response["type"] == "capture":
                env.is_capturing[self.env_id] = True
                env.capture_targets[self.env_id] += response["target"]
            elif response["type"] == "stop":
                self._handle_stop_response(response)
            elif response["type"] == "anchor":
                self.todos[0]["contact_targets"] = response["contact_targets"]
            elif response["type"] == "complete":
                self._handle_complete_response()
                return
            elif response["type"] == "fail":
                self._handle_fail_response()
                return

        # Reset frame state
        self.left_take_target = None
        self.right_take_target = None
        self.enable_lefthand_ik = False
        self.enable_righthand_ik = False
        env.debug_lines[self.env_id] = None
        change_state = self.change_state_next
        self.change_state_next = False
        has_action_done = self.has_action_done_next
        self.has_action_done_next = False

        def clear_pending():
            self.need_pending = False
            for stop_action in self.pending_stop:
                for index, action in enumerate(self.todos):
                    if action["action"] == stop_action:
                        self.todos.pop(index)
                        break
            self.pending_stop = []

        if self.todos:
            self._process_todos(env, objects, parents, position, change_state,
                               has_action_done, clear_pending)
        else:
            self._handle_idle(env, position)

        self.flush_action = change_state or has_action_done
        if self.flush_action:
            print("Action Flushed")

        # Log current frame state to CSV
        self.action_logger.log_frame(env, self)

    def _handle_action_response(self, response, parents):
        """Handle action response from planner."""
        if len(self.todos) > 0:
            last_action = self.todos[-1]
            last_target = last_action["at"][0] if (last_action["at"] is not None and len(last_action["at"]) > 0) else last_action["target"]
            inco_target = response["at"][0] if (response["at"] is not None and len(response["at"]) > 0) else response["target"]
            # If same parent and last action has no position, replace it
            if last_action["position"] is None and last_target is not None and inco_target is not None and parents[last_target] == parents[inco_target]:
                self.todos.pop(-1)
        self.todos.append(response)

    def _handle_update_response(self, response, parents):
        """Handle update response from planner - updates an existing action with full data."""
        action_name = response["action"]
        for action in self.todos:
            if action["action"] == action_name:
                # Update the existing action with new data
                for key in response:
                    if key != "type":
                        action[key] = response[key]
                return
        # If action not found in todos, treat as new action
        self._handle_action_response(response, parents)

    def _handle_stop_response(self, response):
        """Handle stop response from planner."""
        if self.need_pending:
            self.pending_stop.append(response["action"])
        else:
            found_index = None
            for index, action in enumerate(self.todos):
                if action["action"] == response["action"]:
                    found_index = index
                    break
            if found_index is not None:
                self.todos.pop(found_index)

    def _handle_fail_response(self):
        """Handle fail response from planner."""
        self.failed = True

    def _handle_complete_response(self):
        """Handle complete response from planner - task finished successfully."""
        self.completed = True
        print("==================== Task Completed ====================")

    def _handle_idle(self, env, position):
        """Handle idle state when no todos."""
        target_position = position.tolist() + [0.9]
        target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
        self.target_state_dict["position"] = {"pelvis": target_position}
        self.target_state_dict["velocity"] = None
        self.prompt = "A person is standing still."

    def _process_todos(self, env, objects, parents, position, change_state,
                      has_action_done, clear_pending):
        """Process todo list actions."""
        from planner.update_handler import process_todos
        process_todos(self, env, objects, parents, position, change_state,
                     has_action_done, clear_pending)

    def navigation(self):
        """Navigation backend process entry point."""
        backend = NavigationBackend(
            self.env_id,
            self.navigator_request_queue,
            self.navigator_result_queue
        )
        backend.run()

    def planning(self):
        """Planning backend process entry point."""
        backend = PlanningBackend(
            self.env_id,
            self.planner_request_queue,
            self.planner_result_queue
        )
        backend.run()
"""Action logger for saving detailed planner state to CSV."""
import os
import csv
import numpy as np
from typing import Dict, Any, Optional


class ActionLogger:
    """Logger for recording planner state to CSV file."""

    # Joint names that may appear in target_state_dict
    JOINT_NAMES = [
        "traj", "pelvis", "left_wrist", "right_wrist",
        "left_hand", "right_hand", "left_foot", "right_foot", "head"
    ]

    def __init__(self, env_id: int, log_dir: str = "logs"):
        """Initialize action logger.

        Args:
            env_id: Environment identifier
            log_dir: Directory to save log files
        """
        self.env_id = env_id
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f"action_log_{env_id}.csv")
        self.frame_count = 0
        self._file = None
        self._writer = None
        self._initialized = False

    def _init_csv(self):
        """Initialize CSV file with headers."""
        os.makedirs(self.log_dir, exist_ok=True)

        # Build column headers
        headers = [
            "frame",
            "agent_pos_x",
            "agent_pos_y",
            "agent_facing",
            "prompt",
            "target_facing",
            "num_todos",
            "todo_target",
            "todo_position",
            "todo_stage",
            "path_len",
            "navigating"
        ]
        # Add joint position columns
        for joint in self.JOINT_NAMES:
            headers.append(f"target_{joint}")

        self._file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(headers)
        self._initialized = True

    def log_frame(self, env, planner):
        """Log current frame state to CSV.

        Args:
            env: Environment instance
            planner: LLMPlanner instance
        """
        if not self._initialized:
            self._init_csv()

        self.frame_count += 1

        # Get current agent state
        agent_position = env.get_agent_position(planner.env_id)
        agent_facing = env.current_facing_angle[planner.env_id]

        # Get target state
        target_state = planner.target_state_dict
        prompt = planner.prompt

        # Extract target facing
        target_facing = target_state.get("heading")
        if target_facing is not None:
            target_facing = f"{float(target_facing):.2f}"
        else:
            target_facing = ""

        # Extract todo info
        num_todos = len(planner.todos)
        todo_target = ""
        todo_position = ""
        todo_stage = ""
        if num_todos > 0:
            first_todo = planner.todos[0]
            todo_target = first_todo.get("target", "")
            pos = first_todo.get("position")
            if pos is not None:
                todo_position = f"({pos[0]:.2f}, {pos[1]:.2f})"
            else:
                todo_position = "None"
            todo_stage = str(first_todo.get("stage", ""))

        # Path info
        path_len = len(planner.path) if planner.path else 0
        navigating = "Y" if planner.navigating else "N"

        # Extract joint positions
        position_dict = target_state.get("position", {})
        joint_positions = {}

        for joint in self.JOINT_NAMES:
            if joint in position_dict:
                pos = position_dict[joint]
                # Handle tensor
                if hasattr(pos, "cpu"):
                    pos = pos.cpu().numpy()
                elif hasattr(pos, "tolist"):
                    pos = np.array(pos)
                # Format as string
                if len(pos) >= 3:
                    joint_positions[joint] = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                elif len(pos) == 2:
                    joint_positions[joint] = f"({pos[0]:.2f}, {pos[1]:.2f})"
                else:
                    joint_positions[joint] = str(pos)
            else:
                joint_positions[joint] = ""

        # Build row
        row = [
            self.frame_count,
            f"{agent_position[0]:.2f}",
            f"{agent_position[1]:.2f}",
            f"{agent_facing:.2f}",
            prompt[:100] if prompt else "",  # Truncate long prompts
            target_facing,
            num_todos,
            todo_target,
            todo_position,
            todo_stage,
            path_len,
            navigating
        ]

        # Add joint positions
        for joint in self.JOINT_NAMES:
            row.append(joint_positions.get(joint, ""))

        self._writer.writerow(row)

    def close(self):
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
            self._initialized = False

    def __del__(self):
        """Cleanup on destruction."""
        self.close()

"""Mock environment for planner testing.

This module provides a mock IsaacGym environment that replicates the exact
behavior of planner_old/env.py for testing purposes without requiring
the actual IsaacGym simulator.

Key implementation details matching env.py:
1. Parents calculation: Uses intersection_threshold (0.75) + NetworkX connected components
2. Shape calculation: Parses URDF files for collision boxes using parse_urdf()
3. Object naming: Uses type counter like "{type}{index}" (e.g., "single cabinet1")
4. Position: Converts to relative coordinates using start_point
"""
import os
import json
import copy
import numpy as np
import torch
import networkx as nx
import xml.etree.ElementTree as ET


def is_type(name, type_set):
    """Check if name contains any type from type_set.

    Matches env.py:105-106
    """
    return any([t in name for t in type_set])


def compute_bbox_vertices(center, size):
    """Compute 8 vertices of a bounding box from center and size.

    Matches env.py:69-83
    """
    x, y, z = center
    dx, dy, dz = [s / 2 for s in size]

    vertices = np.array([
        [x - dx, y - dy, z - dz],
        [x + dx, y - dy, z - dz],
        [x - dx, y + dy, z - dz],
        [x + dx, y + dy, z - dz],
        [x - dx, y - dy, z + dz],
        [x + dx, y - dy, z + dz],
        [x - dx, y + dy, z + dz],
        [x + dx, y + dy, z + dz]
    ])
    return vertices


def parse_urdf(file_path):
    """Parse URDF file to extract collision box vertices.

    Matches env.py:85-103

    Args:
        file_path: Path to URDF file

    Returns:
        numpy array of shape (N, 8, 3) where N is number of collision boxes
    """
    if not os.path.exists(file_path):
        return np.zeros((1, 8, 3))

    tree = ET.parse(file_path)
    root = tree.getroot()

    bbox_list = []

    for collision in root.findall(".//collision"):
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        box = geometry.find("box") if geometry is not None else None

        if origin is not None and box is not None:
            xyz = list(map(float, origin.get("xyz").split()))
            size = list(map(float, box.get("size").split()))

            bbox = compute_bbox_vertices(xyz, size)
            bbox_list.append(bbox)

    if not bbox_list:
        return np.zeros((1, 8, 3))

    return np.array(bbox_list)


class MockEnv:
    """Mock IsaacGym environment for testing planner without simulation.

    This class replicates the scene loading, object building, and parent
    calculation logic from planner_old/env.py (CLoSDPlanner class).
    """

    # Type sets matching env.py
    DECORATION_TYPES = {
        "rug",
        "window",
        "floor",
        "pillar",
        "ceiling",
        "exterior",
        "wall",
        "support",
        "mirror",
        "door",
    }

    IGNORE_TYPES = {
        "ceiling",
        "exterior"
    }

    TAKEABLE_TYPES = {
        "large plant container",
        "plant container",
        "nature shelf trinkets"
    }

    def __init__(self, env_id, scene_dir, capture_dir, task_dir, image_dir):
        """Initialize mock environment.

        Args:
            env_id: Environment identifier
            scene_dir: Path to scene configuration directory (contains config.json and URDF files)
            capture_dir: Path to capture data directory
            task_dir: Path to task configuration directory
            image_dir: Path to scene images directory
        """
        self.env_id = env_id
        self.scene_dir = scene_dir
        self.capture_dir = capture_dir
        self.task_dir = task_dir
        self.image_dir = image_dir

        # Device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Intersection threshold matching env.py:363
        self.intersection_threshold = 0.75

        # Load start point first (needed for relative positions)
        self._load_start_point()

        # Load scene config
        self._load_scene_config()

        # Build parents first (before filtering by floor polygon)
        # Matches env.py order: _build_parents -> _build_start_point -> _build_objects
        self._build_parents()

        # Build objects (filter and create final object dict)
        self._build_objects()

        # Load task
        self._load_task()

        # Initialize state dictionaries (indexed by env_id)
        self.objects = {env_id: self._objects}
        self.parents = {env_id: self._parents}

        # State variables matching env.py
        self.is_ready = {env_id: False}
        self.capture_results = {env_id: []}
        self.is_capturing = {env_id: False}
        self.capture_targets = {env_id: []}
        self.debug_lines = {env_id: None}
        self.has_map_change = {env_id: False}

        # Agent state (starts at origin in relative coordinates)
        self._agent_position = np.array([0, 0, 0], dtype=np.float32)
        self.current_facing_angle = torch.tensor(
            [np.pi / 2], dtype=torch.float32, device=self.device
        )

        # Capture cache
        self._capture_cache = {}
        self._preload_captures()

        # Frame counter for simulation
        self.frame_count = 0

        # Tracking for action state (used by update_handler)
        self.target_state_dict = {env_id: {"position": {}}}
        self.actual_used_prompts = {env_id: "A person is standing still."}
        self.left_slot = {env_id: None}
        self.right_slot = {env_id: None}

    def _load_start_point(self):
        """Load starting point from capture data."""
        start_path = os.path.join(self.capture_dir, "start_point.pt")
        self._start_point = torch.load(start_path, weights_only=False)
        if isinstance(self._start_point, torch.Tensor):
            self._start_point = self._start_point.numpy()
        # Match env.py:584 - set z to 0.13
        self._start_point[2] = 0.13

    def _load_scene_config(self):
        """Load scene configuration from config.json."""
        config_path = os.path.join(self.scene_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        self._raw_objects = config.get("objects", {})
        self._room_type = config.get("room", "bedroom")

    def _build_parents(self):
        """Build parent-child relationships using intersection threshold.

        This exactly matches env.py:689-748 (_build_parents method).
        Uses bounding box intersection and NetworkX connected components.
        """
        objects = self._raw_objects

        # Filter out decoration and ignore types (matching env.py:696-716)
        filtered_objects = {
            name: setting for name, setting in objects.items()
            if not is_type(setting["type"], self.IGNORE_TYPES)
            and not is_type(setting["type"], self.DECORATION_TYPES)
        }

        object_list = list(filtered_objects.keys())
        if not object_list:
            self._parents_raw = {}
            return

        # Compute bounding boxes (matching env.py:719-723)
        bboxes = [np.array(filtered_objects[name]["bbox"])[None, :] for name in object_list]
        bboxes = np.concatenate(bboxes, 0)
        positions = [np.array(filtered_objects[name]["position"])[None, None, :] for name in object_list]
        positions = np.concatenate(positions, 0)
        bboxes = bboxes + positions

        # Get height (min Z) for each object
        height = bboxes[:, :, 2].min(axis=-1)

        # Get 2D bboxes for intersection calculation
        bboxes_2d = bboxes[:, :, :2]
        min_axis = bboxes_2d.min(1)
        max_axis = bboxes_2d.max(1)

        # Compute pairwise intersections (matching env.py:727-736)
        n = len(object_list)
        pairs = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1)
        pair_min_axis = min_axis[pairs]
        pair_max_axis = max_axis[pairs]
        pair_min_axis = pair_min_axis.max(-2)
        pair_max_axis = pair_max_axis.min(-2)
        pair_length = np.maximum(pair_max_axis - pair_min_axis, 0)
        pair_intersection = pair_length.prod(-1)
        area = (max_axis - min_axis).prod(-1)
        pair_min_area = area[pairs].min(-1)

        # Find connected pairs using intersection threshold
        connected = (pair_intersection / (pair_min_area + 1e-8)) > self.intersection_threshold
        start, end = np.where(connected)

        # Build graph and find connected components
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(list(zip(start, end)))
        object_sets = list(nx.connected_components(graph))

        # Assign parent as the lowest object in each connected component
        parents = {}
        for object_set in object_sets:
            object_set = list(object_set)
            parent_idx = object_set[height[object_set].argmin()]
            for obj_idx in object_set:
                parents[object_list[obj_idx]] = object_list[parent_idx]

        self._parents_raw = parents

    def _build_objects(self):
        """Build objects dictionary with proper naming and attributes.

        This matches env.py:547-687 (_build_objects method).
        """
        objects = self._raw_objects

        # Name counters for generating simple names like "bed1", "bed2"
        name_dict = {}
        index_dict = {}  # Maps factory_name -> simple_name
        object_dict = {}

        for factory_name, setting in objects.items():
            type_name = setting["type"]

            # Skip wall (special handling in env.py:590-604)
            if f"{self._room_type}_0_0_wall" in factory_name:
                continue

            # Skip ignore types
            if is_type(type_name, self.IGNORE_TYPES):
                continue

            # Skip decoration types
            if is_type(type_name, self.DECORATION_TYPES):
                continue

            # Build instance (matching env.py:623-679)
            instance = {}
            instance["name"] = type_name
            instance["bbox"] = setting["bbox"]
            instance["bbox_oriented"] = setting.get("bbox_oriented", setting["bbox"])

            # Convert position to relative coordinates
            position = np.array(setting["position"]) - self._start_point
            instance["position"] = position.tolist()
            instance["rotation"] = [1, 0, 0, 0]  # Identity quaternion

            # Parse URDF for shape (matching env.py:634-635)
            urdf_path = os.path.join(self.scene_dir, f"{factory_name}.urdf")
            instance["shape"] = parse_urdf(urdf_path)

            # Handle assignment
            instance["handle"] = len(object_dict) + 1

            # Initialize orientation fields
            instance["front"] = None
            instance["end"] = None
            instance["labels"] = {}

            # Generate simple name (matching env.py:676-679)
            name_dict[type_name] = name_dict.get(type_name, 0) + 1
            name_index = name_dict[type_name]
            simple_name = f"{type_name}{name_index}"

            index_dict[factory_name] = simple_name
            object_dict[simple_name] = instance

        self._objects = object_dict
        self._index_dict = index_dict

        # Convert parents from factory names to simple names (matching env.py:682-684)
        self._parents = {}
        for factory_k, factory_v in self._parents_raw.items():
            if factory_k in index_dict and factory_v in index_dict:
                self._parents[index_dict[factory_k]] = index_dict[factory_v]

        # Objects without parent mapping should be their own parent
        for simple_name in object_dict:
            if simple_name not in self._parents:
                self._parents[simple_name] = simple_name

    def _load_task(self):
        """Load task configuration."""
        task_path = os.path.join(self.task_dir, "task.json")
        with open(task_path, "r") as f:
            task_data = json.load(f)

        self.prompt = task_data.get("prompt", "")
        self.mission = task_data.get("mission", [])

    def _preload_captures(self):
        """Preload all capture data into memory.

        This loads orientation and position labels from capture directory,
        matching env.py's capture result format:
        - front: int or None (direction index)
        - end: int or None (direction index)
        - labels: dict mapping position index to numpy array [x, y, z]
        """
        # Load object-specific captures
        for item in os.listdir(self.capture_dir):
            item_path = os.path.join(self.capture_dir, item)
            if os.path.isdir(item_path):
                self._load_object_capture(item, item_path)

    def _load_object_capture(self, obj_name, obj_path):
        """Load capture data for a specific object.

        Reads orientation.npy and *_tpl_position.npy files to build
        capture results matching env.py format.

        Args:
            obj_name: Object name (e.g., "bed1", "monitor1")
            obj_path: Path to object's capture directory
        """
        capture_data = {"name": obj_name, "path": obj_path}

        # Load orientation if exists (contains front, end, rotation)
        ori_path = os.path.join(obj_path, "orientation.npy")
        if os.path.exists(ori_path):
            ori_data = np.load(ori_path, allow_pickle=True)
            # Handle numpy array containing dict
            if ori_data.shape == ():
                ori_data = ori_data.item()
            capture_data["front"] = ori_data.get("front")
            capture_data["end"] = ori_data.get("end")
            capture_data["rotation"] = ori_data.get("rotation")
        else:
            capture_data["front"] = None
            capture_data["end"] = None
            capture_data["rotation"] = None

        # Load position labels (0-7 index -> 1-8 label)
        # Labels are already in world coordinates, don't transform
        labels = {}
        for i in range(8):
            pos_path = os.path.join(obj_path, f"{i}_tpl_position.npy")
            if os.path.exists(pos_path):
                pos = np.load(pos_path)
                # Labels are in world coordinates - use as-is
                labels[i + 1] = pos.astype(np.float32)
        capture_data["labels"] = labels

        self._capture_cache[obj_name] = capture_data

    def get_agent_position(self, env_id):
        """Get agent's 2D position.

        Args:
            env_id: Environment identifier

        Returns:
            numpy array of shape (2,) with x, y coordinates
        """
        return self._agent_position[:2].copy()

    def get_joint_info(self, env_id, joint):
        """Get joint position and force.

        Args:
            env_id: Environment identifier
            joint: Joint name (e.g., 'pelvis', 'left_wrist')

        Returns:
            Tuple of (position, force) numpy arrays
        """
        position = self._agent_position.copy()

        # Adjust height based on joint
        if joint == "pelvis":
            position[2] = 0.9
        elif joint == "head":
            position[2] = 1.7
        elif joint in ["left_wrist", "right_wrist"]:
            position[2] = 1.0
        elif joint in ["left_foot", "right_foot"]:
            position[2] = 0.0

        force = np.zeros(3, dtype=np.float32)
        return position, force

    def get_heading(self, angle, env_id):
        """Calculate heading difference.

        Matches env.py:893-908

        Args:
            angle: Target angle in radians
            env_id: Environment identifier

        Returns:
            Heading difference in radians
        """
        if angle is None:
            return 0

        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        current_angle = float(self.current_facing_angle[0])

        if angle > current_angle:
            if angle > current_angle + np.pi:
                return angle - 2 * np.pi - current_angle
            else:
                return angle - current_angle
        else:
            if current_angle > angle + np.pi:
                return angle + 2 * np.pi - current_angle
            else:
                return angle - current_angle

    def set_agent_position(self, position):
        """Set agent position (for simulation).

        Args:
            position: New position [x, y] or [x, y, z]
        """
        if len(position) == 2:
            self._agent_position[:2] = position
        else:
            self._agent_position = np.array(position, dtype=np.float32)

    def set_facing_angle(self, angle):
        """Set agent facing angle (for simulation).

        Args:
            angle: Facing angle in radians
        """
        self.current_facing_angle[0] = angle

    def prepare_capture_result(self, targets):
        """Prepare capture results for the specified targets.

        Matches env.py capture result format:
        - front: int or None
        - end: int or None
        - labels: dict mapping position index to numpy array [x, y, z]

        Args:
            targets: List of target object names to capture
        """
        results = []

        for target in targets:
            result = {}

            # Get cached capture data
            if target in self._capture_cache:
                cache = self._capture_cache[target]

                # Load orientation (front, end) - matching env.py:1737-1741
                result["front"] = cache.get("front")
                result["end"] = cache.get("end")

                # Load labels - matching env.py:1740
                result["labels"] = cache.get("labels", {})

            else:
                # No capture data available - return empty result
                result["front"] = None
                result["end"] = None
                result["labels"] = {}

            results.append(result)

        self.capture_results[self.env_id] = results
        self.is_ready[self.env_id] = True
        self.is_capturing[self.env_id] = False

    def reset(self):
        """Reset environment state."""
        self._agent_position = np.array([0, 0, 0], dtype=np.float32)
        self.current_facing_angle[0] = np.pi / 2
        self.is_ready[self.env_id] = False
        self.capture_results[self.env_id] = []
        self.is_capturing[self.env_id] = False
        self.capture_targets[self.env_id] = []
        self.frame_count = 0
        # Reset action tracking
        self.target_state_dict[self.env_id] = {"position": {}}
        self.actual_used_prompts[self.env_id] = "A person is standing still."
        self.left_slot[self.env_id] = None
        self.right_slot[self.env_id] = None

    def step(self):
        """Advance simulation by one frame."""
        self.frame_count += 1

        # Handle capture requests
        if self.is_capturing[self.env_id] and self.capture_targets[self.env_id]:
            targets = self.capture_targets[self.env_id]
            self.prepare_capture_result(targets)
            self.capture_targets[self.env_id] = []

    def get_init_data(self):
        """Get initialization data for planner.

        Returns:
            Tuple of (prompt, objects, parents)
        """
        return (
            self.prompt,
            copy.deepcopy(self._objects),
            copy.deepcopy(self._parents),
        )

    def print_objects_summary(self):
        """Print a summary of objects and their parent relationships."""
        print("\n=== Objects Summary ===")
        print(f"Total objects: {len(self._objects)}")
        print(f"Start point: {self._start_point.tolist()}")

        # Group objects by parent
        parent_children = {}
        for name, parent in self._parents.items():
            if parent not in parent_children:
                parent_children[parent] = []
            if name != parent:
                parent_children[parent].append(name)

        # Print each parent and its children
        for parent_name in sorted(parent_children.keys()):
            children = parent_children[parent_name]
            obj = self._objects.get(parent_name, {})
            pos = obj.get("position", [0, 0, 0])
            shape = obj.get("shape", np.zeros((1, 8, 3)))
            n_collision_boxes = shape.shape[0] if len(shape.shape) == 3 else 1

            if children:
                children_str = ", ".join(sorted(children))
                print(f"- {parent_name} at ({pos[0]:.1f}, {pos[1]:.1f}), "
                      f"shape: {n_collision_boxes} collision boxes, "
                      f"children: [{children_str}]")
            else:
                print(f"- {parent_name} at ({pos[0]:.1f}, {pos[1]:.1f}), "
                      f"shape: {n_collision_boxes} collision boxes")

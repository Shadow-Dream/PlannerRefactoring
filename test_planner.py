#!/usr/bin/env python3
"""
Planner Mock Test Script

This script tests the refactored planner by:
1. Loading scene, task, and capture data from planner/scene, planner/capture, etc.
2. Creating a mock environment that simulates IsaacGym
3. Using pre-recorded VLM responses from capture/response.json
4. Handling capture requests by reading from capture directory

Usage:
    python planner/test_planner.py

Requirements:
    - mgpt conda environment
    - planner/scene/, planner/capture/, planner/task/, planner/image/ directories
    - planner/capture/response.json with pre-recorded VLM responses

NOTE: This script does NOT write anything to the capture directory.
VLM responses are read from capture/response.json.
Capture data (orientation, labels) is read from capture/<object_name>/ directories.
"""

import os
import sys
import copy
import time
import numpy as np
import cv2 as cv
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.mock_env import MockEnv
from planner.mock_vlm import MockVLMClient


# =============================================================================
# Logging utilities
# =============================================================================

class ColorLogger:
    """Colored console logger for better readability."""

    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m',
        'bold': '\033[1m',
    }

    @classmethod
    def log(cls, msg, color='white', bold=False):
        prefix = cls.COLORS['bold'] if bold else ''
        print(f"{prefix}{cls.COLORS.get(color, '')}{msg}{cls.COLORS['reset']}")

    @classmethod
    def header(cls, msg):
        cls.log(f"\n{'='*70}", 'cyan', bold=True)
        cls.log(f"  {msg}", 'cyan', bold=True)
        cls.log(f"{'='*70}", 'cyan', bold=True)

    @classmethod
    def section(cls, msg):
        cls.log(f"\n--- {msg} ---", 'yellow')

    @classmethod
    def info(cls, msg):
        cls.log(f"[INFO] {msg}", 'white')

    @classmethod
    def success(cls, msg):
        cls.log(f"[OK] {msg}", 'green')

    @classmethod
    def warning(cls, msg):
        cls.log(f"[WARN] {msg}", 'yellow')

    @classmethod
    def error(cls, msg):
        cls.log(f"[ERROR] {msg}", 'red')

    @classmethod
    def debug(cls, msg):
        cls.log(f"[DEBUG] {msg}", 'magenta')

    @classmethod
    def frame(cls, frame_num, msg):
        cls.log(f"[Frame {frame_num:04d}] {msg}", 'blue')


log = ColorLogger()


# =============================================================================
# Debug visualization
# =============================================================================

def render_debug_view(env, agent_pos, planner_path, frame, facing_angle=0,
                      current_prompt="", target_positions=None, output_dir="logs"):
    """Render debug view showing objects and agent position.

    Args:
        env: MockEnv instance
        agent_pos: Agent position [x, y]
        planner_path: Current path from planner (list of [x, y])
        frame: Current frame number
        facing_angle: Agent facing angle in radians
        current_prompt: Current action prompt string
        target_positions: Dict of joint target positions {joint_name: [x, y, z]}
        output_dir: Output directory for debug images
    """
    # Image settings
    IMG_SIZE = 800
    PADDING = 50  # pixels padding around scene

    # Calculate scene bounding box from all objects
    all_points = [[0, 0]]  # Include origin
    all_points.append(list(agent_pos[:2]))  # Include agent

    for name, obj in env._objects.items():
        pos = np.array(obj.get("position", [0, 0, 0])[:2])
        bbox = np.array(obj.get("bbox", []))
        if len(bbox) > 0:
            bbox_2d = bbox[:, :2] + pos
            all_points.extend(bbox_2d.tolist())
        else:
            all_points.append(pos.tolist())

    if planner_path:
        all_points.extend(planner_path)

    all_points = np.array(all_points)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)

    # Add some margin
    margin = 1.0  # 1 meter margin
    min_xy -= margin
    max_xy += margin

    scene_size = max_xy - min_xy
    scale = (IMG_SIZE - 2 * PADDING) / max(scene_size[0], scene_size[1])

    def world_to_pixel(pos):
        """Convert world coordinates to pixel coordinates."""
        px = int(PADDING + (pos[0] - min_xy[0]) * scale)
        py = int(IMG_SIZE - PADDING - (pos[1] - min_xy[1]) * scale)  # Y is flipped
        return (px, py)

    # Create white background
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    # Color palette for objects (BGR)
    colors = [
        (200, 150, 150),  # Light red
        (150, 200, 150),  # Light green
        (150, 150, 200),  # Light blue
        (200, 200, 150),  # Light yellow
        (200, 150, 200),  # Light magenta
        (150, 200, 200),  # Light cyan
        (180, 180, 180),  # Light gray
        (220, 180, 140),  # Light orange
    ]

    # Draw objects (bboxes)
    for idx, (name, obj) in enumerate(env._objects.items()):
        pos = np.array(obj.get("position", [0, 0, 0])[:2])
        bbox = np.array(obj.get("bbox", []))

        color = colors[idx % len(colors)]

        if len(bbox) > 0:
            # Draw bbox as polygon
            bbox_2d = bbox[:, :2] + pos
            pts = np.array([world_to_pixel(p) for p in bbox_2d], dtype=np.int32)
            cv.fillPoly(img, [pts], color)
            cv.polylines(img, [pts], True, (100, 100, 100), 1)
        else:
            # Draw as circle if no bbox
            center_px = world_to_pixel(pos)
            cv.circle(img, center_px, 5, color, -1)
            cv.circle(img, center_px, 5, (100, 100, 100), 1)

    # Draw path if exists
    if planner_path and len(planner_path) > 0:
        path_pts = [world_to_pixel(p) for p in planner_path]
        for i in range(len(path_pts) - 1):
            cv.line(img, path_pts[i], path_pts[i + 1], (0, 165, 255), 2)  # Orange
        # Draw waypoints
        for pt in path_pts:
            cv.circle(img, pt, 4, (0, 165, 255), -1)

    # Draw joint target positions as small blue dots
    if target_positions:
        for joint_name, target_pos in target_positions.items():
            if target_pos is not None and len(target_pos) >= 2:
                target_px = world_to_pixel(target_pos[:2])
                cv.circle(img, target_px, 4, (255, 100, 100), -1)  # Small blue dot

    # Draw agent as triangle showing facing direction
    agent_px = world_to_pixel(agent_pos)
    triangle_size = 12  # pixels
    # Calculate triangle vertices based on facing angle
    # Note: in pixel coords, Y is flipped, so we negate the Y component
    front_x = agent_px[0] + int(triangle_size * 1.5 * np.cos(facing_angle))
    front_y = agent_px[1] - int(triangle_size * 1.5 * np.sin(facing_angle))  # Negate for pixel coords
    # Back left and right vertices (perpendicular to facing)
    back_angle_left = facing_angle + np.pi * 2/3
    back_angle_right = facing_angle - np.pi * 2/3
    back_left_x = agent_px[0] + int(triangle_size * np.cos(back_angle_left))
    back_left_y = agent_px[1] - int(triangle_size * np.sin(back_angle_left))
    back_right_x = agent_px[0] + int(triangle_size * np.cos(back_angle_right))
    back_right_y = agent_px[1] - int(triangle_size * np.sin(back_angle_right))

    triangle_pts = np.array([
        [front_x, front_y],
        [back_left_x, back_left_y],
        [back_right_x, back_right_y]
    ], dtype=np.int32)
    cv.fillPoly(img, [triangle_pts], (0, 0, 255))  # Red filled triangle
    cv.polylines(img, [triangle_pts], True, (0, 0, 0), 2)  # Black border

    # Draw origin
    origin_px = world_to_pixel([0, 0])
    cv.drawMarker(img, origin_px, (0, 200, 0), cv.MARKER_CROSS, 20, 2)

    # Draw scale bar (1 meter)
    scale_bar_len = int(scale)  # 1 meter in pixels
    cv.line(img, (20, IMG_SIZE - 30), (20 + scale_bar_len, IMG_SIZE - 30), (0, 0, 0), 2)
    cv.putText(img, "1m", (20, IMG_SIZE - 35), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw frame info
    cv.putText(img, f"Frame: {frame}", (20, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.putText(img, f"Agent: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})",
               (20, 45), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw current action caption
    if current_prompt:
        # Truncate if too long
        display_prompt = current_prompt[:60] + "..." if len(current_prompt) > 60 else current_prompt
        cv.putText(img, f"Action: {display_prompt}",
                   (20, 65), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 50, 50), 1)

    # Save image (overwrite, no frame number)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "debug.png")
    cv.imwrite(output_path, img)
    return output_path


# =============================================================================
# Test runner
# =============================================================================

class PlannerTestRunner:
    """Test runner that simulates the IsaacGym environment loop."""

    def __init__(self, mock_env, max_frames=5000, verbose=True):
        self.mock_env = mock_env
        self.max_frames = max_frames
        self.verbose = verbose
        self.outputs = []

    def run(self, planner):
        """Run planner test loop."""
        env = self.mock_env
        env_id = env.env_id

        log.section("Starting test loop")
        log.info(f"Max frames: {self.max_frames}")

        # Send init request to planning backend
        init_data = env.get_init_data()
        planner.planner_request_queue.put({
            "type": "init",
            "content": init_data
        })
        log.info("Sent init request to planning backend")

        # Wait for planning backend to initialize (it needs time for VLM calls)
        log.info("Waiting for planning backend to initialize...")
        time.sleep(1.0)

        # Track active actions
        active_actions = {"Standing on the origin"}
        action_frame_counter = {}
        FRAMES_TO_COMPLETE_STAGE2 = 200  # 200 frames * 10ms = 2 seconds minimum before done
        MOVEMENT_SPEED = 0.3  # 0.3 m/s (slower walking for better observation)
        IDLE_TIMEOUT = 1000  # frames before timeout
        DEBUG_FRAME_INTERVAL = 10  # Render debug view every N frames
        last_nav_time = time.time()

        frame = 0
        last_todos_len = 0
        last_prompt = ""
        idle_frames = 0
        start_time = time.time()

        while frame < self.max_frames:
            frame += 1

            # Simulate environment step (handles capture requests)
            env.step()

            # Handle capture requests - log when capture is requested
            if env.is_capturing[env_id] and env.capture_targets[env_id]:
                targets = env.capture_targets[env_id]
                if self.verbose:
                    log.frame(frame, f"Capture requested: {targets}")

            # Simulate navigation: move agent based on target_state_dict set by planner
            current_time = time.time()
            elapsed_nav_time = current_time - last_nav_time
            last_nav_time = current_time

            # Handle turning: if planner requests a heading change, update facing angle
            if planner.target_state_dict.get("heading") is not None:
                target_heading = planner.target_state_dict["heading"]
                if hasattr(target_heading, 'item'):
                    target_heading = target_heading.item()
                current_heading = float(env.current_facing_angle[0])
                # Gradually turn towards target heading
                heading_diff = (target_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                turn_speed = 2.0  # radians per second
                max_turn = turn_speed * elapsed_nav_time
                if abs(heading_diff) <= max_turn:
                    env.set_facing_angle(target_heading)
                else:
                    new_heading = current_heading + np.sign(heading_diff) * max_turn
                    env.set_facing_angle(float(new_heading))

            # Get target from target_state_dict (set by planner's _follow_path)
            target_pos = None
            if "position" in planner.target_state_dict:
                pos_dict = planner.target_state_dict["position"]
                if "traj" in pos_dict:
                    traj = pos_dict["traj"]
                    if hasattr(traj, 'cpu'):
                        target_pos = traj.cpu().numpy()[:2]
                    else:
                        target_pos = np.array(traj)[:2]

            current_pos = env.get_agent_position(env_id)

            if target_pos is not None:
                distance_to_move = MOVEMENT_SPEED * elapsed_nav_time

                direction = target_pos - current_pos
                dist = np.linalg.norm(direction)

                if dist > 0.01:
                    # Update facing angle based on movement direction
                    facing_angle = float(np.arctan2(direction[1], direction[0]))
                    env.set_facing_angle(facing_angle)

                    if dist <= distance_to_move:
                        current_pos = target_pos.copy()
                    else:
                        current_pos = current_pos + (direction / dist) * distance_to_move
                    env.set_agent_position(current_pos)

            # Render debug view every N frames
            if frame % DEBUG_FRAME_INTERVAL == 0:
                agent_pos = env.get_agent_position(env_id)
                path_copy = list(planner.path) if planner.path else []
                facing_angle = float(env.current_facing_angle[0])
                current_prompt = planner.prompt if hasattr(planner, 'prompt') else ""

                # Extract joint target positions from target_state_dict
                target_positions = {}
                if hasattr(planner, 'target_state_dict') and 'position' in planner.target_state_dict:
                    pos_dict = planner.target_state_dict['position']
                    for key, val in pos_dict.items():
                        if val is not None:
                            if hasattr(val, 'cpu'):
                                target_positions[key] = val.cpu().numpy()
                            elif isinstance(val, (list, np.ndarray)):
                                target_positions[key] = np.array(val)

                render_debug_view(env, agent_pos, path_copy, frame,
                                facing_angle=facing_angle,
                                current_prompt=current_prompt,
                                target_positions=target_positions)

            # Send stage_update messages to planning backend for action progress
            # This simulates the stage updates that would come from update_handler
            for action, info in list(action_frame_counter.items()):
                if action.startswith("Walking to "):
                    continue

                start_frame = info.get("stage2_frame", info.get("start_frame", frame))
                elapsed_frames = frame - start_frame
                elapsed_time = elapsed_frames * 0.01  # 10ms per frame

                # Determine stage based on elapsed frames
                if "stage2_frame" in info:
                    # Action has reached stage 2 (acting)
                    if elapsed_frames >= FRAMES_TO_COMPLETE_STAGE2:
                        # Mark as done after completion time
                        stage = "done"
                        fulfilled = True
                    else:
                        stage = "acting"
                        fulfilled = False
                else:
                    stage = "moving"
                    fulfilled = False

                # Send stage update to planning backend
                planner.planner_request_queue.put({
                    "type": "stage_update",
                    "action": action,
                    "stage": stage,
                    "duration": elapsed_time,
                    "fulfilled": fulfilled
                })

                # Also send "done" message when completed (for backwards compatibility)
                if stage == "done" and action not in info.get("done_sent", set()):
                    planner.planner_request_queue.put({
                        "type": "done",
                        "content": action
                    })
                    info["done_sent"] = info.get("done_sent", set()) | {action}
                    if self.verbose:
                        log.frame(frame, f"Action completed: {action}")

            # Initial state action - no longer needed since planning_backend handles this
            # The VLM will send stop("Standing on the origin") command
            # if frame == 1 and "Standing on the origin" in active_actions:
            #     planner.planner_request_queue.put({
            #         "type": "done",
            #         "content": "Standing on the origin"
            #     })
            #     active_actions.remove("Standing on the origin")
            #     if self.verbose:
            #         log.frame(frame, "Sent done for initial state")

            # Update planner
            planner.update(env)

            # Track new actions
            for todo in planner.todos:
                action_name = todo.get("action", "")
                if action_name and action_name not in active_actions:
                    active_actions.add(action_name)
                    if not action_name.startswith("Walking to "):
                        action_frame_counter[action_name] = {"start_frame": frame}
                    if self.verbose:
                        log.frame(frame, f"New action started: {action_name}")

            # Clean up actions that were stopped (removed from planner.todos)
            current_todo_actions = {todo.get("action", "") for todo in planner.todos}
            stopped_actions = []
            for action in list(action_frame_counter.keys()):
                if action not in current_todo_actions:
                    stopped_actions.append(action)
            for action in stopped_actions:
                del action_frame_counter[action]
                if action in active_actions:
                    active_actions.remove(action)
                if self.verbose:
                    log.frame(frame, f"Action stopped: {action}")

            # Update stage2_frame when action reaches execution stage
            for todo in planner.todos:
                action_name = todo.get("action", "")
                stage = todo.get("stage", 0)
                if action_name in action_frame_counter:
                    info = action_frame_counter[action_name]
                    # Stage 2 is reached when position is set and stage >= 2
                    if stage >= 2 and "stage2_frame" not in info:
                        info["stage2_frame"] = frame
                        if self.verbose:
                            log.frame(frame, f"Action reached stage 2: {action_name}")

            # Collect output snapshot
            snapshot = {
                "frame": frame,
                "prompt": planner.prompt,
                "todos_len": len(planner.todos),
                "todos": [t.get("action", "?") for t in planner.todos] if planner.todos else [],
                "state": copy.deepcopy(planner.state),
                "failed": planner.failed,
                "completed": planner.completed,
                "navigating": planner.navigating,
                "position": env.get_agent_position(env_id).tolist(),
            }
            self.outputs.append(snapshot)

            # Log state changes
            if len(planner.todos) != last_todos_len:
                if self.verbose:
                    log.frame(frame, f"Todos changed: {last_todos_len} -> {len(planner.todos)}")
                    for i, todo in enumerate(planner.todos):
                        action = todo.get("action", "?")
                        target = todo.get("target", "?")
                        log.info(f"  [{i}] {action} -> {target}")
                last_todos_len = len(planner.todos)
                idle_frames = 0

            if planner.prompt != last_prompt:
                if self.verbose:
                    log.frame(frame, f"Prompt: {planner.prompt[:70]}...")
                last_prompt = planner.prompt
                idle_frames = 0
            else:
                idle_frames += 1

            # Check for completion
            if planner.failed:
                log.error(f"Planner failed at frame {frame}")
                break

            if planner.completed:
                log.success(f"Task completed successfully at frame {frame}")
                break

            # Check for idle
            if idle_frames > IDLE_TIMEOUT and len(planner.todos) == 0:
                log.warning(f"Idle for {idle_frames} frames with no todos")
                break

            # Progress indicator
            if frame % 500 == 0:
                elapsed = time.time() - start_time
                log.info(f"Progress: frame {frame}, elapsed {elapsed:.1f}s")

            time.sleep(0.01)  # 10ms per frame for better timing

        elapsed = time.time() - start_time
        log.success(f"Test completed: {frame} frames in {elapsed:.1f}s")
        return self.outputs


# =============================================================================
# Main test function
# =============================================================================

def run_single_test(planner_module, mock_env, test_name):
    """Run a single planner test."""
    log.header(f"Testing: {test_name}")

    try:
        mock_env.reset()
        planner = planner_module.LLMPlanner(mock_env, mock_env.env_id)

        log.success("Planner initialized")
        log.info(f"Planning backend PID: {planner.planning_backend.pid}")
        log.info(f"Navigation backend PID: {planner.navigation_backend.pid}")

        runner = PlannerTestRunner(mock_env, max_frames=3000, verbose=True)
        outputs = runner.run(planner)

        # Cleanup
        planner.exit()
        log.success("Planner cleaned up")

        return outputs, True

    except Exception as e:
        log.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return [], False


def main():
    """Main test entry point."""
    log.header("Planner Mock Test")
    log.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Working directory: {os.getcwd()}")

    # Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scene_dir = os.path.join(base_dir, "scene")
    capture_dir = os.path.join(base_dir, "capture")
    task_dir = os.path.join(base_dir, "task")
    image_dir = os.path.join(base_dir, "image")
    response_file = os.path.join(capture_dir, "response.json")

    env_id = 0

    # Verify directories exist
    log.section("Verifying data directories")
    for name, path in [("scene", scene_dir), ("capture", capture_dir),
                       ("task", task_dir), ("image", image_dir)]:
        if os.path.exists(path):
            log.success(f"{name}: {path}")
        else:
            log.error(f"{name} not found: {path}")
            return 1

    # Verify response.json exists
    if os.path.exists(response_file):
        log.success(f"response.json: {response_file}")
    else:
        log.error(f"response.json not found: {response_file}")
        return 1

    # Create logs directory (outside capture)
    os.makedirs("logs", exist_ok=True)

    # Create mock environment
    log.section("Creating MockEnv")
    mock_env = MockEnv(
        env_id=env_id,
        scene_dir=scene_dir,
        capture_dir=capture_dir,
        task_dir=task_dir,
        image_dir=image_dir
    )

    log.success(f"MockEnv created")
    log.info(f"Objects: {len(mock_env.objects[env_id])}")
    log.info(f"Task: {mock_env.prompt[:60]}...")
    log.info(f"Start position: {mock_env._start_point}")

    # Print object summary
    log.section("Scene objects summary")
    mock_env.print_objects_summary()

    # Show capture data availability
    log.section("Capture data availability")
    for obj_name in mock_env._capture_cache:
        cache = mock_env._capture_cache[obj_name]
        front = cache.get("front")
        end = cache.get("end")
        labels_count = len(cache.get("labels", {}))
        log.info(f"  {obj_name}: front={front}, end={end}, labels={labels_count}")

    # Create MockVLMClient from response.json
    log.section("Creating MockVLMClient")
    mock_vlm = MockVLMClient(env_id=env_id, response_file=response_file)
    log.success(f"MockVLMClient created with {len(mock_vlm.responses)} responses")

    # Show first few responses
    for i, resp in enumerate(mock_vlm.responses[:5]):
        preview = resp[:60].replace('\n', ' ') + "..." if len(resp) > 60 else resp.replace('\n', ' ')
        log.debug(f"  [{i}] {preview}")
    if len(mock_vlm.responses) > 5:
        log.debug(f"  ... and {len(mock_vlm.responses) - 5} more responses")

    # Run new planner test
    log.header("Testing New Planner")
    import planner.llm_planner as new_planner
    outputs_new, success_new = run_single_test(new_planner, mock_env, "new_planner")

    if not success_new:
        log.error("New planner test failed!")
        return 1

    # Summary
    log.header("Test Summary")
    log.info(f"New planner: {len(outputs_new)} frames")

    if outputs_new:
        final = outputs_new[-1]
        log.info(f"Final state:")
        log.info(f"  - Failed: {final.get('failed')}")
        log.info(f"  - Completed: {final.get('completed')}")
        log.info(f"  - Todos: {final.get('todos_len')}")
        log.info(f"  - Position: {final.get('position')}")
        log.info(f"  - Prompt: {final.get('prompt', '')[:50]}...")

    # Check for generated files (in logs directory, not capture)
    log.section("Generated files")
    for fname in ["logs/llm_0.txt", "logs/navigation_0.png", "logs/debug.png"]:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            log.success(f"{fname} ({size} bytes)")
        else:
            log.warning(f"{fname} not found")

    log.header("Test Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

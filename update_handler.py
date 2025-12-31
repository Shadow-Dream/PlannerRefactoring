"""Update handler for processing todo actions in LLMPlanner."""
import torch
import numpy as np
import copy
import re
import shapely as sl
from shapely import ops

from planner.core import quaternion
from planner.core.action import merge_action, action_local_to_world
from planner.utils.geometry_utils import get_convex_hull
from planner.prompts import SURFACE_TO_JOINT


def process_todos(planner, env, objects, parents, position, change_state,
                  has_action_done, clear_pending):
    """Process todo list actions for the planner."""
    actions = []
    for action in planner.todos:
        actions.append(action)

    def set_stage(stage):
        actions[0]["stage"] = stage
        planner.change_state_next = True
        if stage != 0:
            clear_pending()

    def update_turn_tick():
        actions[0]["turn_tick"] = actions[0].get("turn_tick", 0) + 1

    def reset_turn_tick():
        actions[0]["turn_tick"] = 0

    # Initialize stage if not set
    if "stage" not in actions[0]:
        planner.path = []
        if planner.navigating:
            if not planner.navigator_result_queue.empty():
                planner.navigator_result_queue.get()
                planner.navigating = False
            else:
                return
        set_stage(0)

    # Merge concurrent behaviors
    action = merge_action([action_local_to_world(action, objects) for action in actions])

    def reset_stage():
        set_stage(0)
        action["stage"] = 0
        act = actions[0]
        if "fulfilled" in act:
            del act["fulfilled"]
        if "fulfilled_tick" in act:
            del act["fulfilled_tick"]
        if "fulfilled" in action:
            del action["fulfilled"]
        if "fulfilled_tick" in action:
            del action["fulfilled_tick"]

    # Check if action or position changed significantly
    if (action["action"] != planner.last_action
        and action["position"] is not None
        and (planner.last_position is None
             or np.linalg.norm(np.array(action["position"])[:2] - np.array(planner.last_position)[:2]) > 0.2)):
        reset_stage()

    planner.last_action = action["action"]
    planner.last_position = action["position"]

    # Determine target
    if action["place"] and action["at"] is not None and len(action["at"]) > 0:
        target = action["at"][0]
    else:
        target = action["target"]

    if target not in objects:
        target = None

    # Clean up real_target_position
    if "real_target_position" in planner.target_state_dict:
        del planner.target_state_dict["real_target_position"]

    # Build ongoing prompts
    ongoing_prompts = []
    for act in actions:
        if act["position"] is None:
            ongoing_prompts.append(act["action"])
        else:
            break

    def merge_ongoing_prompts(prompt, keep_default=False):
        if keep_default:
            prompts = ongoing_prompts + [prompt]
        elif len(ongoing_prompts) == 0:
            prompts = [prompt]
        else:
            prompts = copy.deepcopy(ongoing_prompts)
        prompts = [p.strip().lower() for p in prompts]
        prompt = "A person is "
        if len(prompts) == 1:
            prompt += prompts[0]
        elif len(prompts) == 2:
            prompt += " and ".join(prompts)
        else:
            prompt += ", and ".join(prompts[:-1]) + ", and " + prompts[-1]
        return prompt

    # Process based on current stage
    if action["stage"] == 0:
        _process_stage_0(planner, env, objects, parents, position, action, actions,
                        target, set_stage, clear_pending, merge_ongoing_prompts,
                        change_state)
    elif action["stage"] == 1:
        _process_stage_1(planner, env, action, merge_ongoing_prompts, set_stage,
                        update_turn_tick, reset_turn_tick)
    elif action["stage"] == 2:
        _process_stage_2(planner, env, objects, position, action, actions, target,
                        merge_ongoing_prompts, change_state)


def _process_stage_0(planner, env, objects, parents, position, action, actions,
                    target, set_stage, clear_pending, merge_ongoing_prompts,
                    change_state):
    """Process navigation stage."""
    if planner.two_hand_grab_target is not None and target is not None:
        _handle_two_hand_navigation(planner, env, objects, position, action, actions,
                                   target, set_stage)
    else:
        # Non-interactive behavior inplace
        if target is None and action["position"] is None:
            set_stage(2)
            return

        # Decide target position
        if action["position"] is None:
            target_position = np.array(objects[target]["position"], dtype=np.float32)[:2]
            target_position += np.array(objects[target]["bbox"], dtype=np.float32)[::2, :2].mean(0)
            target_position = target_position.tolist()
            target_direction = [0, 0]
        else:
            target_position = action["position"]
            target_direction = np.array(target_position, dtype=np.float32) - np.array(objects[target]["position"][:2], dtype=np.float32)
            target_direction = target_direction / np.linalg.norm(target_direction)
            target_direction = target_direction.tolist()

        last_position = target_position
        if planner.path:
            last_position = planner.path[-1]

        # Check if navigation finished
        if (action["position"] is not None and
            np.linalg.norm(np.array(position, dtype=np.float32) - np.array(last_position, dtype=np.float32)) < 0.5):
            print("==================== arrive ====================")
            if action["facing"] is not None:
                set_stage(1)
            else:
                set_stage(2)
        elif (action["position"] is None and len(planner.todos) > 1 and
              np.linalg.norm(np.array(position, dtype=np.float32) - np.array(last_position, dtype=np.float32)) < 0.5):
            planner.todos.pop(0)
            clear_pending()
        else:
            # Perform path planning
            if planner.navigating:
                if not planner.navigator_result_queue.empty():
                    path = planner.navigator_result_queue.get()
                    if isinstance(path, dict):
                        planner.failed = True
                        return
                    if len(path) > 0:
                        planner.path = path
                    planner.navigating = False
            else:
                planner.navigator_request_queue.put([objects, parents, position, target_position, target_direction, target])
                planner.navigating = True

        _perform_navigation(planner, env, objects, position, action, target,
                           target_position, merge_ongoing_prompts, change_state)


def _handle_two_hand_navigation(planner, env, objects, position, action, actions,
                               target, set_stage):
    """Handle navigation when carrying a two-handed object."""
    if not action.get("set_two_hand_place", False):
        instance = objects[target]
        ins_position = np.array(instance["position"])
        ins_rotation = np.array(instance["rotation"])
        shape = instance["shape"]
        quat = ins_rotation[None][None]
        quat = np.broadcast_to(quat, (shape.shape[0], shape.shape[1], 4))
        shape = quaternion.qrot_np(quat, shape)
        shape += ins_position[None][None]
        shape = shape[:, :, :2]
        object_polygon = None
        for box in shape:
            polygon = get_convex_hull(box)
            if polygon is None:
                continue
            if object_polygon is None:
                object_polygon = polygon
            else:
                object_polygon = object_polygon.union(polygon)
        object_polygon = object_polygon.buffer(0.75, join_style="mitre")
        cur_point = sl.Point(position[:2])
        if object_polygon.contains(cur_point):
            set_stage(2)
            return
        closest_point = ops.nearest_points(object_polygon, cur_point)[0]
        closest_point = [closest_point.x, closest_point.y]
        action["two_hand_place"] = closest_point
        actions[0]["two_hand_place"] = closest_point
        actions[0]["set_two_hand_place"] = True
        actions[0]["place"] = True

    target_position = list(action["two_hand_place"])
    target_direction = np.array(target_position, dtype=np.float32) - np.array(position, dtype=np.float32)
    target_distance = np.linalg.norm(target_direction)
    target_direction /= target_distance
    angle = np.arctan2(target_direction[1], target_direction[0])

    if target_distance < 0.5:
        set_stage(2)
        return

    target_position = torch.tensor(list(target_position) + [0], dtype=torch.float32, device=env.device)
    planner.target_state_dict["position"] = {"traj": target_position}
    planner.prompt = "A person is walking."
    planner.target_state_dict["heading"] = None


def _perform_navigation(planner, env, objects, position, action, target,
                       target_position, merge_ongoing_prompts, change_state):
    """Perform navigation along the path."""
    is_decided = False

    if action["position"] is None:
        # Stop near object if no precise position
        instance = objects[target]
        ins_position = np.array(instance["position"])
        ins_rotation = np.array(instance["rotation"])
        shape = instance["shape"]
        quat = ins_rotation[None][None]
        quat = np.broadcast_to(quat, (shape.shape[0], shape.shape[1], 4))
        shape = quaternion.qrot_np(quat, shape)
        shape += ins_position[None][None]
        shape = shape[:, :, :2]
        object_polygon = None
        for box in shape:
            polygon = get_convex_hull(box)
            if polygon is None:
                continue
            if object_polygon is None:
                object_polygon = polygon
            else:
                object_polygon = object_polygon.union(polygon)
        object_polygon = object_polygon.buffer(1, join_style="mitre")
        cur_point = sl.Point(position[:2])
        if object_polygon.contains(cur_point):
            if planner.prompt != merge_ongoing_prompts("standing still"):
                change_state = True
            is_decided = True
            target_position = position.tolist()
            if len(target_position) == 3:
                target_position[-1] = 0
            else:
                target_position += [0]
            target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
            planner.target_state_dict["position"] = {"traj": target_position}
            planner.target_state_dict["velocity"] = None
            planner.target_state_dict["heading"] = None
            planner.prompt = merge_ongoing_prompts("standing still")

    if not is_decided and planner.path and len(planner.path) >= 1:
        _follow_path(planner, env, position, target_position, merge_ongoing_prompts, change_state)
    elif not is_decided:
        # Wait at current position if path not available
        target_position = position.tolist() + [0]
        target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
        planner.target_state_dict["position"] = {"traj": target_position}
        planner.target_state_dict["velocity"] = torch.zeros_like(target_position)
        planner.target_state_dict["heading"] = None
        planner.prompt = merge_ongoing_prompts("standing still")


def _follow_path(planner, env, position, target_position, merge_ongoing_prompts, change_state):
    """Follow the planned path.

    This function:
    1. Finds the nearest segment to current position
    2. Trims path to path[i:] (removes passed segments, keeps segment start)
    3. Sets a nearby target along the path (not too far to ensure following the path)
    """
    STEP_DISTANCE = 0.5  # Target distance along path per step

    if planner.path and len(planner.path) >= 2:
        # Step 1: Find nearest segment i (segment from path[i] to path[i+1])
        nearest_index = 0
        nearest_distance = float("inf")
        nearest_proj_point = None
        nearest_t = 0

        for i in range(len(planner.path) - 1):
            p1 = np.array(planner.path[i], dtype=np.float32)
            p2 = np.array(planner.path[i + 1], dtype=np.float32)

            # Project position onto segment [p1, p2]
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)

            if seg_len_sq < 1e-10:
                proj_point = p1
                t = 0
            else:
                t = np.clip(np.dot(position[:2] - p1, seg_vec) / seg_len_sq, 0, 1)
                proj_point = p1 + t * seg_vec

            dist_to_seg = np.linalg.norm(position[:2] - proj_point)

            if dist_to_seg < nearest_distance:
                nearest_distance = dist_to_seg
                nearest_index = i
                nearest_proj_point = proj_point
                nearest_t = t

        # Step 2: Trim path = path[i:] (remove segments before current one)
        if nearest_index > 0:
            planner.path = planner.path[nearest_index:]
            # Recalculate projection for new segment 0
            p1 = np.array(planner.path[0], dtype=np.float32)
            p2 = np.array(planner.path[1], dtype=np.float32)
            seg_vec = p2 - p1
            seg_len_sq = np.dot(seg_vec, seg_vec)
            if seg_len_sq >= 1e-10:
                nearest_t = np.clip(np.dot(position[:2] - p1, seg_vec) / seg_len_sq, 0, 1)
                nearest_proj_point = p1 + nearest_t * seg_vec

        # Step 3: Calculate target position along the path
        # Start from projection point, walk STEP_DISTANCE along the path
        remaining_dist = STEP_DISTANCE
        current_seg_idx = 0
        p1 = np.array(planner.path[0], dtype=np.float32)
        p2 = np.array(planner.path[1], dtype=np.float32)

        # Start from where we are on the segment
        current_point = nearest_proj_point if nearest_proj_point is not None else p1

        while remaining_dist > 0 and current_seg_idx < len(planner.path) - 1:
            p1 = np.array(planner.path[current_seg_idx], dtype=np.float32)
            p2 = np.array(planner.path[current_seg_idx + 1], dtype=np.float32)

            dist_to_p2 = np.linalg.norm(p2 - current_point)

            if dist_to_p2 <= remaining_dist:
                # Can reach end of this segment
                remaining_dist -= dist_to_p2
                current_point = p2
                current_seg_idx += 1
            else:
                # Target is within this segment
                direction = p2 - current_point
                direction = direction / np.linalg.norm(direction)
                current_point = current_point + direction * remaining_dist
                remaining_dist = 0

        target_position = current_point

        # Check if this is the final segment and we're close to the endpoint
        is_final_segment = len(planner.path) == 2
        final_target = np.array(planner.path[-1], dtype=np.float32)
        dist_to_final = np.linalg.norm(position[:2] - final_target)

        # If close to final target and it's the last segment, target the final point
        if is_final_segment and dist_to_final < STEP_DISTANCE:
            target_position = final_target

    elif planner.path and len(planner.path) == 1:
        target_position = np.array(planner.path[0], dtype=np.float32)
        is_final_segment = True
        dist_to_final = np.linalg.norm(position[:2] - target_position)
    else:
        # No path
        target_position = position[:2]
        is_final_segment = False
        dist_to_final = 0

    target_direction = target_position - position[:2]
    target_distance = np.linalg.norm(target_direction)

    # Advance to next waypoint if close (but not for final segment)
    if target_distance < 0.1 and planner.path and len(planner.path) > 2:
        planner.path = planner.path[1:]
        target_position = np.array(planner.path[1], dtype=np.float32)
        target_direction = target_position - position[:2]
        target_distance = np.linalg.norm(target_direction)

    if target_distance < 1e-6:
        target_distance = 1e-6
    target_direction = target_direction / target_distance
    real_target_position = target_position

    angle = np.arctan2(target_direction[1], target_direction[0])
    delta_angle = abs(env.get_heading(angle, planner.env_id))

    if delta_angle < np.pi / 4:
        # Walk when facing the waypoint
        # For final segment, walk all the way to endpoint (no 0.5m threshold)
        if target_distance < 0.1 and not is_final_segment:
            target_position = list(position[:2]) + [0]
            target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
            planner.target_state_dict["position"] = {"traj": target_position}
            planner.target_state_dict["velocity"] = torch.zeros_like(target_position)
            planner.target_state_dict["heading"] = None
            planner.prompt = merge_ongoing_prompts("standing still")
        elif is_final_segment and dist_to_final < 0.1:
            # Reached final destination
            target_position = list(position[:2]) + [0]
            target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
            planner.target_state_dict["position"] = {"traj": target_position}
            planner.target_state_dict["velocity"] = torch.zeros_like(target_position)
            planner.target_state_dict["heading"] = None
            planner.prompt = merge_ongoing_prompts("standing still")
        else:
            env_target_position = env.target_state_dict[planner.env_id]["position"]
            env_prompt = env.actual_used_prompts[planner.env_id]
            if "traj" in env_target_position and "turning" not in env_prompt:
                env_target_position = env_target_position["traj"].cpu().numpy()[:2]
                env_target_distance = np.linalg.norm(env_target_position - position[:2])
                if env_target_distance < 0.5 and target_distance > 0.5:
                    change_state = True

            target_position = list(target_position) + [0]
            target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
            planner.target_state_dict["position"] = {"traj": target_position}
            planner.target_state_dict["velocity"] = None
            planner.target_state_dict["real_target_position"] = real_target_position
            planner.target_state_dict["heading"] = None
            planner.prompt = merge_ongoing_prompts("walking")
    else:
        # Turn when not facing the waypoint
        target_position = position.tolist() + [0]
        target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
        planner.target_state_dict["position"] = {"traj": target_position}
        planner.target_state_dict["velocity"] = torch.zeros_like(target_position)
        delta_angle = env.get_heading(angle, planner.env_id)
        sub_angle = delta_angle / 4 if delta_angle < np.pi * (2 / 3) else (np.pi / 4 + delta_angle / 4)
        angle -= sub_angle
        planner.target_state_dict["heading"] = angle
        planner.prompt = merge_ongoing_prompts("slowly turning around in place")


def _process_stage_1(planner, env, action, merge_ongoing_prompts, set_stage,
                    update_turn_tick, reset_turn_tick):
    """Process turning stage."""
    planner.prompt = merge_ongoing_prompts("slowly turning around in place")
    target_position = np.array(action["position"])
    target_position = target_position.tolist() + [0]
    target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
    planner.target_state_dict["position"] = {"traj": target_position}
    planner.target_state_dict["velocity"] = torch.zeros_like(target_position)
    planner.target_state_dict["heading"] = action["facing"]
    delta_angle = (env.current_facing_angle[planner.env_id] - action["facing"] + np.pi) % (2 * np.pi) - np.pi

    if abs(delta_angle) < np.pi / 6:
        if planner.prompt != merge_ongoing_prompts("standing still"):
            planner.prompt = merge_ongoing_prompts("standing still")
        update_turn_tick()
        if action.get("turn_tick", 0) > 10:
            set_stage(2)
    else:
        reset_turn_tick()


def _process_stage_2(planner, env, objects, position, action, actions, target,
                    merge_ongoing_prompts, change_state):
    """Process action execution stage."""
    planner.prompt = action["action"]
    planner.prompt = re.sub(r'\d', '', planner.prompt)
    planner.prompt = planner.prompt[:1].lower() + planner.prompt[1:]
    planner.prompt = "A person is " + planner.prompt

    take_failed = False
    take_done = False

    # Handle two-hand take
    if (action["touch"]
        and action["take"]
        and "left_hand" in action["contact_points"]
        and "right_hand" in action["contact_points"]):
        take_failed, take_done = _handle_two_hand_take(planner, env, objects, position,
                                                        action, actions, target, change_state)
    elif action["place"]:
        _handle_place_action(planner, env, objects, position, action, target)
    elif action["touch"]:
        _handle_touch_action(planner, env, action, target)
    else:
        _handle_non_touch_action(planner, env, position, action)

    # Termination detection
    _check_termination(planner, env, objects, position, actions, action, target,
                      take_done, take_failed)


def _process_contact_targets(contact_targets, contact_points, objects, target, action):
    """Process contact targets to add pelvis position if needed."""
    if ("left_wrist" in contact_points or "right_wrist" in contact_points) and "pelvis" not in contact_points:
        t_position = np.array(objects[target]["position"])[:2]
        a_position = np.array(action["position"])[:2]
        t_direction = t_position - a_position
        t_direction = t_direction / np.linalg.norm(t_direction)
        wrist_distance = 0.5

        if "left_wrist" in contact_points and "right_wrist" in contact_points:
            l_position = np.array(contact_targets[contact_points.index("left_wrist")])
            r_position = np.array(contact_targets[contact_points.index("right_wrist")])
            wrist_position = (l_position + r_position) / 2
            wrist_distance = max(0.5 - np.linalg.norm(l_position - r_position) / 2, 0.3)
        elif "left_wrist" in contact_points:
            wrist_position = np.array(contact_targets[contact_points.index("left_wrist")])
        else:
            wrist_position = np.array(contact_targets[contact_points.index("right_wrist")])

        wrist_height = wrist_position[2]
        wrist_position = wrist_position[:2]

        wrist_distance_sub = max(wrist_height - 1.5, 0) / 2
        wrist_distance = max(wrist_distance - wrist_distance_sub, 0.3)
        wrist_distance = wrist_distance * 5 / 3

        pelvis_position = np.zeros([3])
        pelvis_position[:2] = wrist_position - t_direction * wrist_distance
        pelvis_position[2] = min(max(wrist_height + 0.4, 0.6), 0.9)

        contact_points.append("pelvis")
        contact_targets.append(pelvis_position.tolist())

    return contact_targets, contact_points


def _handle_two_hand_take(planner, env, objects, position, action, actions, target, change_state):
    """Handle two-hand take action."""
    planner.enable_lefthand_ik = True
    planner.enable_righthand_ik = True

    def update_take_stage(new_stage):
        action["take_stage"] = new_stage
        for act in actions:
            if act["take"]:
                act["take_stage"] = new_stage
                break

    def set_take_tick(value=0):
        action["take_ticks"] = value
        for act in actions:
            if act["take"]:
                act["take_ticks"] = value
                break

    def update_take_stand_tick():
        action["take_stand_tick"] = action.get("take_stand_tick", 0) + 1
        for act in actions:
            if act["take"]:
                act["take_stand_tick"] = act.get("take_stand_tick", 0) + 1
                break

    def update_take_tick():
        action["take_ticks"] = action.get("take_ticks", 0) + 1
        for act in actions:
            if act["take"]:
                act["take_ticks"] = act.get("take_ticks", 0) + 1
                break

    if "take_stage" not in action:
        update_take_stage(0)

    target = action["target"]
    target_instance = objects[target]

    agent_position = env.get_joint_info(planner.env_id, "pelvis")[0]
    agent_position = torch.tensor(agent_position).cuda().float()
    box_position = torch.tensor(target_instance["position"]).cuda().float()
    box_rotation = torch.tensor(target_instance["rotation"]).cuda().float()

    disc_box_directions = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    disc_box_directions = torch.tensor(disc_box_directions).cuda().float()
    disc_box_directions = quaternion.qrot(box_rotation.unsqueeze(0).repeat(4, 1), disc_box_directions)
    disc_box_directions = disc_box_directions[..., :2]
    disc_box_directions = disc_box_directions / (disc_box_directions.norm(dim=-1, keepdim=True) + 1e-5)

    box_bbox = torch.tensor(target_instance["bbox"]).cuda().float()
    box_bbox = quaternion.qrot(box_rotation.unsqueeze(0).repeat(8, 1), box_bbox)
    box_bbox += box_position
    box_position = box_bbox.mean(0)

    box_direction = box_position - agent_position
    bd_2d = box_direction[:2]
    box_f = bd_2d / bd_2d.norm()
    box_r = torch.zeros_like(box_f)
    box_r[0] = box_f[1]
    box_r[1] = -box_f[0]

    disc_box_f = disc_box_directions[(disc_box_directions * box_f[:2]).sum(-1).argmax()]

    take_failed = False
    take_done = False

    if action["take_stage"] == 0:
        take_failed, take_done = _two_hand_take_stage_0(
            planner, env, action, agent_position, box_position, box_f, disc_box_f,
            update_take_stage, update_take_tick, set_take_tick, change_state
        )
    elif action["take_stage"] == 2:
        take_failed, take_done = _two_hand_take_stage_2(
            planner, env, action, actions, agent_position, box_position, box_rotation,
            box_bbox, box_f, disc_box_f, update_take_stage, update_take_tick, set_take_tick
        )
    elif action["take_stage"] == 3:
        take_failed, take_done = _two_hand_take_stage_3(
            planner, env, objects, action, target, agent_position, box_position,
            box_rotation, box_bbox, box_f, update_take_stand_tick, update_take_tick, set_take_tick
        )

    return take_failed, take_done


def _two_hand_take_stage_0(planner, env, action, agent_position, box_position, box_f,
                           disc_box_f, update_take_stage, update_take_tick, set_take_tick,
                           change_state):
    """Stage 0: Approach the box."""
    box_loc = box_position[:2] - disc_box_f * 0.5
    box_loc_z = min(max(box_position[2] + 0.4, 0.6), 0.9)
    box_loc = torch.tensor([box_loc[0], box_loc[1], box_loc_z]).cuda().float()
    box_facing = np.arctan2(box_f[1], box_f[0])
    pelvis_position = box_loc.clone()

    planner.target_state_dict["position"] = {"pelvis": pelvis_position}
    planner.target_state_dict["velocity"] = None
    planner.target_state_dict["heading"] = box_facing
    planner.grab_position_left = None
    planner.grab_position_right = None
    planner.left_drop_target = None
    planner.right_drop_target = None

    delta_angle = (env.current_facing_angle[planner.env_id] - box_facing + np.pi) % (2 * np.pi) - np.pi
    if (box_loc - agent_position)[:2].norm() < 0.2 and abs(delta_angle) < np.pi / 6:
        planner.prompt = "A person is standing still."
        if action.get("take_ticks", 0) == 0:
            change_state = True
        update_take_tick()
        if action["take_ticks"] > 10:
            update_take_stage(2)
            planner.smooth_reference = True
            planner.change_state_next = True
            set_take_tick()
    elif (box_loc - agent_position)[:2].norm() < 0.2:
        planner.prompt = "A person is turning around slowly."
        set_take_tick()
    else:
        planner.prompt = "A person is walking slowly."
        set_take_tick()

    return False, False


def _two_hand_take_stage_2(planner, env, action, actions, agent_position, box_position,
                           box_rotation, box_bbox, box_f, disc_box_f,
                           update_take_stage, update_take_tick, set_take_tick):
    """Stage 2: Reach for the box."""
    left_wrist_position = env.get_joint_info(planner.env_id, "left_wrist")[0]
    right_wrist_position = env.get_joint_info(planner.env_id, "right_wrist")[0]
    left_wrist_position = torch.tensor(left_wrist_position).cuda().float()
    right_wrist_position = torch.tensor(right_wrist_position).cuda().float()
    left_wrist_position -= box_position
    right_wrist_position -= box_position

    box_x = box_bbox[6] - box_bbox[0]
    box_y = box_bbox[2] - box_bbox[0]
    box_z = box_bbox[1] - box_bbox[0]
    box_dirs = torch.stack([box_x, box_y])

    dir_index = (box_dirs[:, :2] * (right_wrist_position - left_wrist_position)[:2]).abs().sum(-1).argmin()
    axis0 = box_dirs[dir_index]
    axis1 = box_z
    axis2 = box_dirs[(dir_index + 1) % 2]

    axis0v = axis0 / axis0.norm()
    axis1v = axis1 / axis1.norm()
    axis2v = axis2 / axis2.norm()

    lwp0 = (left_wrist_position * axis0v).sum()
    lwp1 = (left_wrist_position * axis1v).sum()
    lwp2 = (left_wrist_position * axis2v).sum()
    rwp0 = (right_wrist_position * axis0v).sum()
    rwp1 = (right_wrist_position * axis1v).sum()
    rwp2 = (right_wrist_position * axis2v).sum()

    margin = 0.3

    box_x = box_bbox[6] - box_bbox[0]
    box_y = box_bbox[2] - box_bbox[0]
    box_x /= 2
    box_y /= 2
    box_dirs_4 = torch.stack([box_x, box_y, -box_x, -box_y])
    dir_index = (box_dirs_4[:, :2] * box_f).sum(-1).argmin()

    box_dirs_list = [
        [-box_y, box_y],
        [box_x, -box_x],
        [box_y, -box_y],
        [-box_x, box_x]
    ]
    lpos, rpos = box_dirs_list[dir_index]
    lpos = lpos + lpos / lpos.norm() * margin
    rpos = rpos + rpos / rpos.norm() * margin
    lpos += box_position
    rpos += box_position

    box_loc = box_position[:2] - disc_box_f * 0.3
    box_loc_z = min(max(box_position[2] + 0.4, 0.6), 0.9)
    box_loc = torch.tensor([box_loc[0], box_loc[1], box_loc_z]).cuda().float()
    box_facing = np.arctan2(box_f[1], box_f[0])
    pos = box_position + 0
    pos[2] = box_bbox[:, 2].min()
    pelvis_position = box_loc.clone()

    planner.target_state_dict["position"] = {
        "left_wrist": pos,
        "right_wrist": pos,
        "pelvis": pelvis_position
    }
    planner.grab_position_left = lpos
    planner.grab_position_right = rpos
    planner.target_state_dict["velocity"] = None
    planner.target_state_dict["heading"] = box_facing
    planner.prompt = "A person is lifting a box from the ground."

    update_take_tick()
    if action["take_ticks"] > 360:
        update_take_stage(0)
        planner.change_state_next = True

    if (-0.1 < lwp0 < 0.1 and -0.1 < lwp1 < 0.1
        and -0.1 < rwp0 < 0.1 and -0.1 < rwp1 < 0.1
        and lwp2 * rwp2 < 0):
        update_take_stage(3)
        planner.change_state_next = True
        print("================== Standing Up ===================")
        set_take_tick()
        pos = box_position
        pos[2] = box_bbox[:, 2].min()
        pos = pos - agent_position
        agent_rotation = env.current_facing_angle[planner.env_id]
        agent_forward = torch.tensor([np.cos(agent_rotation), np.sin(agent_rotation)]).cuda().float()
        agent_right = torch.tensor([np.sin(agent_rotation), -np.cos(agent_rotation)]).cuda().float()

        pos_f = (pos[:2] * agent_forward).sum()
        pos_r = (pos[:2] * agent_right).sum()

        pos[0] = max(pos_f, 0.75)
        pos[1] = pos_r

        actions[0]["box_pos"] = pos.cpu().numpy().tolist()

    return False, False


def _two_hand_take_stage_3(planner, env, objects, action, target, agent_position,
                           box_position, box_rotation, box_bbox, box_f,
                           update_take_stand_tick, update_take_tick, set_take_tick):
    """Stage 3: Stand up with the box."""
    box_facing = np.arctan2(box_f[1], box_f[0])

    agent_rotation = env.current_facing_angle[planner.env_id]
    agent_forward = torch.tensor([np.cos(agent_rotation), np.sin(agent_rotation)]).cuda().float()
    agent_right = torch.tensor([np.sin(agent_rotation), -np.cos(agent_rotation)]).cuda().float()
    pos = torch.tensor(action["box_pos"]).cuda().float()
    pos[:2] = pos[0] * agent_forward + pos[1] * agent_right
    pos += agent_position

    update_take_stand_tick()
    if action["take_stand_tick"] < 10:
        planner.target_state_dict["position"] = {
            "left_wrist": pos,
            "right_wrist": pos
        }
    else:
        if action["take_stand_tick"] == 10:
            pass  # change_state would be set in caller
        pelvis_position = agent_position.clone()
        pelvis_position[:2] = pelvis_position[:2] + agent_forward * 0.5
        pelvis_position[2] = 0.9
        planner.target_state_dict["position"] = {
            "pelvis": pelvis_position
        }

    planner.two_hand_grab_target = target
    planner.two_hand_grab_position = copy.deepcopy(action["box_pos"])

    planner.target_state_dict["velocity"] = None
    planner.target_state_dict["heading"] = box_facing
    planner.prompt = "A person is standing up."

    left_wrist_position = env.get_joint_info(planner.env_id, "left_hand")[0]
    right_wrist_position = env.get_joint_info(planner.env_id, "right_hand")[0]
    hand_distance = left_wrist_position - right_wrist_position
    min_box_z = box_bbox[:, 2].min()
    hand_distance = np.linalg.norm(hand_distance)

    take_failed = False
    take_done = False

    if hand_distance < 0.25:
        take_failed = True
        planner.failed = True
    else:
        if min_box_z > 0.1:
            update_take_tick()
            if action["take_ticks"] > 10:
                take_done = True
        else:
            set_take_tick()

    return take_failed, take_done


def _handle_place_action(planner, env, objects, position, action, target):
    """Handle place action."""
    if planner.two_hand_grab_target is None:
        if action.get("fulfilled", False):
            target_position = position.tolist() + [0.9]
            target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
            planner.target_state_dict["position"] = {"pelvis": target_position}
            planner.target_state_dict["velocity"] = None
            planner.prompt = "A person is standing still."
        else:
            contact_points = action["contact_points"]
            try:
                contact_points = [SURFACE_TO_JOINT[p] for p in contact_points]
            except:
                print(contact_points, flush=True)
            contact_targets = action["contact_targets"]
            target_object = action["target"]

            contact_targets, contact_points = _process_contact_targets(
                contact_targets, contact_points, objects, target, action
            )

            if len(contact_points) == len(contact_targets):
                planner.target_state_dict["position"] = {
                    p: torch.tensor(t, dtype=torch.float32, device=env.device)
                    for p, t in zip(contact_points, contact_targets)
                }
                planner.target_state_dict["velocity"] = None
            else:
                return

            planner.target_state_dict["heading"] = action["facing"]

            if "left_wrist" in contact_points:
                planner.enable_lefthand_ik = True
                position_left = contact_targets[contact_points.index("left_wrist")]
                position_left = torch.tensor(position_left).cuda().float()
                planner.grab_position_left = position_left
                if env.left_slot[planner.env_id] == target_object:
                    planner.left_drop_target = position_left

            if "right_wrist" in contact_points:
                planner.enable_righthand_ik = True
                position_right = contact_targets[contact_points.index("right_wrist")]
                position_right = torch.tensor(position_right).cuda().float()
                planner.grab_position_right = position_right
                if env.right_slot[planner.env_id] == target_object:
                    planner.right_drop_target = position_right
    else:
        instance = objects[target]
        ins_position = torch.tensor(instance["position"]).cuda().float()
        ins_bbox = torch.tensor(instance["bbox"]).cuda().float()
        ins_position = ins_position + ins_bbox.mean(0)
        ins_position = ins_position.cpu().numpy()
        ins_direction = ins_position[:2] - position[:2]
        ins_angle = np.arctan2(ins_direction[1], ins_direction[0])

        agent_position = env.get_joint_info(planner.env_id, "pelvis")[0]
        agent_position = torch.tensor(agent_position).cuda().float()
        agent_position[2] = 0.5
        planner.target_state_dict["position"] = {"pelvis": agent_position}
        planner.target_state_dict["velocity"] = None
        planner.target_state_dict["heading"] = ins_angle
        planner.prompt = "A person is placing a box on the ground."


def _handle_touch_action(planner, env, action, target):
    """Handle touch action."""
    contact_points = action["contact_points"]
    contact_targets = action["contact_targets"]
    contact_points = [SURFACE_TO_JOINT[p] for p in contact_points]

    contact_targets, contact_points = _process_contact_targets(
        contact_targets, contact_points, {target: {"position": [0, 0, 0]}}, target, action
    )

    if len(contact_points) == len(contact_targets):
        planner.target_state_dict["position"] = {
            p: torch.tensor(t, dtype=torch.float32, device=env.device)
            for p, t in zip(contact_points, contact_targets)
        }
        planner.target_state_dict["velocity"] = None
    else:
        return

    planner.target_state_dict["heading"] = action["facing"]

    if "left_wrist" in contact_points:
        planner.enable_lefthand_ik = True
        planner.grab_position_left = torch.tensor(
            contact_targets[contact_points.index("left_wrist")]
        ).cuda().float()

    if "right_wrist" in contact_points:
        planner.enable_righthand_ik = True
        planner.grab_position_right = torch.tensor(
            contact_targets[contact_points.index("right_wrist")]
        ).cuda().float()

    if action["take"]:
        if "left_wrist" in contact_points:
            planner.left_take_target = action["target"]
        if "right_wrist" in contact_points:
            planner.right_take_target = action["target"]


def _handle_non_touch_action(planner, env, position, action):
    """Handle non-touch action."""
    if action["position"] is None:
        target_position = position.tolist() + [0]
    else:
        target_position = np.array(action["position"])
        target_position = target_position.tolist() + [0]
    target_position = torch.tensor(target_position, dtype=torch.float32, device=env.device)
    planner.target_state_dict["position"] = {"traj": target_position}
    planner.target_state_dict["velocity"] = None
    planner.target_state_dict["heading"] = action["facing"]


def _check_termination(planner, env, objects, position, actions, action, target,
                      take_done, take_failed):
    """Check termination conditions for actions."""
    for act in actions:
        if act.get("done", False):
            continue

        if "take_stage" in act:
            if take_done or take_failed:
                planner.has_action_done_next = act["done"] = True
                planner.planner_request_queue.put(
                    {"type": "done", "content": act["action"]}
                )
                print("Take Done")
        else:
            cur_tick = act.get("tick", 0)
            act["tick"] = cur_tick + 1

            if cur_tick > 240:
                planner.has_action_done_next = act["done"] = True
                planner.planner_request_queue.put(
                    {"type": "done", "content": act["action"], "exceed": True}
                )
                if act["take"]:
                    planner.failed = True

            if act["touch"] and act["take"]:
                _check_take_termination(planner, env, act)
            elif act["place"]:
                _check_place_termination(planner, env, objects, act, cur_tick)
            elif act["touch"]:
                _check_touch_termination(planner, env, objects, act, cur_tick)
            elif act["long_range"]:
                _check_long_range_termination(planner, env, objects, act, cur_tick)
            else:
                if cur_tick > 120:
                    planner.has_action_done_next = act["done"] = True
                    planner.planner_request_queue.put(
                        {"type": "done", "content": act["action"], "exceed": True}
                    )


def _check_take_termination(planner, env, act):
    """Check take action termination."""
    contact_points = act["contact_points"]
    contact_target = act["target"]

    if "left_hand" in contact_points and env.left_slot[planner.env_id] == contact_target:
        planner.has_action_done_next = act["done"] = True
        planner.planner_request_queue.put({"type": "done", "content": act["action"]})

    if "right_hand" in contact_points and env.right_slot[planner.env_id] == contact_target:
        planner.has_action_done_next = act["done"] = True
        planner.planner_request_queue.put({"type": "done", "content": act["action"]})


def _check_place_termination(planner, env, objects, act, cur_tick):
    """Check place action termination."""
    if planner.two_hand_grab_target is None:
        contact_target = act["target"]
        if (env.left_slot[planner.env_id] != contact_target
            and env.right_slot[planner.env_id] != contact_target
            and not act.get("fulfilled", False)):
            act["fulfilled"] = True
            act["fulfilled_tick"] = cur_tick
            print("Place!")

        if act.get("fulfilled", False) and cur_tick - act.get("fulfilled_tick", cur_tick) > 30:
            planner.has_action_done_next = act["done"] = True
            planner.planner_request_queue.put({"type": "done", "content": act["action"]})
    else:
        agent_position = env.get_joint_info(planner.env_id, "pelvis")[0]
        contact_target = planner.two_hand_grab_target
        target_instance = objects[contact_target]
        box_position = torch.tensor(target_instance["position"]).cuda().float()
        box_rotation = torch.tensor(target_instance["rotation"]).cuda().float()
        box_bbox = torch.tensor(target_instance["bbox"]).cuda().float()
        box_bbox = quaternion.qrot(box_rotation.unsqueeze(0).repeat(8, 1), box_bbox)
        box_bbox += box_position
        box_minz = box_bbox[:, 2].min()

        if box_minz < 0.1 or agent_position[2] < 0.6:
            planner.two_hand_grab_target = None
            planner.two_hand_grab_position = None
            planner.has_action_done_next = act["done"] = True
            planner.planner_request_queue.put({"type": "done", "content": act["action"]})


def _check_touch_termination(planner, env, objects, act, cur_tick):
    """Check touch action termination."""
    transformed_action = action_local_to_world(act, objects)
    contact_points = transformed_action["contact_points"]
    contact_points = [SURFACE_TO_JOINT[p] for p in contact_points]
    contact_targets = transformed_action["contact_targets"]

    if len(contact_points) == len(contact_targets):
        all_contact = True

        for joint, joint_target in zip(contact_points, contact_targets):
            joint_target = np.array(joint_target)
            if "wrist" in joint:
                joint = joint.replace("wrist", "hand")
            state, force = env.get_joint_info(planner.env_id, joint)

            target = transformed_action["target"]
            instance = objects[target]
            bbox = np.array(instance["bbox"]) + np.array(instance["position"])
            greater = (state > (bbox.min(0) - 0.1)).all()
            smaller = (state < (bbox.max(0) + 0.1)).all()

            is_touching = (np.abs(force) > 1e-3).any()
            is_inplace = (np.linalg.norm(state - joint_target) < 0.5)
            is_bbox = greater and smaller

            if cur_tick < 60:
                if not (is_touching and is_inplace and is_bbox):
                    all_contact = False
                    break
            else:
                if not (is_touching and is_bbox):
                    all_contact = False
                    break

        if all_contact and act.get("fulfilled", False) == False:
            act["fulfilled"] = True
            act["fulfilled_tick"] = cur_tick
            print("Touch!")

        if act.get("fulfilled", False) and cur_tick - act.get("fulfilled_tick", cur_tick) > 240:
            planner.has_action_done_next = act["done"] = True
            planner.planner_request_queue.put({"type": "done", "content": act["action"]})


def _check_long_range_termination(planner, env, objects, act, cur_tick):
    """Check long range action termination."""
    transformed_action = action_local_to_world(act, objects)
    all_fulfilled = True

    if transformed_action["position"] is not None:
        tar_position = np.array(transformed_action["position"], dtype=np.float32)[:2]
        position = env.get_joint_info(planner.env_id, "pelvis")[0][:2]
        if np.linalg.norm(tar_position - position) > 0.5:
            all_fulfilled = False

    if transformed_action["facing"] is not None:
        tar_facing = transformed_action["facing"]
        facing = env.current_facing_angle[planner.env_id]
        if (tar_facing - facing + np.pi) % (2 * np.pi) - np.pi > np.pi / 6:
            all_fulfilled = False

    if all_fulfilled and act.get("fulfilled", False) == False:
        act["fulfilled"] = True
        act["fulfilled_tick"] = cur_tick
        print("Long Range!")

    if act.get("fulfilled", False) and cur_tick - act.get("fulfilled_tick", cur_tick) > 240:
        planner.has_action_done_next = act["done"] = True
        planner.planner_request_queue.put({"type": "done", "content": act["action"]})

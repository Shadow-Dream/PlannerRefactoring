"""Human action representation and processing."""
import re
import copy
import numpy as np
from planner.core import quaternion


class HumanAction:
    """Represents a human action with targets and attributes."""

    def __init__(self, action, objects):
        """Initialize HumanAction from action string and objects dictionary."""
        targets = []
        marked_action = action
        sorted_objects = sorted(objects, key=lambda x: -len(x))
        pattern = r'\b(' + '|'.join(re.escape(name) for name in sorted_objects) + r')\b'
        targets = []

        def replacer(match):
            name = match.group(0)
            targets.append(name)
            return f"<{name}>"

        marked_action = re.sub(pattern, replacer, action)

        self.targets = targets
        self.marked_action = marked_action
        self.action = action
        self.target = None
        self.facing = None
        self.at = None
        self.by = None
        self.touch = False
        self.long_range = False
        self.take = False
        self.place = False
        self.position = None
        self.position_tag = None
        self.tag_direction = None
        self.contact_points = []
        self.contact_targets = []
        self.glb_position = None
        self.glb_facing = None
        self.glb_contact_targets = []

    def serialize(self):
        """Serialize action to dictionary."""
        return {
            "type": "action",
            "action": self.action,
            "target": self.target,
            "facing": self.facing,
            "at": self.at,
            "by": self.by,
            "touch": self.touch,
            "long_range": self.long_range,
            "take": self.take,
            "place": self.place,
            "position": self.position,
            "contact_points": self.contact_points,
            "contact_targets": self.contact_targets
        }


def merge_action(actions):
    """Merge multiple actions into a single unified action."""
    final_action = copy.deepcopy(actions[0])
    for action_index, action in enumerate(actions[1:]):
        if final_action["target"] is None:
            final_action["target"] = action["target"]

        if action["position"] is not None:
            final_action["position"] = action["position"]

        current_action = final_action["action"].replace(".", "")
        incoming_action = action["action"].replace(".", "")
        if not incoming_action.startswith("Walking to"):
            current_action += ", " + incoming_action
        final_action["action"] = current_action

        if action["facing"] is not None:
            final_action["facing"] = action["facing"]

        if final_action["at"] is None:
            final_action["at"] = action["at"]
        elif action["at"] is not None:
            final_action["at"] = list(set(final_action["at"] + action["at"]))

        if final_action["by"] is None:
            final_action["by"] = action["by"]
        elif action["by"] is not None:
            final_action["by"] = list(set(final_action["by"] + action["by"]))

        final_action["touch"] = final_action["touch"] or action["touch"]

        if len(final_action["contact_points"]) != len(final_action["contact_targets"]):
            continue
        if len(action["contact_points"]) != len(action["contact_targets"]):
            continue

        current_contacts = {
            k: v for k, v in zip(
                final_action["contact_points"],
                final_action["contact_targets"]
            )
        }
        incoming_contacts = {
            k: v for k, v in zip(
                action["contact_points"],
                action["contact_targets"]
            )
        }
        current_contacts.update(incoming_contacts)
        final_action["contact_points"] = list(current_contacts.keys())
        final_action["contact_targets"] = [
            current_contacts[k] for k in final_action["contact_points"]
        ]

    action_string = final_action["action"]
    action_string = action_string.replace(".", "").strip() + "."
    action_strings = action_string.split(",")
    if len(action_strings) == 1:
        action_string = action_string
    elif len(action_strings) == 2:
        action_string = " and".join(action_strings)
    else:
        action_strings[-1] = " and" + action_strings[-1]
        action_string = ",".join(action_strings)
    final_action["action"] = action_string
    return final_action


def action_local_to_world(action, objects):
    """Transform action coordinates from local to world space."""
    action = copy.deepcopy(action)
    if action["target"] is not None and action["position"] is not None:
        act_position = np.array(action["position"]).tolist()
        act_position += [0]
        act_position = np.array(act_position)
        if action["place"] and action["at"] is not None and len(action["at"]) > 0:
            target = action["at"][0]
        else:
            target = action["target"]
        tar_instance = objects[target]
        tar_position = np.array(tar_instance["position"])

        tar_rotation = np.array(tar_instance["rotation"])
        tar_rotation = tar_rotation[None]

        act_position = act_position[None]
        act_position = quaternion.qrot_np(tar_rotation, act_position)
        act_position = act_position + tar_position
        act_position = act_position[0][:2].tolist()
        action["position"] = act_position

        if action["contact_targets"] is not None and len(action["contact_targets"]) > 0:
            con_position = np.array(action["contact_targets"])
            tar_rota_con = np.broadcast_to(tar_rotation, (con_position.shape[0], 4))
            con_position = quaternion.qrot_np(tar_rota_con, con_position)
            con_position = con_position + tar_position
            con_position = con_position.tolist()
            action["contact_targets"] = con_position

        if action["facing"] is not None:
            fac_direction = [np.cos(action["facing"]), np.sin(action["facing"]), 0]
            fac_direction = np.array(fac_direction)[None]
            fac_direction = quaternion.qrot_np(tar_rotation, fac_direction)
            fac_direction = fac_direction[0, :2]
            fac_direction = float(np.arctan2(fac_direction[1], fac_direction[0]))
            action["facing"] = fac_direction
    return action

"""Formatting utility functions."""
import numpy as np
from planner.prompts import STATE_TEMPLATE


def format_coordinate_string(coordinate):
    """Format coordinate as string."""
    return f"({coordinate[0]:.1f}, {coordinate[1]:.1f})"


def format_coordinate(coordinate):
    """Format coordinate with rounding."""
    x, y = coordinate[:2]
    return (round(x * 10) / 10, round(y * 10) / 10)


def format_objects(objects, object_dict, index=False):
    """Format objects dictionary as string."""
    object_dict = {
        name: (format_coordinate_string(objects[name]["position"][:2]), contain)
        for name, contain in object_dict.items()
    }
    formatted = []
    i = 1
    for name, (position, contain) in object_dict.items():
        if name == "origin":
            continue
        if index:
            object_string = f"- {i}: {name}"
            i += 1
        else:
            object_string = f"- {name}: {position}"
        if contain:
            contain_string = ", ".join(contain)
            object_string += f", containing [{contain_string}]"
        formatted.append(object_string)
    object_string = "\n".join(formatted)
    return object_string


def format_objects_labels(objects, object_dict, parent_labels):
    """Format objects with labels."""
    object_dict = {
        name: (
            format_coordinate_string(objects[name]["position"][:2]),
            sorted(contain, key=lambda x: objects[x]["position"][2])
        )
        for name, contain in object_dict.items()
    }
    rev = {v: k for k, v in parent_labels.items()}
    formatted = []
    missing_objects = []
    for name, (position, contain) in object_dict.items():
        if name == "origin":
            continue
        if name not in rev:
            missing_objects.append(name)
            continue
        i = rev[name]
        object_string = f"- {i}: {name}, in {position}"

        if contain:
            contain_string = ", ".join(contain)
            object_string += f", containing [{contain_string}] from bottom to top"
        formatted.append(object_string)

    formatted = sorted(formatted, key=lambda x: int(x[2:x.index(":")]))
    for name in missing_objects:
        formatted.append(f"- {name}")
    object_string = "\n".join(formatted)
    return object_string


def format_state(state, actions):
    """Format agent state as string."""
    position_name = state["position_name"]
    position = format_coordinate_string(state["position"])
    return (STATE_TEMPLATE
            ).replace("position_placeholder", f"{position_name} {position}"
            ).replace("state_placeholder", ", ".join(actions)
            ).replace("left_slot_placeholder", str(state["left_slot"])
            ).replace("right_slot_placeholder", str(state["right_slot"]))


def get_volume(instance):
    """Calculate volume of an instance's bounding box."""
    bbox = np.array(instance["bbox"], dtype=np.float32)
    min_axis, max_axis = bbox.min(0), bbox.max(0)
    return np.prod(max_axis - min_axis)

"""Constants and templates used across the planner."""

# State display template
STATE_TEMPLATE = """Position: position_placeholder
Actions: [action_list_placeholder]
Left Hand Holding: left_slot_placeholder
Right Hand Holding: right_slot_placeholder"""

# Joint name mapping from surface names to skeleton joint names
SURFACE_TO_JOINT = {
    'back': 'pelvis',
    'pelvis': 'pelvis',
    'left_foot': 'left_foot',
    'right_foot': 'right_foot',
    'left_hand': 'left_wrist',
    'right_hand': 'right_wrist',
    'head': 'head',
}

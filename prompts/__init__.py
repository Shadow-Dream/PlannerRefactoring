"""
Prompts package for LLM Planner.

This package contains all prompt templates organized by functionality:
- instruction_refinement: Converting vague commands to clear instructions
- command_execution: Main planning loop prompts
- touch_detection: Determining interaction attributes
- target_specification: Identifying interaction targets
- position_specification: Determining where to stand
- contact_localization: Finding exact contact positions
- constants: Shared constants and templates
"""

# Import from submodules for convenience
from planner.prompts.instruction_refinement import (
    STEP1_SYSTEM, STEP1_USER_EXAMPLE, STEP1_ASSISTANT_EXAMPLE, STEP1_USER_TEMPLATE,
    # Legacy aliases
    STEP1_SYSTEM1, STEP1_USER1, STEP1_ASSISTANT1, STEP1_USER2
)

from planner.prompts.command_execution import (
    SYSTEM_COMMAND, USER_EXAMPLE, ASSISTANT_EXAMPLE,
    USER_FAILED_NOT_CLOSE_ENOUGH, USER_FAILED_MOVE, USER_FAILED_OBJECT,
    # Legacy aliases
    SYSTEM1, USER1, ASSISTANT1
)

from planner.prompts.touch_detection import (
    SYSTEM_SPECIFY_TOUCH, USER1_SPECIFY_TOUCH, ASSISTANT1_SPECIFY_TOUCH,
    USER2_SPECIFY_TOUCH
)

from planner.prompts.target_specification import (
    USER_SPECIFY_TARGET
)

from planner.prompts.position_specification import (
    SYSTEM1_SPECIFY_POSITION, SYSTEM1_SPECIFY_POSITION_LONGRANGE,
    USER1_SPECIFY_POSITION_TARGET, USER1_SPECIFY_POSITION_MARKER,
    USER1_SPECIFY_POSITION_ACTION,
    USER1_DIRECTION_LONGRANGE, USER1_SPECIFY_DISTANCE_LONGRANGE,
    USER1_SPECIFY_FACING_LONGRANGE
)

from planner.prompts.contact_localization import (
    USER1_SPECIFY_TARGET_POINT_IMAGE, USER1_SPECIFY_TARGET_POINT_JOINT,
    USER1_SPECIFY_TARGET_POINT_MULTIJOINTS,
    USER_POSITIONING_INITIAL, USER_POSITIONING_LOCATE,
    USER_POSITIONING_INITIAL_PLACE, USER_POSITIONING_LOCATE_PLACE
)

from planner.prompts.constants import (
    STATE_TEMPLATE, SURFACE_TO_JOINT
)

__all__ = [
    # Instruction refinement
    'STEP1_SYSTEM', 'STEP1_USER_EXAMPLE', 'STEP1_ASSISTANT_EXAMPLE', 'STEP1_USER_TEMPLATE',
    'STEP1_SYSTEM1', 'STEP1_USER1', 'STEP1_ASSISTANT1', 'STEP1_USER2',
    # Command execution
    'SYSTEM_COMMAND', 'USER_EXAMPLE', 'ASSISTANT_EXAMPLE',
    'USER_FAILED_NOT_CLOSE_ENOUGH', 'USER_FAILED_MOVE', 'USER_FAILED_OBJECT',
    'SYSTEM1', 'USER1', 'ASSISTANT1',
    # Touch detection
    'SYSTEM_SPECIFY_TOUCH', 'USER1_SPECIFY_TOUCH', 'ASSISTANT1_SPECIFY_TOUCH',
    'USER2_SPECIFY_TOUCH',
    # Target specification
    'USER_SPECIFY_TARGET',
    # Position specification
    'SYSTEM1_SPECIFY_POSITION', 'SYSTEM1_SPECIFY_POSITION_LONGRANGE',
    'USER1_SPECIFY_POSITION_TARGET', 'USER1_SPECIFY_POSITION_MARKER',
    'USER1_SPECIFY_POSITION_ACTION',
    'USER1_DIRECTION_LONGRANGE', 'USER1_SPECIFY_DISTANCE_LONGRANGE',
    'USER1_SPECIFY_FACING_LONGRANGE',
    # Contact localization
    'USER1_SPECIFY_TARGET_POINT_IMAGE', 'USER1_SPECIFY_TARGET_POINT_JOINT',
    'USER1_SPECIFY_TARGET_POINT_MULTIJOINTS',
    'USER_POSITIONING_INITIAL', 'USER_POSITIONING_LOCATE',
    'USER_POSITIONING_INITIAL_PLACE', 'USER_POSITIONING_LOCATE_PLACE',
    # Constants
    'STATE_TEMPLATE', 'SURFACE_TO_JOINT'
]

"""Prompts for instruction refinement - converting vague commands to clear instructions."""

STEP1_SYSTEM = """You are a state-of-the-art intelligent embodied robot.

Given a vague command, you can interpret it into a sequence of clear, executable instructions according to the scene.

Definition of Clear Instruction:
- Imperative sentence.
- Have exactly one verb.
- If the verb denotes an interactive action, it must explicitly state the object. If not stated in the origin command, you should choose one from the object list.
- The object stated in the sentence must exists in the provided object list.
- Make sure every verb in the original command is included in the instruction sequence.
- Don't miss details.
"""

STEP1_USER_EXAMPLE = """Vague Command: Jump in place. Sit, and sleep.

Objects:
- sofa1: (5.0 ,5.0)
- bed1: (0.0 ,5.0)

Clear Instructions:"""

STEP1_ASSISTANT_EXAMPLE = """Jump. Sit on the sofa1. Sleep on bed1."""

STEP1_USER_TEMPLATE = """Vague Command: prompt_placeholder

Objects:
object_placeholder

Clear Instructions:"""

# Legacy aliases for backward compatibility
STEP1_SYSTEM1 = STEP1_SYSTEM
STEP1_USER1 = STEP1_USER_EXAMPLE
STEP1_ASSISTANT1 = STEP1_ASSISTANT_EXAMPLE
STEP1_USER2 = STEP1_USER_TEMPLATE

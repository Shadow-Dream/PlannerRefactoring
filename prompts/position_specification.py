"""Prompts for position specification - determining where to stand."""

# Close-range position specification
SYSTEM1_SPECIFY_POSITION = """You're the state-of-the-art intelligent embodied robot. You can interact with the scene. When interacting with an object, you can determine what position you should stand in. The available positions and their information will be provided in the prompt, and each has its own index.

Please specify the position index where you should stand to interact with the target object. Your answer should be enclosed in >>> and <<<.
"""

USER1_SPECIFY_POSITION_TARGET = """Target Object: object_placeholder"""
USER1_SPECIFY_POSITION_MARKER = """Marker Position:
marker_placeholder"""
USER1_SPECIFY_POSITION_ACTION = """When performing "action_placeholder", which position should you stand at? """

# Long-range position specification
SYSTEM1_SPECIFY_POSITION_LONGRANGE = """You're the state-of-the-art intelligent embodied robot. You can interact with the scene. When you interact with an object from a distance, you can determine your own position relative to it.

Please analyze the direction where you should stand to interact with the target object by the marker index. Your final answer should be enclosed in >>> and <<<.
"""

USER1_DIRECTION_LONGRANGE = """You are in a scene with following objects:

{objects}

You are currently "{action_current}", and are going to perform "{action}".

Please analyze, when performing "{action}", which direction you should stand relative to {target}?

{arrows}

Please provide the label corresponding to the direction you should stand in, and enclose your answer in >>> and <<<.
"""

USER1_SPECIFY_DISTANCE_LONGRANGE = """You are in a scene with following objects:

{objects}

When you are performing {action}, which position should you stand in? Here is a set of candidate positions and their distances to {target}, each has its own index:

{distances}

Please select the most suitable position and provide corresponding index. Wrap your answer with >>> and <<<.
"""

USER1_SPECIFY_FACING_LONGRANGE = """You are in a scene with following objects:

{objects}

When you are "{action}", which direction should you face?

Here's a set of candidate directions with their indices, where direction with index {label} is facing directly to {target}.

- 1: 0°
- 2: 45°
- 3: 90°
- 4: 135°
- 5: 180°
- 6: 225°
- 7: 270°
- 8: 315°

Please select the most suitable direction and provide corresponding index. Wrap your answer with >>> and <<<."""

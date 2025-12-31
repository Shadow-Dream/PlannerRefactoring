"""Prompts for contact point localization - finding exact contact positions."""

USER1_SPECIFY_TARGET_POINT_IMAGE = """You're the state-of-the-art intelligent embodied robot, and capable of interacting with objects and determining which part of the object needs to be contacted during the interaction.

You are now interacting with a object_placeholder, which is shown in the image. The image can be divided into two parts: the left side is a view_placeholder of the object_placeholder, while the right side shows the semantic segmentation of this photo. Each part is marked by a unique color and has its own integer identifier.
"""

USER1_SPECIFY_TARGET_POINT_JOINT = """
When you are performing "action_placeholder", which part of the object_placeholder your joint_placeholder needs to contact? What is the integer identifier of that part in the image? Please provide the integer identifier of the contact part and enclose your final answer in >>> and <<<."""

USER1_SPECIFY_TARGET_POINT_MULTIJOINTS = """
When you are performing "action_placeholder", which parts of the object_placeholder your joint_placeholder need to contact? What is the integer identifiers of these parts in the image? Please provide the integer identifiers of the contact parts in the same order as 'joint_placeholder', and enclose each of your final answer in >>> and <<<."""

USER_POSITIONING_INITIAL = """The image shows {view}.

When you are {action}, which part of the {target} will your {joint} contact? Please provide a noun or noun phrase (which can include adjectives and other modifiers to describe its features) as the answer.

Please note that:
- the left and right sides of the image may differ from those of the object itself.
- wrap your answer in >>> and <<<.
"""

USER_POSITIONING_LOCATE = """Round: {round}

The image shows {view}.

Now, let's play a game! There is a black-bordered, white-filled circular marker in the image. Please determine whether this marker is located on the {part}, where your {joint} will contact when {action}. If it is, output "Yes" If it is not, output "No" and indicate the position of the {part} relative to the marker in the image, using [left, right, up, down, upper left, upper right, lower left, lower right]; for example, "No, the {{part}} is on the left of the marker".

{directions}

Please note that:
- the left and right sides of the image may differ from those of the object itself.
- if "No", the output should be the position of {part} relative to the marker, rather than the marker relative to the {part}.
- consider if the place is suitable for performing {action}.
- you are encouraged to directly output the answer without any explanation, and the answer should be wrapped in >>> and <<<.
"""

USER_POSITIONING_INITIAL_PLACE = """The image shows {view}.

When you are {action}, which part of the {target} will you place {place_target}? Please provide a noun or noun phrase (which can include adjectives and other modifiers to describe its features) as the answer.

Please note that:
- the left and right sides of the image may differ from those of the object itself.
- wrap your answer in >>> and <<<.
"""

USER_POSITIONING_LOCATE_PLACE = """Round: {round}

The image shows {view}.

Now, let's play a game! All three images have a black-bordered, white-filled circular marker at the same position. Please determine whether this marker is located on the {part}. The {part} is where you will place {place_target} when {action}. If it is, output "Yes" If it is not, output "No" and indicate the position of the {part} relative to the marker in the image, using [left, right, up, down, upper left, upper right, lower left, lower right]; for example, "No, the {{part}} is on the left of the marker".

{directions}

Please note that:
- the left and right sides of the image may differ from those of the object itself.
- if "No", the output should be the position of {part} relative to the marker, rather than the marker relative to the {part}.
- consider if the place is suitable for performing {action}.
- you are encouraged to directly output the answer without any explanation, and the answer should be wrapped in >>> and <<<.
"""

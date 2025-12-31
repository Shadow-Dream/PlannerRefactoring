"""Prompts for target specification - identifying interaction targets."""

USER_SPECIFY_TARGET = """The following sentence describes an interaction, and contains multiple objects.
All the objects are marked with "<" and ">".

Within the objects, please identify which is the real target of the interaction, and what are the roles of the remaining objects.

The role includes "by" and "at".
- "by" means the interaction is done by using the object.
- "at" means the interaction is done at the object.

Please answer in the following format:
target: <target>
by: <object1>, <object2>, ...
at: <object3>, <object4>, ...

Example:
Question 1:
Put the <pen1> on the <table1>.
Answer 1:
target: <pen1>
at: <table1>

Question 2:
Grab the <cloth1> from the <washing machine1>.
Answer 2:
target: <cloth1>
at: <washing machine1>

Question 3:
Turn on the <light1> with the <switch1>.
Answer 3:
target: <light1>
by: <switch1>

Sentence:
sentence_placeholder
"""

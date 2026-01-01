# Planning System Continuous Execution Improvement Plan

## Overview

Convert the blocking planning system to a continuous execution model where the VLM can:
1. Use `skip(seconds)` to defer decision-making
2. Stop actions proactively without waiting for automatic completion
3. Observe detailed action state (stage + duration) for informed decisions

## Current Architecture Analysis

### Current Flow (Blocking)
```
VLM: start("sitting on bed1")
  -> Backend sends action to main process
  -> Backend waits for "done" signal from main process (hardcoded 240 frames)
  -> Backend receives "done"
VLM: stop("sitting on bed1")
  -> Backend returns success message
VLM: start(next action) or end()
```

### Problem
- VLM cannot observe intermediate state during action execution
- Cannot react to failures or stuck situations
- All termination relies on hardcoded frame counts
- Actions like "dancing" with no natural end cannot be handled properly

## Proposed Architecture

### New Flow (Continuous)
```
VLM: start("sitting on bed1")
  -> Action starts executing immediately
[State: sitting on bed1 (moving, 0.5s)]
VLM: skip(1)  # Wait and observe
[State: sitting on bed1 (acting, 1.2s)]
VLM: skip(2)  # Continue executing
[State: sitting on bed1 (done, 3.5s)]
VLM: stop("sitting on bed1")  # Stop completed action
```

---

## Implementation Plan

### Phase 1: Enhanced State Information

#### 1.1 Modify State Template

**File:** `planner/prompts/constants.py`

**Current:**
```python
STATE_TEMPLATE = """Position: position_placeholder
State: [state_placeholder]
Left Hand Holding: left_slot_placeholder
Right Hand Holding: right_slot_placeholder"""
```

**New:**
```python
STATE_TEMPLATE = """Position: position_placeholder
Actions: [action_list_placeholder]
Left Hand Holding: left_slot_placeholder
Right Hand Holding: right_slot_placeholder"""
```

#### 1.2 Modify format_state Function

**File:** `planner/utils/format_utils.py`

**Current:** `format_state(state, actions)` - actions is just a list of action names

**New:** `format_state(state, action_infos)` where action_infos contains:
- action name
- stage: "moving" | "acting" | "done"
- duration in seconds

**Example output:**
```
Actions: [sitting on bed1 (acting, 2.3s), looking at monitor1 (moving, 0.5s)]
```

#### 1.3 Add Action Tracking to Planning Backend

**File:** `planner/planning/planning_backend.py`

Add a new data structure to track action states:
```python
# In action_dict, store HumanAction plus execution state
action_info = {
    "action": HumanAction,
    "start_time": float,  # time.time() when action started
    "stage": str,         # "moving" | "acting" | "done"
    "fulfilled": bool,
    "fulfilled_time": float
}
```

### Phase 2: Stage Mapping System

#### 2.1 Stage Mapping Definition

Main process stages (update_handler.py):
- Stage 0: Navigation (path planning + following) -> **"moving"**
- Stage 1: Turning (facing target direction) -> **"moving"**
- Stage 2: Action execution -> **"acting"** or **"done"**

Within Stage 2:
- `fulfilled == False`: **"acting"**
- `fulfilled == True`: **"done"**

#### 2.2 Stage Update Mechanism

**File:** `planner/update_handler.py`

Add function to report stage changes:
```python
def get_action_stage(action):
    """Get semantic stage for an action."""
    stage = action.get("stage", 0)
    if stage in (0, 1):
        return "moving"
    elif stage == 2:
        if action.get("fulfilled", False):
            return "done"
        else:
            return "acting"
    return "unknown"
```

Report stage to planning backend via queue when stage changes.

### Phase 3: Skip Command Implementation

#### 3.1 Add Skip Command to Prompt

**File:** `planner/prompts/command_execution.py`

Add to available commands:
```
4. skip(seconds): Skip decision-making for the specified duration. Default is 1 second if not specified. Use this when actions are progressing normally and no intervention is needed.
```

#### 3.2 Add Skip Handler

**File:** `planner/planning/planning_backend.py`

```python
def _handle_skip(self, command, requests, action_dict, state):
    """Handle skip command - wait for specified duration."""
    # Parse duration
    if "(" in command and ")" in command:
        arg = command[command.index("(") + 1:command.index(")")]
        try:
            duration = float(arg.strip())
        except:
            duration = 1.0
    else:
        duration = 1.0

    # Wait for duration while collecting state updates
    time.sleep(duration)

    # Collect any stage update messages
    while not self.request_queue.empty():
        request = self.request_queue.get()
        requests.append(request)

    return "Waited for {:.1f}s. Current state:".format(duration)
```

### Phase 4: Remove Hardcoded Delays

#### 4.1 Identify Hardcoded Delays to Remove

**File:** `planner/update_handler.py`

**REMOVE these delays:**
1. Line 983: `if cur_tick > 240:` - Force done after 240 frames for all actions
2. Line 1000: `if cur_tick > 120:` - Force done for non-touch/non-long-range actions
3. Line 1093: `cur_tick - act.get("fulfilled_tick", cur_tick) > 240` - Wait 240 frames after touch fulfilled
4. Line 1120: `cur_tick - act.get("fulfilled_tick", cur_tick) > 240` - Wait 240 frames after long_range fulfilled

**KEEP these delays (prevent false positives):**
1. Line 1032: `> 30` after place fulfilled - Brief wait to confirm stable placement
2. Lines 1079-1086: Touch detection with `cur_tick < 60` - Initial settling period

#### 4.2 Modified Termination Logic

Remove forced termination timers. Instead:
- Actions report "done" immediately when fulfilled
- VLM decides when to stop actions (can let done actions continue for a while if desired)

### Phase 5: Non-Terminable Actions

#### 5.1 Action Type Classification

Some actions naturally complete (take, place, touch), others don't (dance, wave):

```python
def can_action_complete(action):
    """Check if action has a natural completion condition."""
    if action["take"] or action["place"]:
        return True  # Has completion when object transferred
    if action["touch"]:
        return True  # Has completion when contact made
    if action["long_range"]:
        return True  # Has completion when position/facing reached
    return False  # Actions like "dancing" never auto-complete
```

#### 5.2 Stage for Non-Terminable Actions

For actions without natural completion:
- Stage is "acting" indefinitely until VLM calls stop()
- VLM uses duration to decide when to stop

### Phase 6: Prompt Updates

#### 6.1 Updated System Prompt

**File:** `planner/prompts/command_execution.py`

```python
SYSTEM_COMMAND = """You are a state-of-the-art intelligent embodied robot.

You are currently standing in a scene with the following objects, and the layout of the scene is provided as followed:
object_placeholder

You should perform a sequence of actions to fulfill the following instructions, step by step:
Origin Instruction: origin_prompt_placeholder
Refined Instruction: prompt_placeholder

In each step, the sensor will provide you with your position, currently executing actions with their stages and durations, and the objects you are holding. You should decide your next action accordingly.

The available commands includes:
1. start(action): Start doing the specified action. If the action requires interacting with a distant object, the system will move you there automatically.
2. stop(action): Stop the specified action. Use this to:
   - Remove completed actions (stage: done) from your action list
   - Cancel actions that are stuck or taking too long
   - Stop continuous actions like dancing when you want to proceed
3. skip(seconds): Wait for the specified duration (default: 1 second) before next decision. Use this when actions are progressing normally.
4. end(): Call when the instruction is fully fulfilled.

Action stages:
- moving: The agent is navigating to the target position
- acting: The action is being executed
- done: The action objective has been achieved (ready to stop)

Note: Actions in "done" stage should be stopped with stop() command. Actions without natural completion (like "dancing") stay in "acting" stage until explicitly stopped.

Example of actions:
1. Jumping excitedly
2. Leaning against car1 in front of it
3. Taking hammer1 from toolbox1
4. Standing on the right of table1 three meters away
5. Sitting on bed1

Constraints:
- Each step you can only give one command.
- Each action should be indivisible (one verb only).
- For interactive actions, explicitly specify the target object.
- You can perform multiple actions simultaneously.
- Output the command directly without explanation.
"""
```

#### 6.2 Updated State Format Example

**File:** `planner/prompts/command_execution.py`

```python
USER_EXAMPLE = """Position: origin (0, 0)
Actions: [Standing on origin (done, 0.0s)]
Left Hand Holding: None
Right Hand Holding: None
"""

ASSISTANT_EXAMPLE = """stop("Standing on origin")"""
```

---

## Testing Plan

### Test 1: Skip Command Basic
- Start an action
- Use skip(1) to wait
- Verify state shows duration increase
- Stop the action

### Test 2: Stage Transitions
- Start "sitting on bed1"
- Verify: moving -> acting -> done transitions
- Verify duration updates correctly

### Test 3: Non-Terminable Actions
- Start "dancing"
- Verify: stays in "acting" indefinitely
- Use skip() multiple times
- Stop explicitly with stop()

### Test 4: Stuck Action Recovery
- Simulate stuck scenario
- Verify VLM can detect via duration
- VLM stops and retries

---

## File Change Summary

| File | Changes |
|------|---------|
| `planner/prompts/constants.py` | Update STATE_TEMPLATE |
| `planner/prompts/command_execution.py` | Update SYSTEM_COMMAND, USER_EXAMPLE, ASSISTANT_EXAMPLE |
| `planner/utils/format_utils.py` | Update format_state() to include stage/duration |
| `planner/planning/planning_backend.py` | Add _handle_skip(), modify action tracking, modify main loop |
| `planner/update_handler.py` | Remove hardcoded delays, add stage reporting |

---

## Implementation Order

1. [x] Phase 1.1: Update STATE_TEMPLATE
2. [x] Phase 1.2: Update format_state function
3. [x] Phase 2: Implement stage mapping and reporting
4. [x] Phase 3: Implement skip command
5. [x] Phase 4: Remove hardcoded delays
6. [x] Phase 6: Update prompts
7. [x] Testing: Run all test scenarios

# Planner Mock Test

## Directory Structure

```
planner/
├── core/                    # Core utilities
│   ├── action.py           # HumanAction class
│   ├── quaternion.py       # Quaternion operations
│   └── vlm_client.py       # VLM client interface
├── navigation/             # Path planning backend
│   └── navigation_backend.py
├── planning/               # LLM planning backend
│   ├── __init__.py
│   ├── planning_backend.py # Main planning orchestrator
│   ├── action_parser.py    # Action attribute parsing
│   ├── position_handler.py # Position specification
│   ├── contact_locator.py  # Contact point localization
│   └── stages/             # Pipeline stages (NEW)
│       ├── __init__.py
│       ├── base.py         # Base classes & data structures
│       ├── basic_analyze.py    # Stage 1: Touch/Take/Place detection
│       ├── pose_reasoning.py   # Stage 2: Position/Facing reasoning
│       └── point_reasoning.py  # Stage 3: Contact point localization
├── prompts/                # Prompt templates
├── utils/                  # Utility functions
├── task/                   # Task configuration
├── scene/                  # Scene data (extract from scene.zip)
├── image/                  # Scene images (extract from image.zip)
├── capture/                # Capture data and VLM responses (extract from capture.zip)
├── logs/                   # Output logs (auto-generated)
├── llm_planner.py
├── mock_env.py
├── mock_vlm.py
├── test_planner.py
└── update_handler.py
```

## Pipeline Architecture

The action planning system uses a three-stage pipeline:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  BasicAnalyzeStage  │ -> │  PoseReasoningStage │ -> │ PointReasoningStage │
│  (Stage 1)          │    │  (Stage 2)          │    │  (Stage 3)          │
├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ • Parse targets     │    │ • Position spec     │    │ • Contact points    │
│ • Touch detection   │    │ • Facing direction  │    │ • Iterative VLM     │
│ • Take/Place voting │    │ • Long/Close range  │    │ • Coordinate trans  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

Each stage is:
- **Stateless**: All state is passed via `ActionContext`
- **Independent**: Can run in separate processes
- **Resumable**: Can continue from capture results

This architecture enables future pipeline parallelism where multiple
actions can be processed concurrently at different stages.

## Setup

1. Extract data archives to `planner/` directory:
   ```
   unzip scene.zip -d planner/
   unzip image.zip -d planner/
   unzip capture.zip -d planner/
   ```

2. Run test:
   ```
   python planner/test_planner.py
   ```

## Output

- `logs/debug.png` - Navigation visualization
- `logs/navigation_0.png` - Path planning result
- `logs/llm_0.txt` - LLM conversation log

## Key Classes

### ActionContext
Data structure for action processing across pipeline stages.

```python
@dataclass
class ActionContext:
    action_id: str           # Unique action identifier
    action_string: str       # Action description

    # Stage 1 outputs
    target: Optional[str]
    touch: Optional[bool]
    take: Optional[bool]
    place: Optional[bool]
    contact_points: Optional[List[str]]

    # Stage 2 outputs
    position: Optional[List[float]]
    facing: Optional[float]

    # Stage 3 outputs
    contact_targets: Optional[List[List[float]]]
```

### StageInput / StageOutput
Input and output data structures for pipeline stages.

### PipelineStage
Abstract base class for all pipeline stages.

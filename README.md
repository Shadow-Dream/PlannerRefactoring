# Planner Mock Test

## Directory Structure

```
planner/
├── core/           # Core utilities (quaternion, action, vlm_client)
├── navigation/     # Path planning backend
├── planning/       # LLM planning backend
├── prompts/        # Prompt templates
├── utils/          # Utility functions
├── task/           # Task configuration
├── scene/          # Scene data (extract from scene.zip)
├── image/          # Scene images (extract from image.zip)
├── capture/        # Capture data and VLM responses (extract from capture.zip)
├── logs/           # Output logs (auto-generated)
├── llm_planner.py
├── mock_env.py
├── mock_vlm.py
├── test_planner.py
└── update_handler.py
```

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

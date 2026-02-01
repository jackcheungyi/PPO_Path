# PPO Path Planner

A Proximal Policy Optimization (PPO) based reinforcement learning project for training a robot to navigate through environments by following walls on its left-hand side and avoiding obstacles.

## Overview

This project uses stable-baselines3 with MaskablePPO to train an agent that:
- Follows walls on the **left-hand side** at a maintained distance (~20 pixels)
- Takes **left turns** around obstacles to avoid them
- Navigates from a start position to a goal position
- Learns efficient paths while maintaining wall-following behavior

## Project Structure

```
ppo_planer/
├── plan_game.py          # Core game environment logic
├── Env_wrapper_mlp.py    # Gym environment with MLP observations
├── Env_wrapper_cnn.py    # Gym environment with CNN observations
├── train_mlp.py          # Training script for MLP policy
├── train_cnn.py          # Training script for CNN policy
├── test_mlp.py           # Testing script for MLP policy
├── test.py               # General testing utilities
├── requirements.txt      # Python dependencies
├── pyproject.toml        # UV project configuration
└── README.md             # This file
```

## Environment

The environment is built on a floor plan map (HKSB_6F.pgm) where:
- **White areas**: Available navigation space
- **Black edges**: Walls/obstacles
- **Blue dot**: Start position (random)
- **Red dot**: Goal position (random)

### Action Space

8 discrete actions representing movement directions:
| Action | Movement |
|--------|----------|
| 0 | Up |
| 1 | Right |
| 2 | Left |
| 3 | Down |
| 4 | Up-Right |
| 5 | Up-Left |
| 6 | Down-Right |
| 7 | Down-Left |

### Observation Space

Two environment variants:

1. **MLP Environment** (`Env_wrapper_mlp.py`):
   - 2D grid representation with normalized float values
   - Available positions: 0.0
   - Path positions: 0.2 to 0.8 (gradient)
   - Current position: 1.0
   - Walls: -0.8
   - Goal: -1.0

2. **CNN Environment** (`Env_wrapper_cnn.py`):
   - RGB image (H × W × 3)
   - Available positions: White (255)
   - Path positions: Gradient from light to dark gray
   - Current position: Red (255, 0, 0)
   - Walls: Black (0, 0, 0)
   - Goal: Blue (0, 0, 255)

### Reward Function

The reward function is designed to encourage left-hand wall following:

| Reward Component | Value | Description |
|-----------------|-------|-------------|
| **Left Wall Following** | +2.0/step | Primary reward for following left wall |
| **Wall Distance (Optimal 15-25px)** | +0.5/step | Maintain ~20px from wall |
| **Wall Distance (<10px)** | -0.3/step | Too close to wall |
| **Wall Distance (>30px)** | -0.5/step | Too far from wall |
| **Left Turn Bonus** | +1.0/step | Taking left actions while following wall |
| **Goal Progress** | +0.3/step | Getting closer to goal |
| **Goal Regress** | -0.3/step | Moving away from goal |
| **Goal Reached** | +15-25 | Bonus based on path efficiency |
| **Crash Penalty** | -10.0 | Hitting wall or invalid move |

## Installation

### Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd ppo_planer
uv venv .venv --python 3.10
source .venv/bin/activate  # or source .venv/bin/activate.fish for fish shell
uv pip install -r requirements.txt
```

### Using Conda

```bash
conda env create -f environment.yaml
conda activate ppo_env
```

## Usage

### Training MLP Policy

```bash
source .venv/bin/activate
python train_mlp.py
```

### Training CNN Policy

```bash
source .venv/bin/activate
python train_cnn.py
```

### Testing Trained Models

```bash
source .venv/bin/activate
python test_mlp.py
# or
python test.py
```

## Key Implementation Details

### Left Wall Following Detection

The `plan_game.py` module uses cross-product calculations to determine if the robot is following a wall on its left side:

```python
cross_product = np.cross(path_vector, wall_vector)
if cross_product > 0:  # Wall is on LEFT side
    return True, closest_wall
```

### Action Masking

The environment uses action masking to prevent invalid moves (e.g., sharp turns >90° or moving into walls).

## Dependencies

- Python 3.10+
- stable-baselines3==2.3.1
- sb3-contrib==2.3.0 (for MaskablePPO)
- gymnasium==0.29.1
- torch==2.3.0+cu118
- torchvision==0.18.0+cu118
- numpy, opencv-python, pygame, matplotlib

## License

MIT License

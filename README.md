# MetaWorld Trajectory Viewer

A tool for visualizing MetaWorld trajectory recordings from the QueST dataset. This viewer extracts RGB frames from recorded demonstrations and creates video files, supporting various sampling frequencies and dataset splits.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Repository Setup](#repository-setup)
- [Required Code Changes](#required-code-changes)
- [Dataset Splits](#dataset-splits)
- [Environment Setup](#environment-setup)
- [Troubleshooting](#troubleshooting)
- [Usage](#usage)
- [Example Output](#example-output)

## Prerequisites

### Directory Structure
Your workspace should have the following structure:
```
QueST/
├── quest/
│   └── utils/
│       └── metaworld_utils.py
├── data/
│   └── metaworld/
│       ├── ML45_PRISE/
│       │   ├── train/
│       │   │   ├── basketball-v3.hdf5
│       │   │   ├── peg-insert-side-v3.hdf5
│       │   │   └── ... (other task files)
│       │   └── test/
│       └── ML45/
│           ├── train/
│           └── test/
├── evaluate_trajectories.py
└── requirements.txt
```

## Repository Setup

### 1. Clone the Repositories

```bash
# Clone the QueST repository
git clone https://github.com/hsyassin/QueST.git
cd QueST

# Clone MetaWorld repository inside QueST
git clone https://github.com/Farama-Foundation/Metaworld.git
```

### 2. Create Conda Environment

```bash
# Create conda environment with Python 3.10
conda create -n quest python=3.10 -y
conda activate quest
```

## Required Code Changes

### MetaWorld API V2 to V3 Migration

The original QueST repository uses MetaWorld V2 API, but the current MetaWorld is V3. The following changes are required:

#### 1. File: `quest/utils/metaworld_utils.py`

**Change 1: Update imports (line ~7)**
```python
# OLD (V2):
from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerBasketballEnvV2,
    # ... other V2 imports
)

# NEW (V3):
from metaworld.envs import (
    SawyerBasketballEnvV3,
    # ... other V3 imports with V3 suffix
)
```

**Change 2: Update environment mapping dictionary (line ~50-100)**
```python
# OLD:
METAWORLD_ENVS = {
    'basketball-v3': SawyerBasketballEnvV2,
    # ...
}

# NEW:
METAWORLD_ENVS = {
    'basketball-v3': SawyerBasketballEnvV3,
    # ... all environments updated to V3 classes
}
```

## Dataset Splits

### ML45 vs ML45_PRISE

| Split | Training Tasks | Test Tasks | Description |
|-------|---------------|------------|-------------|
| **ML45** | 45 tasks | 5 tasks | Standard MetaWorld split used in most papers |
| **ML45_PRISE** | 45 tasks | 5 tasks | Custom split used in QueST paper with different task distribution |
| **MT50** | 50 tasks | - | All tasks used for training |

**Key Differences:**
- ML45 uses the standard MetaWorld task split
- ML45_PRISE redistributes tasks to ensure more diverse test scenarios
- Both have 45 training and 5 test tasks, but different task assignments

## Environment Setup 

### Install Dependencies 

```bash
# Activate conda environment
conda activate quest

# Install PyTorch with CUDA 12 support
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install MetaWorld
cd Metaworld
pip install -e .
cd ..

# Install other requirements
pip install -r requirements.txt

# Install additional required packages
pip install h5py tqdm moviepy imageio-ffmpeg

# Install mujoco
pip install mujoco
```

## Troubleshooting

### Headless Rendering Issue

**Error encountered:**
```
RuntimeError: Failed to initialize OpenGL
```

**Solution:**
Set environment variables for headless rendering on remote servers:

```bash
# Add to your script or shell
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

These are automatically set in the evaluation script, but you may need to set them manually if running other scripts.

### MetaWorld Version Mismatch

**Error encountered:**
```
ModuleNotFoundError: No module named 'metaworld.envs.mujoco.sawyer_xyz.v2'
```

**Solution:**
Update all V2 imports to V3 as described in the [Required Code Changes](#required-code-changes) section.

## Usage

### Script Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--task_name` | str | Yes | - | Task name (e.g., 'basketball-v3', 'peg-insert-side-v3') |
| `--traj_no` | int | Yes | - | Trajectory number to visualize (0-indexed) |
| `--output_dir` | str | No | './trajectory_videos' | Output directory for videos |
| `--sampling_freq` | int | No | 1 | Extract every Nth frame (1 = all frames) |
| `--data_prefix` | str | No | './data' | Data directory path |
| `--split_name` | str | No | 'ML45_PRISE' | Dataset split: 'ML45', 'ML45_PRISE', or 'MT50' |
| `--mode` | str | No | 'train' | Dataset mode: 'train' or 'test' |

### Example Commands

```bash
# Basic usage - visualize full trajectory
python evaluate_trajectories.py --task_name basketball-v3 --traj_no 25 --output_dir ./videos

# With sampling - extract every 10th frame
python evaluate_trajectories.py --task_name peg-insert-side-v3 --traj_no 75 --output_dir ./_videos --sampling_freq 10

# Using different dataset split
python evaluate_trajectories.py --task_name assembly-v3 --traj_no 0 --split_name ML45 --output_dir ./videos

# Test set trajectory
python evaluate_trajectories.py --task_name drawer-open-v3 --traj_no 5 --mode test --output_dir ./videos
```

## Example Output

### Example Command:
```bash
python evaluate_trajectories.py --task_name peg-insert-side-v3 --traj_no 75 --output_dir ./_videos
```

### Example Video Output:



*Peg Insertion Task - Trajectory 75*



https://github.com/user-attachments/assets/43e80ece-37c2-49a8-8bcc-3910a127ca04



### Console Output:
```
============================================================
MetaWorld Trajectory Evaluation Script - Stage 1
============================================================
[INFO] Loading trajectory 75 from dataset...
[INFO] Trajectory loaded successfully!
  Length: 306 steps
  Final success: True
  Total reward: 683.46
...
============================================================
ALL STAGES COMPLETE: Trajectory Evaluation Success!
============================================================
Video saved to: ./_videos/peg-insert-side-v3_traj_75.mp4
Frames: 307
Resolution: (128, 128)
Sampling: Every 1 frame(s)
```


## Dataset Generation

To generate your a custom version of the MetaWorld dataset:

```bash
# Generate sample dataset (5 trajectories per task)
python scripts/generate_metaworld_dataset.py \
    --tasks ML45_PRISE \
    --data_prefix ./data \
    --mode train \
    --n_episodes 5 \
    --save_video

# Generate full dataset (100 trajectories per task)
python scripts/generate_metaworld_dataset.py \
    --tasks ML45_PRISE \
    --data_prefix ./data \
    --mode train \
    --n_episodes 100
```

Ensure:
1. All MetaWorld imports use V3 API
2. Environment variables for headless rendering are properly set
3. Videos include appropriate metadata in filenames

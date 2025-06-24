#!/usr/bin/env python3
"""
This script can:
1. Take task name and trajectory number as arguments
2. Initialize MetaWorld environments properly in headless setup  
3. Extract trajectory data from HDF5 files
4. Set up rendering with sampling frequency support

Usage:
    python evaluate_trajectories.py --task_name assembly-v3 --traj_no 0 --output_dir ./videos
"""

import argparse
import os
import h5py
import numpy as np
from tqdm.auto import tqdm
import sys

# Add quest modules to path
sys.path.append('.')
import quest.utils.metaworld_utils as mu

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MetaWorld Trajectory Evaluation Script')
    
    parser.add_argument('--task_name', type=str, required=True,
                       help='Task name (e.g., assembly-v3, basketball-v3)')
    parser.add_argument('--traj_no', type=int, required=True,
                       help='Trajectory number to visualize (0-indexed)')
    parser.add_argument('--output_dir', type=str, default='./trajectory_videos',
                       help='Output directory for videos')
    parser.add_argument('--sampling_freq', type=int, default=1,
                       help='Sampling frequency - extract every Nth frame (1 = all frames)')
    parser.add_argument('--data_prefix', type=str, default='./data',
                       help='Data prefix path')
    parser.add_argument('--split_name', type=str, default='ML45_PRISE',
                       choices=['ML45', 'ML45_PRISE', 'MT50'],
                       help='Dataset split name')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'],
                       help='Dataset mode')
    
    return parser.parse_args()

def setup_headless_rendering():
    """Configure environment for headless rendering"""
    print("[INFO] Setting up headless rendering...")
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    print("[INFO] Headless rendering configured (EGL backend)")

def validate_inputs(args):
    """Validate input arguments"""
    print(f"[INFO] Validating inputs...")
    
    # Check if task exists in the split
    env_names = mu.get_env_names(args.split_name, args.mode)
    if args.task_name not in env_names:
        raise ValueError(f"Task '{args.task_name}' not found in {args.split_name}/{args.mode}. "
                        f"Available tasks: {env_names[:5]}...")
    
    # Check if data file exists
    dataset_path = os.path.join(
        args.data_prefix,
        'metaworld', 
        args.split_name,
        args.mode,
        f"{args.task_name}.hdf5"
    )
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"[INFO] Validation successful!")
    print(f"  Task: {args.task_name}")
    print(f"  Split: {args.split_name}/{args.mode}")
    print(f"  Dataset: {dataset_path}")
    
    return dataset_path

def load_trajectory_data(dataset_path, traj_no):
    """Load trajectory data from HDF5 file"""
    print(f"[INFO] Loading trajectory {traj_no} from dataset...")
    
    with h5py.File(dataset_path, 'r') as f:
        # Check if trajectory exists
        demo_key = f'demo_{traj_no}'
        if demo_key not in f['data']:
            available_demos = [k for k in f['data'].keys() if k.startswith('demo_')]
            raise ValueError(f"Trajectory {traj_no} not found. "
                           f"Available trajectories: {len(available_demos)} (0-{len(available_demos)-1})")
        
        demo = f['data'][demo_key]
        
        # Load trajectory data
        traj_data = {
            'actions': demo['actions'][:],
            'rewards': demo['reward'][:],
            'success': demo['success'][:],
            'terminated': demo['terminated'][:],
            'truncated': demo['truncated'][:],
        }
        
        # Load observations
        obs_data = {}
        obs_group = demo['obs']
        for obs_key in obs_group.keys():
            obs_data[obs_key] = obs_group[obs_key][:]
        
        traj_data['observations'] = obs_data
        
        # Get trajectory info
        traj_len = len(traj_data['actions'])
        final_success = bool(traj_data['success'][-1])
        total_reward = np.sum(traj_data['rewards'])
        
    print(f"[INFO] Trajectory loaded successfully!")
    print(f"  Length: {traj_len} steps")
    print(f"  Final success: {final_success}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Observation keys: {list(obs_data.keys())}")
    for obs_key, obs_val in obs_data.items():
        print(f"    {obs_key}: {obs_val.shape}")
    
    return traj_data

def initialize_environment(task_name):
    """Initialize MetaWorld environment for the given task"""
    print(f"[INFO] Initializing environment for task: {task_name}")
    
    try:
        # Create environment wrapper
        shape_meta = {
            'observation': {
                'rgb': {'corner_rgb': [3, 128, 128]},
                'lowdim': {'robot_states': 8}
            }
        }
        
        env = mu.MetaWorldWrapper(
            env_name=task_name,
            shape_meta=shape_meta,
            img_height=128,
            img_width=128,
            cameras=['corner2']
        )
        
        print(f"[INFO] Environment initialized successfully!")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space keys: {list(env.observation_space.spaces.keys())}")
        
        return env
    
    except Exception as e:
        print(f"[ERROR] Failed to initialize environment: {e}")
        raise

def apply_sampling_frequency(traj_data, sampling_freq):
    """Apply sampling frequency to trajectory data"""
    if sampling_freq <= 1:
        print(f"[INFO] Using all frames (sampling_freq={sampling_freq})")
        return traj_data
    
    print(f"[INFO] Applying sampling frequency: every {sampling_freq} frames")
    
    # Sample indices
    traj_len = len(traj_data['actions'])
    obs_len = len(traj_data['observations']['corner_rgb'])
    
    sampled_indices = list(range(0, traj_len, sampling_freq))
    
    # FIX: Ensure we get final observation by including the last index if not already included
    obs_sampled_indices = list(range(0, obs_len, sampling_freq))
    if obs_sampled_indices[-1] != obs_len - 1:
        obs_sampled_indices.append(obs_len - 1)
    
    # Create sampled trajectory
    sampled_traj = {}
    
    # Sample action-aligned data
    for key in ['actions', 'rewards', 'success', 'terminated', 'truncated']:
        sampled_traj[key] = traj_data[key][sampled_indices]
    
    # Sample observations
    sampled_obs = {}
    for obs_key, obs_val in traj_data['observations'].items():
        sampled_obs[obs_key] = obs_val[obs_sampled_indices]
    sampled_traj['observations'] = sampled_obs
    
    print(f"[INFO] Sampling applied: {traj_len} actions -> {len(sampled_indices)} actions")
    print(f"[INFO] Sampling applied: {obs_len} observations -> {len(obs_sampled_indices)} observations")
    return sampled_traj

def extract_frames_from_trajectory(traj_data, env, task_name, traj_no):
    """Extract RGB frames directly from recorded observations in trajectory data"""
    print(f"[INFO] Extracting frames from recorded observations...")
    
    # Use recorded RGB observations directly (no environment replay needed)
    recorded_frames = traj_data['observations']['corner_rgb']
    
    print(f"[INFO] Using {len(recorded_frames)} recorded RGB frames")
    print(f"[INFO] Frame shape: {recorded_frames[0].shape}")
    print(f"[INFO] Frame dtype: {recorded_frames.dtype}")
    print(f"[INFO] Frame value range: [{recorded_frames.min():.2f}, {recorded_frames.max():.2f}]")
    
    # Convert to numpy array for consistency
    frames = np.array(recorded_frames)
    
    # Ensure frames are in correct format [0, 255] uint8
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            # Frames are in [0, 1] range, scale to [0, 255]
            frames = (frames * 255).astype(np.uint8)
            print(f"[INFO] Converted frames from [0,1] to [0,255] range")
        else:
            # Frames are already in [0, 255] range but wrong dtype
            frames = frames.astype(np.uint8)
            print(f"[INFO] Converted frames to uint8 dtype")
    
    print(f"[INFO] Final frames shape: {frames.shape}")
    print(f"[INFO] Final frames dtype: {frames.dtype}")
    print(f"[INFO] Final frames value range: [{frames.min()}, {frames.max()}]")
    
    return frames

def create_and_save_video(frames, task_name, traj_no, output_dir, sampling_freq):
    """Create and save MP4 video from extracted frames"""
    print(f"[INFO] Creating video from {len(frames)} frames...")
    
    # Import moviepy for video creation
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        raise ImportError("moviepy is required for video creation. Install with: pip install moviepy")
    
    # Create video filename
    if sampling_freq > 1:
        filename = f"{task_name}_traj_{traj_no}_sampled_{sampling_freq}x.mp4"
    else:
        filename = f"{task_name}_traj_{traj_no}.mp4"
    
    video_path = os.path.join(output_dir, filename)
    
    # Ensure frames are in the correct format (0-255 uint8)
    if frames.dtype != np.uint8:
        print(f"[INFO] Converting frames from {frames.dtype} to uint8")
        frames = frames.astype(np.uint8)
    
    # Clip values to valid range
    frames = np.clip(frames, 0, 255)
    
    # Calculate appropriate FPS for better video duration
    # Aim for 3-10 second videos depending on content
    target_duration = max(3.0, min(10.0, len(frames) * 0.1))  # 3-10 seconds
    fps = max(8, min(30, len(frames) / target_duration))
    fps = round(fps)
    
    print(f"[INFO] Calculated FPS: {fps} (target duration: {target_duration:.1f}s)")
    
    print(f"[INFO] Creating video: {filename}")
    print(f"  Frames: {len(frames)}")
    print(f"  Resolution: {frames.shape[1]}x{frames.shape[2]}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(frames)/fps:.1f}s")
    
    # Create video clip
    try:
        # Convert frames to list for moviepy
        frame_list = [frame for frame in frames]
        
        # Create video clip
        clip = ImageSequenceClip(frame_list, fps=fps)
        
        # Write video file
        print(f"[INFO] Saving video to: {video_path}")
        clip.write_videofile(
            video_path, 
            fps=fps, 
            verbose=False, 
            logger=None,
            codec='libx264',
            audio=False
        )
        
        # Get file size
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"[INFO] Video saved successfully!")
        print(f"  Path: {video_path}")
        print(f"  Size: {file_size:.1f} MB")
        
        return video_path
        
    except Exception as e:
        print(f"[ERROR] Failed to create video: {e}")
        raise

def main():
    """Main evaluation function"""
    print("="*60)
    print("MetaWorld Trajectory Evaluation Script - Stage 1")
    print("="*60)
    
    # Parse arguments
    args = parse_args()
    print(f"[INFO] Arguments: {vars(args)}")
    
    try:
        # Setup headless rendering
        setup_headless_rendering()
        
        # Validate inputs and get dataset path
        dataset_path = validate_inputs(args)
        
        # Load trajectory data
        traj_data = load_trajectory_data(dataset_path, args.traj_no)
        
        # Apply sampling frequency
        sampled_traj = apply_sampling_frequency(traj_data, args.sampling_freq)
        
        # Initialize environment  
        env = initialize_environment(args.task_name)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INFO] Output directory: {args.output_dir}")
        
        # Stage 2: Extract frames from trajectory
        print("\n" + "="*60)
        print("STAGE 2: Frame Extraction")
        print("="*60)
        
        frames = extract_frames_from_trajectory(sampled_traj, env, args.task_name, args.traj_no)
        
        # Stage 3: Create and save video
        print("\n" + "="*60)
        print("STAGE 3: Video Creation & Saving")
        print("="*60)
        
        video_path = create_and_save_video(frames, args.task_name, args.traj_no, 
                                         args.output_dir, args.sampling_freq)
        
        print("\n" + "="*60)
        print("ALL STAGES COMPLETE: Trajectory Evaluation Success!")
        print("="*60)

        # Cleanup
        env.close()
        
    except Exception as e:
        print(f"\n[ERROR] Stage 1 failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
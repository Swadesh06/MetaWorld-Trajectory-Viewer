import copy
from collections import OrderedDict

import numpy as np
import quest.utils.file_utils as FileUtils
import quest.utils.obs_utils as ObsUtils
from PIL import Image
from quest.utils.dataset import SequenceDataset
from torch.utils.data import Dataset
from quest.utils.frame_stack import FrameStackObservationFixed
import torch
import torch.nn as nn
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
import math
import mujoco
import os
from torch.utils.data import ConcatDataset
import metaworld

from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE

from pyinstrument import Profiler


from metaworld.policies import *


_policies = OrderedDict(
    [
        ("assembly-v3", SawyerAssemblyV3Policy),
        ("basketball-v3", SawyerBasketballV3Policy),
        ("bin-picking-v3", SawyerBinPickingV3Policy),
        ("box-close-v3", SawyerBoxCloseV3Policy),
        ("button-press-topdown-v3", SawyerButtonPressTopdownV3Policy),
        ("button-press-topdown-wall-v3", SawyerButtonPressTopdownWallV3Policy),
        ("button-press-v3", SawyerButtonPressV3Policy),
        ("button-press-wall-v3", SawyerButtonPressWallV3Policy),
        ("coffee-button-v3", SawyerCoffeeButtonV3Policy),
        ("coffee-pull-v3", SawyerCoffeePullV3Policy),
        ("coffee-push-v3", SawyerCoffeePushV3Policy),
        ("dial-turn-v3", SawyerDialTurnV3Policy),
        ("disassemble-v3", SawyerDisassembleV3Policy),
        ("door-close-v3", SawyerDoorCloseV3Policy),
        ("door-lock-v3", SawyerDoorLockV3Policy),
        ("door-open-v3", SawyerDoorOpenV3Policy),
        ("door-unlock-v3", SawyerDoorUnlockV3Policy),
        ("drawer-close-v3", SawyerDrawerCloseV3Policy),
        ("drawer-open-v3", SawyerDrawerOpenV3Policy),
        ("faucet-close-v3", SawyerFaucetCloseV3Policy),
        ("faucet-open-v3", SawyerFaucetOpenV3Policy),
        ("hammer-v3", SawyerHammerV3Policy),
        ("hand-insert-v3", SawyerHandInsertV3Policy),
        ("handle-press-side-v3", SawyerHandlePressSideV3Policy),
        ("handle-press-v3", SawyerHandlePressV3Policy),
        ("handle-pull-v3", SawyerHandlePullV3Policy),
        ("handle-pull-side-v3", SawyerHandlePullSideV3Policy),
        ("peg-insert-side-v3", SawyerPegInsertionSideV3Policy),
        ("lever-pull-v3", SawyerLeverPullV3Policy),
        ("peg-unplug-side-v3", SawyerPegUnplugSideV3Policy),
        ("pick-out-of-hole-v3", SawyerPickOutOfHoleV3Policy),
        ("pick-place-v3", SawyerPickPlaceV3Policy),
        ("pick-place-wall-v3", SawyerPickPlaceWallV3Policy),
        ("plate-slide-back-side-v3", SawyerPlateSlideBackSideV3Policy),
        ("plate-slide-back-v3", SawyerPlateSlideBackV3Policy),
        ("plate-slide-side-v3", SawyerPlateSlideSideV3Policy),
        ("plate-slide-v3", SawyerPlateSlideV3Policy),
        ("reach-v3", SawyerReachV3Policy),
        ("reach-wall-v3", SawyerReachWallV3Policy),
        ("push-back-v3", SawyerPushBackV3Policy),
        ("push-v3", SawyerPushV3Policy),
        ("push-wall-v3", SawyerPushWallV3Policy),
        ("shelf-place-v3", SawyerShelfPlaceV3Policy),
        ("soccer-v3", SawyerSoccerV3Policy),
        ("stick-pull-v3", SawyerStickPullV3Policy),
        ("stick-push-v3", SawyerStickPushV3Policy),
        ("sweep-into-v3", SawyerSweepIntoV3Policy),
        ("sweep-v3", SawyerSweepV3Policy),
        ("window-close-v3", SawyerWindowCloseV3Policy),
        ("window-open-v3", SawyerWindowOpenV3Policy),
    ]
)
_env_names = list(_policies)

classes = {
    'ML45': {
        'train': ['assembly-v3', 
                  'basketball-v3', 
                  'button-press-topdown-v3', 
                  'button-press-topdown-wall-v3', 
                  'button-press-v3', 
                  'button-press-wall-v3', 
                  'coffee-button-v3', 
                  'coffee-pull-v3', 
                  'coffee-push-v3', 
                  'dial-turn-v3', 
                  'disassemble-v3', 
                  'door-close-v3', 
                  'door-open-v3', 
                  'drawer-close-v3', 
                  'drawer-open-v3', 
                  'faucet-close-v3', 
                  'faucet-open-v3', 
                  'hammer-v3', 
                  'handle-press-side-v3', 
                  'handle-press-v3', 
                  'handle-pull-side-v3', 
                  'handle-pull-v3', 
                  'lever-pull-v3', 
                  'peg-insert-side-v3', 
                  'peg-unplug-side-v3', 
                  'pick-out-of-hole-v3', 
                  'pick-place-v3', 
                  'pick-place-wall-v3', 
                  'plate-slide-back-side-v3', 
                  'plate-slide-back-v3', 
                  'plate-slide-side-v3', 
                  'plate-slide-v3', 
                  'push-back-v3', 
                  'push-v3', 
                  'push-wall-v3', 
                  'reach-v3', 
                  'reach-wall-v3', 
                  'shelf-place-v3', 
                  'soccer-v3', 
                  'stick-pull-v3', 
                  'stick-push-v3', 
                  'sweep-into-v3', 
                  'sweep-v3', 
                  'window-close-v3', 
                  'window-open-v3'],
        'test': ['bin-picking-v3', 
                 'box-close-v3', 
                 'door-lock-v3', 
                 'door-unlock-v3', 
                 'hand-insert-v3']
    },
    'ML45_PRISE': {
        'train': [
            'assembly-v3',
            'basketball-v3',
            'bin-picking-v3',
            'button-press-topdown-v3',
            'button-press-topdown-wall-v3',
            'button-press-v3',
            'button-press-wall-v3',
            'coffee-button-v3',
            'coffee-pull-v3',
            'coffee-push-v3',
            'dial-turn-v3',
            'door-close-v3',
            'door-lock-v3',
            'door-open-v3',
            'door-unlock-v3',
            'drawer-close-v3',
            'drawer-open-v3',
            'faucet-close-v3',
            'faucet-open-v3',
            'hammer-v3',
            'handle-press-side-v3',
            'handle-press-v3',
            'handle-pull-side-v3',
            'handle-pull-v3',
            'lever-pull-v3',
            'peg-insert-side-v3',
            'peg-unplug-side-v3',
            'pick-out-of-hole-v3',
            'pick-place-v3',
            'plate-slide-back-side-v3',
            'plate-slide-back-v3',
            'plate-slide-side-v3',
            'plate-slide-v3',
            'push-back-v3',
            'push-v3',
            'push-wall-v3',
            'reach-v3',
            'reach-wall-v3',
            'shelf-place-v3',
            'soccer-v3',
            'stick-push-v3',
            'sweep-into-v3',
            'sweep-v3',
            'window-close-v3',
            'window-open-v3'],
        'test': [
            'box-close-v3',
            'disassemble-v3',
            'hand-insert-v3',
            'pick-place-wall-v3',
            'stick-pull-v3',
        ]

    },
    'MT50': {
        'train': list(_env_names),
        'test': []
    }
}

def get_index(env_name):
    return _env_names.index(env_name)

def get_expert():
    env_experts = {
        env_name: _policies[env_name]() for env_name in _policies
    }

    def expert(obs, task_id):
        obs_gt = obs['obs_gt'].squeeze()
        return env_experts[_env_names[task_id]].get_action(obs_gt)
    
    return expert

def get_env_expert(env_name):
    return _policies[env_name]()

def get_benchmark(benchmark_name):
    benchmarks = {
        'ML1': metaworld.ML1,
        'ML10': metaworld.ML10,
        'ML45': metaworld.ML45,
        'MT50': metaworld.MT50,
        'ML45_PRISE': ML45PRISEBenchmark,
    }
    return benchmarks[benchmark_name]()

def get_env_names(benchmark=None, mode=None):
    if benchmark is None:
        return list(_env_names)
    
    if type(benchmark) is str:
        return classes[benchmark][mode]
    else:
        env_names = list(benchmark.train_classes \
            if mode == 'train' else benchmark.test_classes)
        env_names.sort()
        return env_names
    
def get_tasks(benchmark, mode):
    if benchmark is None:
        return []
    return benchmark.train_tasks if mode == 'train' else benchmark.test_tasks


class MetaWorldFrameStack(FrameStackObservationFixed):
    def __init__(self, 
                 env_name,
                 env_factory,
                 num_stack,
                 ):
        self.num_stack = num_stack
        
        env = env_factory(env_name)
        super().__init__(env, num_stack)

    def set_task(self, task):
        try:
            self.env.set_task(task)
        except AttributeError as e:
            if 'sawyer_observation_space' in str(e):
                # MetaWorld V3 API issue - task is already set, ignore this error
                pass
            else:
                raise e
    

class MetaWorldWrapper(gymnasium.Wrapper):
    def __init__(self, 
                 env_name: str,
                 shape_meta,
                 img_height: int = 128,
                 img_width: int = 128,
                 cameras=('corner2',),
                 env_kwargs=None,):
        if env_kwargs is None:
            env_kwargs = {}
        env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[f'{env_name}-goal-observable'](**env_kwargs)
        env._freeze_rand_vec = False
        super().__init__(env)
        # I don't know why
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        self.img_width = img_width
        self.img_height = img_height
        obs_meta = shape_meta['observation']
        self.rgb_outputs = list(obs_meta['rgb'])
        self.lowdim_outputs = list(obs_meta['lowdim'])

        self.cameras = cameras
        self.viewer = OffScreenViewer(
            env.model,
            env.data,
            img_width,
            img_height,
            env.mujoco_renderer.max_geom,
            env.mujoco_renderer._vopt,
        )

        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_meta['lowdim'][key],),
                dtype=np.float32
            )
        obs_space_dict['obs_gt'] = env.observation_space
        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def step(self, action):
        obs_gt, reward, terminated, truncated, info = super().step(action)
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt

        next_obs = self.make_obs(obs_gt)

        terminated = info['success'] == 1
        return next_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs_gt, info = super().reset()
        obs_gt = obs_gt.astype(np.float32)
        info['obs_gt'] = obs_gt

        obs = self.make_obs(obs_gt)

        return obs, info

    def make_obs(self, obs_gt):
        obs = {}
        obs['robot_states'] = np.concatenate((obs_gt[:4],obs_gt[18:22]))
        obs['obs_gt'] = obs_gt

        image_dict = {}
        for camera_name in self.cameras:
            image_obs = self.render(camera_name=camera_name, mode='rgb_array')
            image_dict[camera_name] = image_obs
        for key in self.rgb_outputs:
            obs[key] = image_dict[f'{key[:-4]}2'][::-1] # since generated dataset at the time had corner key instead of corner2

        return obs

    def render(self, camera_name=None, mode='rgb_array'):
        if camera_name is None:
            camera_name = self.cameras[0]
        cam_id = mujoco.mj_name2id(self.env.model, 
                                mujoco.mjtObj.mjOBJ_CAMERA, 
                                camera_name)
        
        return self.viewer.render(
            render_mode=mode,
            camera_id=cam_id
        )
    
    def set_task(self, task):
        try:
            self.env.set_task(task)
        except AttributeError as e:
            if 'sawyer_observation_space' in str(e):
                # MetaWorld V3 API issue - task is already set, ignore this error
                pass
            else:
                raise e
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.viewer.close()


def build_dataset(data_prefix, 
                  suite_name, 
                  benchmark_name, 
                  mode, 
                  seq_len, 
                  frame_stack,
                  shape_meta,
                  extra_obs_modality=None,
                  obs_seq_len=1, 
                  lowdim_obs_seq_len=None, 
                  load_obs=True,
                  n_demos=None,
                  load_next_obs=False,
                  dataset_keys=('actions',)
                  ):
    task_names = get_env_names(benchmark_name, mode)
    n_tasks = len(task_names)
    datasets = []

    obs_modality = {
        'rgb': list(shape_meta['observation']['rgb'].keys()),
        'low_dim': list(shape_meta['observation']['lowdim'].keys())
    }
    if extra_obs_modality is not None:
        for key in extra_obs_modality:
            obs_modality[key] = obs_modality[key] + extra_obs_modality[key]

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for task_name in task_names:
        # currently we assume tasks from same benchmark have the same shape_meta
        task_i_dataset = get_task_dataset(
            dataset_path=os.path.join(
                data_prefix, 
                suite_name,
                benchmark_name,
                mode,
                f"{task_name}.hdf5"
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            lowdim_obs_seq_len=lowdim_obs_seq_len,
            load_obs=load_obs,
            frame_stack=frame_stack,
            n_demos=n_demos,
            load_next_obs=load_next_obs,
            dataset_keys=dataset_keys
        )
        # loaded_datasets.append(task_i_dataset)
        task_id = get_index(task_name)
        datasets.append(SequenceVLDataset(task_i_dataset, task_id))
    n_demos = [dataset.n_demos for dataset in datasets]
    n_sequences = [dataset.total_num_sequences for dataset in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: MetaWorld")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    return concat_dataset


def get_task_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    lowdim_obs_seq_len=None,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    few_demos=None,
    load_obs=True,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=None,
):
    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    seq_len = seq_len
    filter_key = filter_key
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []

    if dataset_keys is None:
        dataset_keys = ['actions',]
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=load_next_obs,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        lowdim_obs_seq_length=lowdim_obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        few_demos=few_demos,
        n_demos=n_demos,
    )
    return dataset


class SequenceVLDataset(Dataset):
    # Note: task_id should be a string
    def __init__(self, sequence_dataset, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_id"] = self.task_id
        return return_dict


class ML45PRISEBenchmark(object):
    def __init__(self):
        benchmark = metaworld.ML45()
        all_classes = dict(benchmark.train_classes)
        all_classes.update(benchmark.test_classes)
        self.train_classes = {name: all_classes[name] for name in classes['ML45_PRISE']['train']}
        self.test_classes = {name: all_classes[name] for name in classes['ML45_PRISE']['test']}

        self.train_tasks = []
        self.test_tasks = []
        for task in benchmark.train_tasks + benchmark.test_tasks:
            if task.env_name in classes['ML45_PRISE']['train']:
                self.train_tasks.append(task)
            else:
                self.test_tasks.append(task)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief   utils
@author  MyeongJin Lee (menggu1234@robros.co.kr)
"""

import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

import IPython

e = IPython.embed


def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path_list,
        norm_stats,
        episode_ids,
        episode_len,
        chunk_size,
        robot_obs_size,
        policy_class,
        use_depth,
        config_loader=None,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.robot_obs_size = robot_obs_size
        self.config_loader = config_loader
        self.cumulative_len = np.cumsum(self.episode_len)
        if len(self.cumulative_len) == 0:
            raise ValueError("Dataset is empty. Please check your data directory.")
        print("dataset size: ", self.cumulative_len[-1])
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        
        self.use_depth = use_depth

        self.relative_action_mode = True
        self.relative_obs_mode = True

        self.is_sim = False
        self.action_pose_delay = 0 #tick
        self.obs_tracker_delay = 0 #tick
        
        self.__getitem__(0)  # initialize self.is_sim and self.transformations

    def __len__(self):
        return self.cumulative_len[-1]

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        t0 = time()
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                with h5py.File(dataset_path, "r") as root:
                    # Load actions
                    action_data = root["/actions"][()]
                    action_data = torch.from_numpy(np.array(action_data)).float()
                    data_len = action_data.shape[0]
                    
                    observation_data = root["/observations"][()]
                    observation_data = torch.from_numpy(np.array(observation_data)).float()
                    
                    # Load images if available
                    image_data = None
                    if "/images" in root:
                        image_data = root["/images"][()]
                        image_data = torch.from_numpy(np.array(image_data)).float()
                    
                    # Set the robot_state_sampling array
                    obs_sampling = np.clip(
                        range(
                            start_ts - 1 - self.obs_tracker_delay,
                            start_ts - 1 - self.obs_tracker_delay - self.robot_obs_size,
                            -1,
                        ),
                        0,
                        self.max_episode_len,
                    )
                    state_np = observation_data[obs_sampling]
                    
                    # Set the action sampling array
                    action_sampling = np.clip(
                        range(
                            start_ts + 1 + self.action_pose_delay,
                            start_ts + 1 + self.action_pose_delay + self.chunk_size,
                            1,
                        ),
                        0,
                        data_len - 1,
                    )

                    action_np = action_data[action_sampling]
                    
                    # Check if we need to pad the action data
                    if len(action_sampling) < self.chunk_size:
                        pad_len = self.chunk_size - len(action_sampling)
                        pad_action = np.zeros((pad_len, action_np.shape[1]))
                        action_np = np.concatenate([action_np, pad_action], axis=0)
                        is_pad = np.concatenate([
                            np.zeros(len(action_sampling)),
                            np.ones(pad_len)
                        ])
                    else:
                        is_pad = np.zeros(len(action_sampling))
                    
                    action_data = torch.from_numpy(np.array(action_np)).float()
                    state_data = torch.from_numpy(np.array(state_np)).float()
                    is_pad = torch.from_numpy(is_pad).bool()
                    
                    # Normalize robot state and action data
                    if self.norm_stats["action_mean"] is not None and self.norm_stats["action_std"] is not None:
                        action_data = (
                            action_data - self.norm_stats["action_mean"]
                        ) / self.norm_stats["action_std"]
                    
                    if self.norm_stats["state_mean"] is not None and self.norm_stats["state_std"] is not None:
                        state_data = (
                            state_data - self.norm_stats["state_mean"]
                        ) / self.norm_stats["state_std"]
                    
                    break
                    
            except Exception as e:
                print(f"Error loading {dataset_path} in __getitem__ (attempt {retry + 1}/{max_retries}): {e}")
                if retry < max_retries - 1:
                    # Try a different episode as fallback
                    fallback_index = (index + retry + 1) % len(self)
                    episode_id, start_ts = self._locate_transition(fallback_index)
                    dataset_path = self.dataset_path_list[episode_id]
                    print(f"Trying fallback episode: {dataset_path}")
                else:
                    # All retries failed, raise the error
                    raise RuntimeError(f"Failed to load any episode after {max_retries} attempts")
            
        if image_data is not None:
            # Sample images corresponding to the action sequence
            image_sampling = np.clip(
                range(
                    start_ts + 1 + self.action_pose_delay,
                    start_ts + 1 + self.action_pose_delay + self.chunk_size,
                    1,
                ),
                0,
                len(image_data) - 1,
            )
            image_np = image_data[image_sampling]
            return state_data, image_np, action_data, is_pad
        else:
            return state_data, action_data, is_pad

def get_episode_len(dataset_path_list):
    all_episode_len = []
    valid_dataset_paths = []
    
    print("Validating HDF5 files...")
    for i, dataset_path in enumerate(dataset_path_list):
        try:
            with h5py.File(dataset_path, "r") as root:
                episode_len = root["/observations"].shape[0]
                all_episode_len.append(episode_len)
                valid_dataset_paths.append(dataset_path)
                if (i + 1) % 10 == 0:
                    print(f"Validated {i + 1}/{len(dataset_path_list)} files...")
        except Exception as e:
            print(f"Error loading {dataset_path} in get_episode_len: {e}")
            print(f"Skipping corrupted file: {dataset_path}")
            continue
    
    if len(valid_dataset_paths) == 0:
        raise RuntimeError("No valid HDF5 files found in dataset!")
    
    print(f"Found {len(valid_dataset_paths)} valid files out of {len(dataset_path_list)} total files")
    
    dataset_path_list.clear()
    dataset_path_list.extend(valid_dataset_paths)
    
    return all_episode_len
    
def compute_norm_stats(dataset, batch_size=128, max_samples = 100000):
    """
    Computes normalization statistics for robot state and action tensors from the dataset.
    
    Args:
        dataset: a dataset instance with __getitem__ implemented
        batch_size: batch size for efficient loading

    Returns:
        norm_stats: dictionary containing mean, std, min, max for 'action' and 'state'
    """

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=20)

    all_actions = []
    all_states = []
    total_seen = 0 

    for i, batch in enumerate(loader):
        try:
            state_data, action_data, _ = batch
            # print(f'action_data.shape: {action_data.shape}')
            B, T, D = action_data.shape
            all_actions.append(action_data)
            # print(f'state_data.shape: {state_data.shape}')
            B, T, D = state_data.shape
            all_states.append(state_data)

            total_seen += B
            if total_seen >= max_samples:
                break
        except Exception as e:
            print(f"[ERROR IN BATCH {i}]: {e}")
            break

    all_actions = torch.cat(all_actions, dim=0)
    all_states = torch.cat(all_states, dim=0)

    all_episode_len = len(all_actions)
    norm_stats = {
        "action_mean": all_actions.mean(dim=0),
        "action_std": all_actions.std(dim=0) + 1e-2,  # avoid divide by zero

        "state_mean": all_states.mean(dim=0),
        "state_std": all_states.std(dim=0) + 1e-2,
    }

    return norm_stats, all_episode_len

def find_all_hdf5(dataset_dir):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = (
        np.array(sample_weights) / np.sum(sample_weights)
        if sample_weights is not None
        else None
    )
    sum_dataset_len_l = np.cumsum(
        [0] + [np.sum(episode_len) for episode_len in episode_len_l]
    )

    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(
                sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1]
            )
            batch.append(step_idx)
        yield batch


def load_data(
    dataset_dir_l,
    name_filter,
    batch_size_train,
    batch_size_val,
    chunk_size,
    robot_obs_size,
    load_pretrain=False,
    policy_class=None,
    stats_dir_l=None,
    sample_weights=None,
    train_ratio=0.95,
    use_depth=False,
    episode_num=-1,
    config_loader=None,
):
    print(f'Finding all hdf5 files in {dataset_dir_l}')
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [
        find_all_hdf5(dataset_dir) for dataset_dir in dataset_dir_l
    ]

    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)

    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]

    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # if episode num is -1 use every episode. if not, use sepcified number of episodes
    if episode_num == -1:
        num_episodes_l = [len(lst) for lst in dataset_path_list_list]
    else:
        num_episodes_l = [min(episode_num, len(lst)) for lst in dataset_path_list_list]

    num_episodes_cumsum = np.cumsum(num_episodes_l)
    num_episodes_0 = num_episodes_l[0]     

    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[: int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0) :]
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(num_episodes) + num_episodes_cumsum[idx]
        for idx, num_episodes in enumerate(num_episodes_l[1:])
    ]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(
        f"\n\nData from: {dataset_dir_l}\n - Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n"
    )

    all_episode_len = get_episode_len(dataset_path_list)
    
    train_episode_len_l = [
        [all_episode_len[i] for i in train_episode_ids]
        for train_episode_ids in train_episode_ids_l
    ]
    val_episode_len_l = [
        [all_episode_len[i] for i in val_episode_ids]
        for val_episode_ids in val_episode_ids_l
    ]
    
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]

    norm_stats = {
        "action_mean": None,
        "action_std": None,  # avoid divide by zero
        "state_mean": None,
        "state_std": None,
    }
    
    dataset_wo_norm_stats = EpisodicDataset(
        dataset_path_list,
        norm_stats,
        train_episode_ids,
        train_episode_len,
        chunk_size,
        robot_obs_size,
        policy_class,
        use_depth,  
        config_loader=config_loader
    )

    norm_stats, _ = compute_norm_stats(dataset_wo_norm_stats)

    train_dataset = EpisodicDataset(
        dataset_path_list,
        norm_stats,
        train_episode_ids,
        train_episode_len,
        chunk_size,
        robot_obs_size,
        policy_class,
        use_depth,
        config_loader=config_loader
    )
    val_dataset = EpisodicDataset(
        dataset_path_list,
        norm_stats,
        val_episode_ids,
        val_episode_len,
        chunk_size,
        robot_obs_size,
        policy_class,
        use_depth,
        config_loader=config_loader
    )
    
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)
    
    train_num_workers = 10
    val_num_workers = 10
    print(
        f"train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        pin_memory=True,
        num_workers=train_num_workers,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=batch_sampler_val,
        pin_memory=True,
        num_workers=val_num_workers,
        prefetch_factor=2,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief   Training script for imitation learning with ACT policy
@author  MyeongJin Lee
"""

import argparse
import os
import pickle
import sys
import time
from copy import deepcopy
from itertools import repeat
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
import torchvision.transforms.v2 as transforms

from utils.utils import load_data, compute_dict_mean, set_seed
from utils.policy import ACTPolicy

try:
    import wandb
except (ImportError, AttributeError):
    wandb = None


def setup_multi_gpu() -> Tuple[bool, int | None]:
    """Detect GPUs and decide whether to enable DataParallel."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return False, None
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPU(s)")
    if gpu_count <= 1:
        print("Single GPU or none detected — using single‑GPU training")
        return False, gpu_count
    print(f"Multiple GPUs detected ({gpu_count}), enabling DataParallel")
    torch.cuda.set_device(0)
    return True, gpu_count


def reduce_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce all tensor values in a dict to scalars by mean-ing all dims."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            if v.dim() > 0:
                v = v.mean()
            out[k] = v
        else:
            out[k] = v
    return out


def to_cuda(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    if torch.cuda.is_available():
        return tuple(t.cuda(non_blocking=True) for t in tensors)
    else:
        return tensors

def forward_pass(data, policy):
    # Handle both cases: with and without images
    if len(data) == 4:
        robot_proprio_data, image_data, action_data, is_pad = data
    else:
        robot_proprio_data, action_data, is_pad = data
        image_data = None
    
    try:
        if image_data is not None:
            robot_proprio_data, image_data, action_data, is_pad = to_cuda(
                robot_proprio_data, image_data, action_data, is_pad
            )
        else:
            robot_proprio_data, action_data, is_pad = to_cuda(
                robot_proprio_data, action_data, is_pad
            )
        batch_size = robot_proprio_data.shape[0]
        
        # Check if policy expects image data (ACT policy)
        if hasattr(policy, 'model') and hasattr(policy.model, 'backbones') and policy.model.backbones is not None:
            # ACT policy with vision - needs image data
            if image_data is not None:
                result = policy(robot_proprio_data, image_data, action_data, is_pad)
            else:
                result = policy(robot_proprio_data, None, action_data, is_pad)
        else:
            # MLP or CNNMLP policy - no image data needed
            result = policy(robot_proprio_data, action_data, is_pad)
        
        if isinstance(result, dict):
            result = reduce_metrics(result)
        return result
    except RuntimeError as e:
        print(f"CUDA error during forward pass: {str(e)}")
        print("Attempting to clear CUDA cache and retry...")
        torch.cuda.empty_cache()
        if hasattr(policy, 'model') and hasattr(policy.model, 'backbones') and policy.model.backbones is not None:
            if image_data is not None:
                result = policy(robot_proprio_data, image_data, action_data, is_pad)
            else:
                result = policy(robot_proprio_data, None, action_data, is_pad)
        else:
            result = policy(robot_proprio_data, action_data, is_pad)
        if isinstance(result, dict):
            result = reduce_metrics(result)
        return result


def forward_pass_with_masks(data, policy):
    # Handle both cases: with and without images
    if len(data) == 4:
        robot_proprio_data, image_data, action_data, is_pad = data
    else:
        robot_proprio_data, action_data, is_pad = data
        image_data = None
    
    if image_data is not None:
        robot_proprio_data, image_data, action_data, is_pad = to_cuda(
            robot_proprio_data, image_data, action_data, is_pad
        )
    else:
        robot_proprio_data, action_data, is_pad = to_cuda(
            robot_proprio_data, action_data, is_pad
        )

    # Check if policy expects image data (ACT policy)
    if hasattr(policy, 'model') and hasattr(policy.model, 'backbones') and policy.model.backbones is not None:
        # ACT policy with vision - needs image data
        if image_data is not None:
            result = policy(robot_proprio_data, image_data, action_data, is_pad)
        else:
            result = policy(robot_proprio_data, None, action_data, is_pad)
    else:
        # MLP or CNNMLP policy - no image data needed
        result = policy(robot_proprio_data, action_data, is_pad)
    
    if isinstance(result, dict):
        result = reduce_metrics(result)
    return result


def make_policy(policy_class: str, policy_config: Dict[str, Any]):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    elif policy_class == "MLP":
        from utils.policy import MLPPolicy
        return MLPPolicy(policy_config)
    elif policy_class == "CNNMLP":
        from utils.policy import CNNMLPPolicy
        return CNNMLPPolicy(policy_config)
    raise NotImplementedError(f"policy class {policy_class} is not defined")


def make_optimizer(policy_class: str, policy):
    if policy_class == "ACT":
        base = policy.module if hasattr(policy, "module") else policy
        return base.configure_optimizers()
    elif policy_class in ["MLP", "CNNMLP"]:
        base = policy.module if hasattr(policy, "module") else policy
        return base.configure_optimizers()
    raise NotImplementedError


def repeater(data_loader: Iterable):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f"Epoch {epoch} done")
        epoch += 1


def train_bc(train_dataloader, val_dataloader, config: Dict[str, Any]):
    num_steps: int = config["num_steps"]
    ckpt_dir: str = config["ckpt_dir"]
    seed: int = config["seed"]
    policy_class: str = config["policy_class"]
    policy_config: Dict[str, Any] = config["policy_config"]
    eval_every: int = config["eval_every"]
    validate_every: int = config["validate_every"]
    save_every: int = config["save_every"]
    is_wandb: bool = config["wandb"] and (wandb is not None)
    use_masks: bool = config["use_masks"]
    use_multi_gpu: bool = config.get("use_multi_gpu", False)
    gpu_count: int = int(config.get("gpu_count", 1) or 1)

    set_seed(seed)
    validation_iteration = 50

    print("Initializing policy…")
    policy = make_policy(policy_class, policy_config)

    if config["load_pretrain"]:
        print("Loading pretrained model…")
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        loading_status = policy.deserialize(
            torch.load(
                os.path.join(
                    config["pretrain_path"],
                    "policy_step_50000_seed_0.ckpt",
                ),
                map_location=map_location
            )
        )
        print(f"Loaded! {loading_status}")

    if config["resume_ckpt_path"] is not None:
        path = config["resume_ckpt_path"]
        print(f"Resuming from checkpoint: {path}")
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=map_location)
        loading_status = policy.deserialize(checkpoint)
        print(f"Resume policy from: {path}, Status: {loading_status}")

    if torch.cuda.is_available():
        print("Moving policy to CUDA…")
        policy.cuda()
    else:
        print("Using CPU for training…")

    if use_multi_gpu and gpu_count > 1:
        print(f"Enabling DataParallel with {gpu_count} GPUs")
        policy = torch.nn.DataParallel(policy)

    print("Creating optimizer…")
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = float("inf")
    best_ckpt_info = None

    print("Starting training loop…")
    train_iter = repeater(train_dataloader)

    for step in tqdm(range(num_steps), dynamic_ncols=True):
        policy.train()
        try:
            data = next(train_iter)
            forward_dict = (
                forward_pass_with_masks(data, policy) if use_masks else forward_pass(data, policy)
            )

            if not isinstance(forward_dict, dict) or "loss" not in forward_dict:
                print(f"Error: forward_dict is invalid at step {step}")
                continue

            # Ensure loss is scalar
            loss = forward_dict["loss"]
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            if not getattr(loss, "requires_grad", False):
                print(f"Warning: loss does not require gradients at step {step}")
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_wandb:
                wandb.log({k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in forward_dict.items()}, step=step)

            # Save
            if step % save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
                to_save = policy.module.serialize() if hasattr(policy, "module") else policy.serialize()
                torch.save(to_save, ckpt_path)
                print(f"Saved checkpoint at step {step}")

            # Validate
            if step % validate_every == 0 and step != 0:
                print(f"\nValidating at step {step}")
                with torch.inference_mode():
                    policy.eval()
                    validation_dicts = []
                    for batch_idx, data in tqdm(
                        enumerate(val_dataloader), total=validation_iteration, dynamic_ncols=True
                    ):
                        fdict = (
                            forward_pass_with_masks(data, policy)
                            if use_masks
                            else forward_pass(data, policy)
                        )
                        validation_dicts.append(fdict)
                        if batch_idx >= validation_iteration:
                            break

                    validation_summary = compute_dict_mean(validation_dicts)
                    epoch_val_loss = validation_summary["loss"]
                    if isinstance(epoch_val_loss, torch.Tensor) and epoch_val_loss.dim() > 0:
                        epoch_val_loss = epoch_val_loss.mean()

                    if float(epoch_val_loss) < min_val_loss:
                        min_val_loss = float(epoch_val_loss)
                        best_state = (
                            policy.module.serialize() if hasattr(policy, "module") else policy.serialize()
                        )
                        best_ckpt_info = (step, min_val_loss, deepcopy(best_state))
                        print(f"New best validation loss: {min_val_loss:.6f}")

                # prefix keys for logging readability
                val_log = {f"val_{k}": (v.item() if isinstance(v, torch.Tensor) else v) for k, v in validation_summary.items()}
                if is_wandb:
                    wandb.log(val_log, step=step)
                print(f"Val loss: {float(epoch_val_loss):.5f}")
                print(" ".join([f"{k}:{(v.item() if isinstance(v, torch.Tensor) else v):.3f}" for k, v in validation_summary.items()]))

        except Exception as e:  # keep training on recoverable errors
            print(f"Error during training step {step}: {str(e)}")
            print("Attempting to continue training…")
            continue

    print("Training finished, saving final checkpoint…")
    last_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    last_state = policy.module.serialize() if hasattr(policy, "module") else policy.serialize()
    torch.save(last_state, last_path)

    if best_ckpt_info is None:
        # Fallback if validation never improved
        best_ckpt_info = (num_steps - 1, float("inf"), deepcopy(last_state))

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    best_path = os.path.join(ckpt_dir, f"policy_best_seed_{seed}.ckpt")
    torch.save(best_state_dict, best_path)
    print(f"Training finished: Seed {seed}, val loss {min_val_loss:.6f} at step {best_step}")

    return best_ckpt_info


def main():
    parser = argparse.ArgumentParser(description="Train imitation learning policy")
    
    # Basic training arguments
    parser.add_argument("--task_name", type=str, default="pusht", help="Task name")
    parser.add_argument("--policy_class", type=str, default="ACT", help="Policy class")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=100000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    
    # Data arguments
    parser.add_argument("--dataset_dir", type=str, default="episodes", help="Dataset directory")
    parser.add_argument("--chunk_size", type=int, default=40, help="Action chunk size")
    parser.add_argument("--robot_obs_size", type=int, default=40, help="Robot observation size")
    parser.add_argument("--skip_mirrored_data", action="store_true", help="Skip mirrored data")
    parser.add_argument("--episode_num", type=int, default=-1, help="Number of episodes to use (-1 for all)")
    
    # Model arguments
    parser.add_argument("--kl_weight", type=int, default=10, help="KL divergence weight")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--use_depth", action="store_true", help="Use depth information")
    parser.add_argument("--use_masks", action="store_true", help="Use mask information")
    parser.add_argument("--use_vq", action="store_true", help="Use vector quantization")
    parser.add_argument("--vq_class", type=int, help="VQ class number")
    parser.add_argument("--vq_dim", type=int, help="VQ dimension")
    parser.add_argument("--no_encoder", action="store_true", help="Disable encoder")
    
    # Training arguments
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluation frequency")
    parser.add_argument("--validate_every", type=int, default=1000, help="Validation frequency")
    parser.add_argument("--save_every", type=int, default=1000, help="Checkpoint save frequency")
    parser.add_argument("--load_pretrain", action="store_true", help="Load pretrained model")
    parser.add_argument("--pretrain_path", type=str, default="checkpoints/pretrain", help="Pretrained model path")
    parser.add_argument("--resume_ckpt_path", type=str, help="Resume from checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    # Logging arguments
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--onscreen_render", action="store_true", help="Enable onscreen rendering")
    
    # Multi-GPU arguments
    parser.add_argument("--use_multi_gpu", action="store_true", help="Use multiple GPUs")
    
    args = parser.parse_args()
    
    # Setup multi-GPU
    use_multi_gpu, gpu_count = setup_multi_gpu()
    if args.use_multi_gpu:
        use_multi_gpu = True
    
    # Setup paths
    ckpt_dir = os.path.join(args.ckpt_dir, args.task_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Setup dataloader kwargs
    if torch.cuda.is_available():
        pin = True
    else:
        pin = False

    if use_multi_gpu:
        dataloader_kwargs = dict(
            num_workers=min(4 * (gpu_count or 1), 16),
            pin_memory=pin,
            persistent_workers=True,
            prefetch_factor=4,
            timeout=3000,
        )
        batch_size_train = args.batch_size * int(gpu_count or 1)
        batch_size_val = args.batch_size * int(gpu_count or 1)
        print(f"Multi‑GPU: batch size -> {batch_size_train} (train) / {batch_size_val} (val)")
    else:
        dataloader_kwargs = dict(
            num_workers=2,
            pin_memory=pin,
            persistent_workers=True if 2 > 0 else False,
            prefetch_factor=2,
            timeout=3000,
        )
        batch_size_train = args.batch_size
        batch_size_val = args.batch_size

    # Policy configuration
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    backbone = "resnet34"
    
    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "num_robot_observations": args.robot_obs_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": args.lr * 0.1,
        "backbone": backbone,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "nheads": nheads,
        "vq": args.use_vq,
        "vq_class": args.vq_class,
        "vq_dim": args.vq_dim,
        "action_dim": 2,  # Based on HDF5 data: actions shape is (220, 2)
        "state_dim": args.robot_obs_size,
        "no_encoder": args.no_encoder,
        "use_depth": args.use_depth,
        "camera_names": [],  # Empty for non-vision policies
    }

    # Run configuration
    run_config = {
        "num_steps": args.num_steps,
        "eval_every": args.eval_every,
        "validate_every": args.validate_every,
        "save_every": args.save_every,
        "ckpt_dir": ckpt_dir,
        "resume_ckpt_path": args.resume_ckpt_path,
        "episode_len": 1000,  # Default episode length
        "state_dim": args.robot_obs_size,
        "lr": args.lr,
        "policy_class": args.policy_class,
        "onscreen_render": args.onscreen_render,
        "policy_config": policy_config,
        "task_name": args.task_name,
        "seed": args.seed,
        "real_robot": True,
        "load_pretrain": args.load_pretrain,
        "pretrain_path": args.pretrain_path,
        "wandb": args.wandb and (wandb is not None),
        "use_masks": args.use_masks,
        "dataloader_kwargs": dataloader_kwargs,
        "episode_num": args.episode_num,
        "use_multi_gpu": use_multi_gpu,
        "gpu_count": gpu_count,
    }

    # Save configuration
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(run_config, f)

    # Initialize wandb
    if args.wandb and wandb is not None:
        wandb.init(project=args.task_name, reinit=True, name=os.path.basename(ckpt_dir))
        wandb.config.update(run_config)

    # Load dataset
    try:
        train_dl, val_dl, stats, _ = load_data(
            args.dataset_dir,
            lambda n: True,  # name_filter
            batch_size_train,
            batch_size_val,
            args.chunk_size,
            args.robot_obs_size,
            load_pretrain=args.load_pretrain,
            policy_class=args.policy_class,
            stats_dir_l=None,
            sample_weights=None,
            train_ratio=0.95,
            use_depth=args.use_depth,
            episode_num=args.episode_num,
            config_loader=None,
        )
    except Exception as e:
        print(f"Error loading data with workers={dataloader_kwargs.get('num_workers')}: {e}")
        print("Retrying with num_workers=0…")
        dataloader_kwargs["num_workers"] = 0
        dataloader_kwargs["persistent_workers"] = False
        train_dl, val_dl, stats, _ = load_data(
            args.dataset_dir,
            lambda n: True,
            batch_size_train,
            batch_size_val,
            args.chunk_size,
            args.robot_obs_size,
            load_pretrain=args.load_pretrain,
            policy_class=args.policy_class,
            stats_dir_l=None,
            sample_weights=None,
            train_ratio=0.95,
            use_depth=args.use_depth,
            episode_num=args.episode_num,
            config_loader=None,
        )

    # Rebuild dataloaders with unified kwargs
    if hasattr(train_dl, "dataset"):
        train_dl = torch.utils.data.DataLoader(
            train_dl.dataset,
            batch_size=batch_size_train,
            shuffle=True,
            **dataloader_kwargs,
        )
    if hasattr(val_dl, "dataset"):
        val_dl = torch.utils.data.DataLoader(
            val_dl.dataset,
            batch_size=batch_size_val,
            shuffle=False,
            **dataloader_kwargs,
        )

    # Save dataset stats
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # Start training
    best_ckpt_info = train_bc(train_dl, val_dl, run_config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # Save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best checkpoint saved: val loss {min_val_loss:.6f} @ step {best_step}")
    
    if args.wandb and wandb is not None:
        wandb.finish()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

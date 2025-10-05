# Imitation Learning with PushT

This repository implements imitation learning for the PushT task using different policy architectures including ACT (Action Chunking with Transformers), MLP, and CNN-MLP.

## Features

- **Multiple Policy Types**: Support for ACT, MLP, and CNN-MLP policies
- **Image Data Collection**: Automatic saving of gym-pusht screen images to dataset
- **Flexible Training**: Configurable training parameters and data loading
- **Interactive Data Collection**: Keyboard and mouse controls for data collection

## Installation

```bash
# Create conda environment
conda create -n il python=3.10
conda activate il

# Install dependencies
pip install torch torchvision
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops
pip install packaging h5py ipython
pip install gym-pusht
```

## Quick Start

### 1. Data Collection

Collect demonstration data using the interactive data collection script:

```bash
# Keyboard control (default)
python scripts/pusht_data_collection.py

# Mouse control
python scripts/pusht_data_collection.py --mouse

# With image saving (enabled by default)
python scripts/pusht_data_collection.py --save_img_obs
```

**Controls:**
- `w/a/s/d`: Move agent
- `space`: Place T and save data
- `r`: Reset episode
- `q`: Quit

### 2. Training

Train different policy types:

```bash
# MLP Policy (recommended for state-only data)
python train.py --policy_class MLP --num_steps 10000 --batch_size 16

# ACT Policy (for vision-based tasks)
python train.py --policy_class ACT --num_steps 10000 --batch_size 8

# CNN-MLP Policy
python train.py --policy_class CNNMLP --num_steps 10000 --batch_size 8
```

### 3. Key Parameters

- `--policy_class`: Choose from `MLP`, `ACT`, or `CNNMLP`
- `--num_steps`: Number of training steps
- `--batch_size`: Batch size for training
- `--chunk_size`: Action sequence length (default: 40)
- `--robot_obs_size`: Robot observation size (default: 40)
- `--lr`: Learning rate (default: 1e-5)

## Data Format

The collected data is saved in HDF5 format with the following structure:

```
episode_X.hdf5
├── observations: (N, 5) - Robot state observations
├── actions: (N, 2) - Action sequences
├── rewards: (N, 1) - Reward values
├── t_positions: (N, 2) - T-block positions
└── images: (N, H, W, 3) - Screen images (if enabled)
```

## Policy Types

### MLP Policy
- **Use case**: State-only imitation learning
- **Input**: Robot proprioceptive data
- **Architecture**: Simple multi-layer perceptron

### ACT Policy
- **Use case**: Vision-based imitation learning
- **Input**: Robot state + camera images
- **Architecture**: Transformer-based with ResNet backbone

### CNN-MLP Policy
- **Use case**: Vision-based tasks with simpler architecture
- **Input**: Robot state + camera images
- **Architecture**: CNN feature extractor + MLP

## Troubleshooting

### Common Issues

1. **NumPy compatibility error**: Use the correct conda environment
   ```bash
   conda activate il
   ```

2. **CUDA out of memory**: Reduce batch size
   ```bash
   python train.py --batch_size 4
   ```

3. **Data loading errors**: Check if episodes directory exists and contains valid HDF5 files

### Performance Tips

- Use `--num_workers 0` if you encounter data loading issues
- For vision-based policies, ensure images are collected with `--save_img_obs`
- Start with MLP policy for quick testing before moving to more complex architectures

## File Structure

```
├── scripts/
│   ├── pusht_data_collection.py  # Data collection script
│   └── pusht_example.py         # Example usage
├── utils/
│   ├── policy.py               # Policy implementations
│   └── utils.py                # Data loading utilities
├── detr/                       # DETR model implementation
├── episodes/                    # Collected data (HDF5 files)
├── checkpoints/                # Saved model checkpoints
└── train.py                   # Main training script
```
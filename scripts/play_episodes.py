#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class SimpleEpisodePlayer:
    """Simple episode replay player for PushT environment"""
    
    def __init__(self, data_dir="episodes", delay=0.1):
        self.data_dir = data_dir
        self.delay = delay
        
        # Setup visualization
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("PushT Episode Player")
        
        # Create patches for visualization
        self.agent = patches.Circle((0, 0), 10, color='red', label='Agent')
        self.block = patches.Circle((0, 0), 15, color='blue', label='T Block')
        self.target = patches.Circle((256, 256), 50, color='green', alpha=0.3, label='Target')
        
        self.ax.add_patch(self.agent)
        self.ax.add_patch(self.block)
        self.ax.add_patch(self.target)
        self.ax.legend()
        
        # Text display for step info
        self.step_text = self.ax.text(10, 480, "", fontsize=12, 
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
    def load_episode(self, episode_path):
        """Load episode data from HDF5 file"""
        try:
            with h5py.File(episode_path, 'r') as f:
                self.episode_data = {
                    'observations': f['observations'][:],
                    'actions': f['actions'][:],
                    'rewards': f['rewards'][:],
                    't_positions': f['t_positions'][:],
                    'metadata': {
                        'episode_id': f.attrs.get('episode_id', 0),
                        'timestamp': f.attrs.get('timestamp', 'unknown'),
                        'success': f.attrs.get('success', False),
                        'total_steps': f.attrs.get('total_steps', 0),
                        'save_img_obs': f.attrs.get('save_img_obs', False)
                    }
                }
                
            print(f"Loaded episode {self.episode_data['metadata']['episode_id']}")
            print(f"Total steps: {self.episode_data['metadata']['total_steps']}")
            print(f"Success: {self.episode_data['metadata']['success']}")
            print(f"Timestamp: {self.episode_data['metadata']['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading episode: {e}")
            return False
    
    def update_visualization(self, step_idx):
        """Update visualization for current step"""
        if self.episode_data is None or step_idx >= len(self.episode_data['observations']):
            return
            
        obs = self.episode_data['observations'][step_idx]
        reward = self.episode_data['rewards'][step_idx]
        action = self.episode_data['actions'][step_idx]
        
        # Extract positions from observation
        agent_x, agent_y = obs[0], obs[1]
        block_x, block_y = obs[2], obs[3]
        block_angle = obs[4]
        
        # Update agent position
        self.agent.center = (agent_x, agent_y)
        
        # Update block position
        self.block.center = (block_x, block_y)
        
        # Handle reward format (could be (1,) or scalar)
        if hasattr(reward, '__len__') and len(reward) > 0:
            reward_value = reward[0] if len(reward.shape) > 0 else float(reward)
        else:
            reward_value = float(reward)
        
        # Update step info
        self.step_text.set_text(f"Step: {step_idx}/{len(self.episode_data['observations'])-1}\n"
                               f"Reward: {reward_value:.6f}\n"
                               f"Action: [{action[0]:.1f}, {action[1]:.1f}]\n"
                               f"Agent: ({agent_x:.1f}, {agent_y:.1f})\n"
                               f"Block: ({block_x:.1f}, {block_y:.1f})")
        
        # Update title with episode info
        metadata = self.episode_data['metadata']
        self.ax.set_title(f"PushT Episode Player - Episode {metadata['episode_id']} "
                         f"({'Success' if metadata['success'] else 'Failed'})")
        
        self.fig.canvas.draw()
        plt.pause(0.001)
    
    def play_episode(self, episode_path, start_step=0, end_step=None):
        """Play the loaded episode"""
        if not self.load_episode(episode_path):
            return
        
        print(f"Starting playback of episode {self.episode_data['metadata']['episode_id']}")
        print("Press Ctrl+C to stop playback")
        
        # Initialize visualization
        self.update_visualization(start_step)
        
        # Play episode step by step
        try:
            total_steps = len(self.episode_data['observations'])
            if end_step is None:
                end_step = total_steps
            
            for step_idx in range(start_step, min(end_step, total_steps)):
                self.update_visualization(step_idx)
                time.sleep(self.delay)
                
        except KeyboardInterrupt:
            print("\nPlayback interrupted by user")
        finally:
            plt.close(self.fig)

def list_episodes(data_dir):
    """List available episodes"""
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return []
    
    episodes = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.hdf5'):
            filepath = os.path.join(data_dir, filename)
            try:
                with h5py.File(filepath, 'r') as f:
                    episode_id = f.attrs.get('episode_id', 0)
                    success = f.attrs.get('success', False)
                    total_steps = f.attrs.get('total_steps', 0)
                    timestamp = f.attrs.get('timestamp', 'unknown')
                    episodes.append({
                        'filepath': filepath,
                        'episode_id': episode_id,
                        'success': success,
                        'total_steps': total_steps,
                        'timestamp': timestamp
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return episodes

def main():
    parser = argparse.ArgumentParser(description='Simple PushT Episode Player')
    parser.add_argument('--data_dir', type=str, default='episodes',
                       help='Directory containing episode data')
    parser.add_argument('--episode', type=int, default=None,
                       help='Specific episode number to play (default: latest)')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific HDF5 file to play')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between steps in seconds (default: 0.1)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start step (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='End step (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available episodes and exit')
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    if not os.path.isabs(args.data_dir):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        args.data_dir = os.path.join(parent_dir, args.data_dir)
    
    # List episodes if requested
    if args.list:
        episodes = list_episodes(args.data_dir)
        if not episodes:
            print("No episodes found")
            return
        
        print("Available episodes:")
        for ep in episodes:
            status = "SUCCESS" if ep['success'] else "FAILED"
            print(f"  Episode {ep['episode_id']}: {ep['total_steps']} steps, {status}, {ep['timestamp']}")
        return
    
    # Find episode to play
    if args.file:
        # Play specific file
        if not os.path.isabs(args.file):
            episode_path = os.path.join(args.data_dir, args.file)
        else:
            episode_path = args.file
        
        if not os.path.exists(episode_path):
            print(f"File {episode_path} not found")
            return
    else:
        # Find episode by number
        episodes = list_episodes(args.data_dir)
        if not episodes:
            print("No episodes found")
            return
        
        if args.episode is not None:
            # Find specific episode
            episode_path = None
            for ep in episodes:
                if ep['episode_id'] == args.episode:
                    episode_path = ep['filepath']
                    break
            if episode_path is None:
                print(f"Episode {args.episode} not found")
                return
        else:
            # Play latest episode
            episode_path = episodes[-1]['filepath']
            print(f"Playing latest episode: {episodes[-1]['episode_id']}")
    
    # Create player and play episode
    player = SimpleEpisodePlayer(data_dir=args.data_dir, delay=args.delay)
    player.play_episode(episode_path, start_step=args.start, end_step=args.end)

if __name__ == "__main__":
    main()
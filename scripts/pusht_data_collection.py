#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import gymnasium as gym
import gym_pusht
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_auto_index(dataset_dir, dataset_name_prefix='', data_suffix='hdf5'):
    os.makedirs(dataset_dir, exist_ok=True)
    for i in range(10000):
        path = os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')
        if not os.path.isfile(path):
            return i
    raise RuntimeError("Too many episodes (>10000)")
        
class EnvViewer:
    def __init__(self, title="PushT Viewer", max_fps=30):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = None
        self.txt = None
        self.ax.axis('off')
        self.fig.canvas.manager.set_window_title(title)
        self.max_fps = max_fps
        self._min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_t = 0.0

    def show(self, frame: np.ndarray, reward: float | None = None):
        now = time.time()
        if now - self._last_t < self._min_dt:
            return
        if self.im is None:
            self.im = self.ax.imshow(frame)
            self.txt = self.ax.text(
                5, 15, "", color="w", fontsize=10, ha="left", va="top",
                bbox=dict(facecolor="black", alpha=0.4, pad=3, edgecolor="none")
            )
        else:
            self.im.set_data(frame)
        if reward is not None:
            self.txt.set_text(f"reward: {reward:.3f}")
            self.fig.canvas.manager.set_window_title(f"PushT Viewer  |  reward: {reward:.3f}")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        self._last_t = now

    def close(self):
        plt.close(self.fig)


class KeyboardController:
    def __init__(self):
        self.action_scale = 10.0  
    def start(self):
        print("Keyboard Controls:")
        print("  w/a/s/d: Move agent")
        print("  space: Place T and save data")
        print("  r: Reset episode")
        print("  q: Quit")
        print()
    def stop(self):
        pass
    def get_action(self):
        try:
            user_input = input("Command (w/a/s/d/space/r/q): ").strip().lower()
            if not user_input:
                return 'move', np.array([0.0, 0.0])
            if user_input == 'space' or user_input == ' ':
                return 'place', None
            elif user_input == 'r':
                return 'reset', None
            elif user_input == 'q':
                return 'quit', None
            delta_action = np.array([0.0, 0.0])
            for char in user_input:
                if char == 'w':   delta_action[1] += self.action_scale
                elif char == 's': delta_action[1] -= self.action_scale
                elif char == 'a': delta_action[0] -= self.action_scale
                elif char == 'd': delta_action[0] += self.action_scale
                else:
                    print(f"Invalid character '{char}'. Use w/a/s/d only.")
                    return 'move', np.array([0.0, 0.0])
            if np.any(delta_action != 0):
                print(f"Action: [{delta_action[0]:.1f}, {delta_action[1]:.1f}]")
                return 'move', delta_action
            else:
                return 'move', np.array([0.0, 0.0])
        except (EOFError, KeyboardInterrupt):
            return 'quit', None


class MouseController:
    def __init__(self, action_callback):
        self.action_callback = action_callback
        self.current_pos = np.array([256.0, 256.0])  
        self.action_scale = 1.0
        self.dragging = False
        self.last_pos = None
        self.setup_plot()

    def setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, 512)
        self.ax.set_ylim(0, 512)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("PushT Mouse Controller")
        self.target = patches.Circle((256, 256), 50, color='green', alpha=0.3, label='Target')
        self.ax.add_patch(self.target)
        self.agent = patches.Circle((256, 256), 10, color='red', label='Agent')
        self.ax.add_patch(self.agent)
        self.ax.legend()
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show(block=False)

    def on_mouse_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:
            self.dragging = True
            self.last_pos = (event.xdata, event.ydata)
        elif event.button == 3:
            self.place_t()
        elif event.button == 2:
            self.reset_episode()

    def on_mouse_drag(self, event):
        if not self.dragging or event.inaxes != self.ax or self.last_pos is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.last_pos[0]
        dy = event.ydata - self.last_pos[1]
        self.current_pos[0] = np.clip(self.current_pos[0] + dx, 0, 512)
        self.current_pos[1] = np.clip(self.current_pos[1] + dy, 0, 512)
        self.agent.center = (self.current_pos[0], self.current_pos[1])
        self.action_callback('move', self.current_pos.copy())
        self.last_pos = (event.xdata, event.ydata)
        self.fig.canvas.draw()

    def on_mouse_release(self, event):
        self.dragging = False
        self.last_pos = None

    def on_key_press(self, event):
        if event.key == 'w':
            self.current_pos[1] = np.clip(self.current_pos[1] + 10, 0, 512); self.update_position()
        elif event.key == 's':
            self.current_pos[1] = np.clip(self.current_pos[1] - 10, 0, 512); self.update_position()
        elif event.key == 'a':
            self.current_pos[0] = np.clip(self.current_pos[0] - 10, 0, 512); self.update_position()
        elif event.key == 'd':
            self.current_pos[0] = np.clip(self.current_pos[0] + 10, 0, 512); self.update_position()
        elif event.key == ' ':
            self.place_t()
        elif event.key == 'r':
            self.reset_episode()
        elif event.key == 'q':
            self.quit_app()

    def place_t(self):
        print("Place T - Saving data...")
        self.action_callback('place', None)

    def reset_episode(self):
        print("Reset episode")
        self.current_pos = np.array([256.0, 256.0])
        self.agent.center = (256, 256)
        self.fig.canvas.draw()
        self.action_callback('reset', None)

    def quit_app(self):
        print("Quit")
        self.action_callback('quit', None)
        plt.close(self.fig)

    def update_position(self):
        self.agent.center = (self.current_pos[0], self.current_pos[1])
        self.fig.canvas.draw()
        self.action_callback('move', self.current_pos.copy())

    def close(self):
        plt.close(self.fig)


class PushTDataCollector:
    def __init__(self, save_img_obs=False, data_dir="data", use_mouse=False, max_fps=30):
        self.save_img_obs = save_img_obs
        self.data_dir = data_dir
        self.use_mouse = use_mouse
        self.env = None
        self.controller = None
        self.mouse_controller = None
        self.viewer = EnvViewer(max_fps=max_fps)
        
        if self.save_img_obs:
            print("Image observation saving is ENABLED - images will be saved to dataset")
        else:
            print("Image observation saving is DISABLED - only state data will be saved")

        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            't_positions': [],  
            'images': [] if save_img_obs else None
        }
        self.episode_count = 0
        self.total_steps = 0
        self.successful_placements = 0
        os.makedirs(data_dir, exist_ok=True)
        self.last_reward = 0.0  

    def setup_environment(self):
        print("Initializing PushT environment...")
        self.env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        
        while hasattr(self.env, 'env') and hasattr(self.env, 'spec') and self.env.spec.max_episode_steps == 300:
            print(f"Removing TimeLimit wrapper: {type(self.env).__name__}")
            self.env = self.env.env
        
        from gymnasium.wrappers import TimeLimit
        self.env = TimeLimit(self.env, max_episode_steps=10000)
        print("Environment max episode steps set to 10000")
        
        if self.use_mouse:
            print("Setting up mouse controller...")
            self.mouse_controller = MouseController(self.handle_mouse_action)
        else:
            self.controller = KeyboardController()
            self.controller.start()
        print("Environment initialized successfully!")

    def handle_mouse_action(self, action_type, action):
        self.pending_action = (action_type, action)

    def is_successful_placement(self, observation, info):
        r = info.get('reward', self.last_reward) if isinstance(info, dict) else self.last_reward
        return r >= 0.95

    def save_episode_data(self, episode_data, success=False):
        episode_index = get_auto_index(self.data_dir, dataset_name_prefix='', data_suffix='hdf5')
        filename = f"episode_{episode_index}.hdf5"
        filepath = os.path.join(self.data_dir, filename)
        print(f"Saving episode {episode_index} to {filepath}")
        with h5py.File(filepath, 'w') as f:
            f.attrs['episode_id'] = episode_index
            f.attrs['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            f.attrs['success'] = success
            f.attrs['total_steps'] = len(episode_data['observations'])
            f.attrs['save_img_obs'] = self.save_img_obs
            f.create_dataset('observations', data=np.array(episode_data['observations']))
            f.create_dataset('actions', data=np.array(episode_data['actions']))
            rewards_array = np.array(episode_data['rewards']).reshape(-1, 1)
            f.create_dataset('rewards', data=rewards_array)
            f.create_dataset('t_positions', data=np.array(episode_data['t_positions']))
            if self.save_img_obs and episode_data['images']:
                images_array = np.array(episode_data['images'])
                f.create_dataset('images', data=images_array)
                print(f"Saved {len(episode_data['images'])} images with shape {images_array.shape}")
            elif self.save_img_obs and not episode_data['images']:
                print("Warning: Image saving enabled but no images collected")
        print(f"Episode {self.episode_count} saved to {filepath}")
        if success:
            print("Successful placement detected!")

    def reset_episode_data(self):
        self.episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            't_positions': [],
            'images': [] if self.save_img_obs else None
        }

    def _render_and_show(self, info=None):
        try:
            frame = self.env.render()
            r = None
            if isinstance(info, dict) and 'reward' in info:
                r = float(info['reward'])
            elif self.last_reward is not None:
                r = float(self.last_reward)
            if frame is not None:
                self.viewer.show(frame, r)
        except Exception:
            pass

    def collect_data(self):
        print("Starting data collection...")
        print("Move agent to push T object to target position")
        print("Press SPACE when T is in the correct position to save data")

        observation, info = self.env.reset()
        self.reset_episode_data()
        step_count = 0
        max_steps_per_episode = 1000
        self.pending_action = None

        self._render_and_show(info)

        while True:
            if self.use_mouse:
                if self.mouse_controller:
                    plt.pause(0.005)
                if self.pending_action is not None:
                    action_type, action = self.pending_action
                    self.pending_action = None
                else:
                    self._render_and_show(info)
                    continue
            else:
                action_type, action = self.controller.get_action()

            if action_type == 'quit':
                print("Quitting data collection...")
                break
            elif action_type == 'reset':
                print("Resetting episode...")
                observation, info = self.env.reset()
                self.reset_episode_data()
                step_count = 0
                self._render_and_show(info)
                continue
            elif action_type == 'toggle_img':
                self.save_img_obs = not self.save_img_obs
                print(f"Image observation saving: {'ON' if self.save_img_obs else 'OFF'}")
                continue
            elif action_type == 'place':
                if len(self.episode_data['observations']) == 0:
                    print("No data to save. Please move the agent first.")
                    continue
                
                success = self.is_successful_placement(observation, info)
                if success:
                    self.successful_placements += 1
                    print("Successful placement! Saving data...")
                else:
                    print("Placement not successful, but saving data anyway...")
                
                print(f"Saving episode with {len(self.episode_data['observations'])} steps...")
                self.save_episode_data(self.episode_data, success)
                self.episode_count += 1
                observation, info = self.env.reset()
                self.reset_episode_data()
                step_count = 0
                self._render_and_show(info)
                continue

            if action_type == 'move' and action is not None:
                action = np.clip(action.astype(np.float32), 0, 512)
                action[1] = 512 - action[1]  # flip Y
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.last_reward = float(reward)
                
                step_reward_component = step_count * 1e-6  # Very small increment per step
                unique_reward = self.last_reward + step_reward_component
              
                if len(observation) >= 4:
                    t_position = observation[2:4]  # block_x, block_y
                else:
                    t_position = np.array([0.0, 0.0])  # fallback

                self.episode_data['observations'].append(observation.copy())
                self.episode_data['actions'].append(action.copy())
                self.episode_data['rewards'].append(unique_reward)
                self.episode_data['t_positions'].append(t_position.copy())

                if self.save_img_obs:
                    try:
                        img = self.env.render()
                        if img is not None:
                            self.episode_data['images'].append(img.copy())
                            if step_count < 5:  # Debug info for first few steps
                                print(f"Step {step_count}: Saved image with shape {img.shape}")
                    except Exception as e:
                        print(f"Warning: Failed to save image at step {step_count}: {e}")
                        pass

                step_count += 1
                self.total_steps += 1

                self._render_and_show(info)

                if self.last_reward >= 0.98:
                    print(f"Success! Reward: {self.last_reward:.3f} - Auto saving and resetting...")
                    self.successful_placements += 1
                    self.save_episode_data(self.episode_data, True)
                    self.episode_count += 1
                    observation, info = self.env.reset()
                    self.reset_episode_data()
                    step_count = 0
                    self._render_and_show(info)
                    continue

                if terminated or truncated or step_count >= max_steps_per_episode:
                    print(f"Episode {self.episode_count} ended (steps: {step_count})")
                    if terminated:
                        print("Episode terminated by environment (success or failure condition)")
                    elif truncated:
                        print("Episode truncated by environment (time limit or other condition)")
                    elif step_count >= max_steps_per_episode:
                        print("Max steps reached, saving data...")
                    
                    if step_count >= max_steps_per_episode:
                        self.save_episode_data(self.episode_data, False)
                        self.episode_count += 1
                        observation, info = self.env.reset()
                        self.reset_episode_data()
                        step_count = 0
                        self._render_and_show(info)
                    else:
                        break

        if self.use_mouse and self.mouse_controller:
            self.mouse_controller.close()
        elif self.controller:
            self.controller.stop()
        if self.viewer:
            self.viewer.close()
        if self.env:
            self.env.close()

        print(f"\nData collection completed!")
        print(f"Total episodes: {self.episode_count}")
        print(f"Successful placements: {self.successful_placements}")
        print(f"Success rate: {self.successful_placements/max(1, self.episode_count)*100:.1f}%")
        print(f"Total steps: {self.total_steps}")
        print(f"Data saved to: {self.data_dir}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    parser = argparse.ArgumentParser(description='PushT Data Collection for Imitation Learning')
    parser.add_argument('--save_img_obs', action='store_true', default=True, help='Save simulation environment images to data (default: True)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(parent_dir, 'episodes'),
                        help='Directory to save collected data')
    parser.add_argument('--mouse', action='store_true', help='Use mouse controller instead of keyboard')
    parser.add_argument('--max_fps', type=int, default=30, help='Viewer max FPS (default: 30)')
    args = parser.parse_args()
    
    if args.data_dir is None:
        os.makedirs(args.data_dir, exist_ok=True)

    collector = PushTDataCollector(
        save_img_obs=args.save_img_obs,
        data_dir=args.data_dir,
        use_mouse=args.mouse,
        max_fps=args.max_fps
    )

    try:
        collector.setup_environment()
        collector.collect_data()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if collector.controller:
            collector.controller.stop()
        if collector.env:
            collector.env.close()

if __name__ == "__main__":
    main()

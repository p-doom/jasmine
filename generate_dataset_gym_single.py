"""
Generates a dataset from the gym environment.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import tyro
import time

import crafter


@dataclass
class Args:
    num_episodes: int = 10000
    env_name: str = "Acrobot-v1"
    min_episode_length: int = 50
    seed: int = 42


def main():
    args = tyro.cli(Args)
    output_dir = Path(f"data/{args.env_name}_episodes_{args.num_episodes}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate episodes ---
    i = 0
    metadata = []
    time_per_episode = []
    while i < args.num_episodes:
        time_start_episode = time.time()
        env = gym.make(args.env_name, render_mode="rgb_array")
        observation, info = env.reset(seed=args.seed)
        dataseq = []

        # --- Run episode ---
        for j in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            dataseq.append(env.render())
            if terminated or truncated:
                break
            
        # --- Save episode ---
        if len(dataseq) >= args.min_episode_length:
            episode_data = np.stack(dataseq, axis=0)
            episode_path = output_dir / f"episode_{i}.npy"
            np.save(episode_path, episode_data.astype(np.uint8))
            time_per_episode.append(time.time() - time_start_episode)
            metadata.append({"path": str(episode_path), "length": len(dataseq)})
            if i % 5 == 0:
                print(f"Episode {i} completed, length: {len(dataseq)}, time: {time_per_episode[-1]} seconds")
            else:
                print(f"Episode {i} completed, length: {len(dataseq)}")
            i += 1
        else:
            print(f"Episode too short ({len(dataseq)}), resampling...")

    # --- Save metadata ---
    np.save(output_dir / "metadata.npy", metadata)
    print(f"Dataset generated with {len(metadata)} valid episodes, saving to {output_dir}")
    print(f"Average time per episode: {np.mean(time_per_episode)} seconds")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

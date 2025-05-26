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
import multiprocessing as mp

@dataclass
class Args:
    num_episodes: int = 10000
    env_name: str = "Acrobot-v1"
    min_episode_length: int = 50
    seed: int = 42


def generate_episode(args_tuple):
    env_name, min_episode_length, seed, episode_idx, output_dir = args_tuple
    env = gym.make(env_name, render_mode="rgb_array")
    observation, info = env.reset(seed=seed + episode_idx)
    dataseq = []
    print(f"Episode {episode_idx} started")
    for j in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        dataseq.append(env.render())
        if terminated or truncated:
            break
    if len(dataseq) >= min_episode_length:
        episode_data = np.stack(dataseq, axis=0)
        episode_path = output_dir / f"episode_{episode_idx}.npy"
        np.save(episode_path, episode_data.astype(np.uint8))
        print(f"Episode {episode_idx} saved")

        return {"path": str(episode_path), "length": len(dataseq)}
    else:
        return None

def main():
    args = tyro.cli(Args)
    output_dir = Path(f"data/{args.env_name}_episodes_{args.num_episodes}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_args = [
        (args.env_name, args.min_episode_length, args.seed, i, output_dir)
        for i in range(args.num_episodes)
    ]

    print(f"Number of processes: {mp.cpu_count()}")

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(generate_episode, pool_args)

    # Filter out None (episodes that were too short)
    metadata = [r for r in results if r is not None]
    np.save(output_dir / "metadata.npy", metadata)
    print(f"Dataset generated with {len(metadata)} valid episodes, saving to {output_dir}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

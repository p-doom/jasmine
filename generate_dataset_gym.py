"""
Generates a dataset from the gym environment.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path

import gym3
import numpy as np
import tyro
import time

@dataclass
class Args:
    num_episodes: int = 10000
    env_name: str = "Acrobot-v1"
    min_episode_length: int = 50


def main():
    args = tyro.cli(Args)
    output_dir = Path(f"data/{args.env_name}_episodes_{args.num_episodes}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate episodes ---
    i = 0
    metadata = []
    while i < args.num_episodes:
        env = gym3.vectorize_gym(num=1, env_kwargs={"id": args.env_name})
        dataseq = []

        # --- Run episode ---
        for j in range(1000):
            env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
            rew, obs, first = env.observe()
            dataseq.append(obs)
            if first:
                break

        # --- Save episode ---
        if len(dataseq) >= args.min_episode_length:
            episode_data = np.concatenate(dataseq, axis=0)
            episode_path = output_dir / f"episode_{i}.npy"
            np.save(episode_path, episode_data.astype(np.uint8))
            metadata.append({"path": str(episode_path), "length": len(dataseq)})
            print(f"Episode {i} completed, length: {len(dataseq)}")
            i += 1
        else:
            print(f"Episode too short ({len(dataseq)}), resampling...")

    # --- Save metadata ---
    np.save(output_dir / "metadata.npy", metadata)
    print(f"Dataset generated with {len(metadata)} valid episodes, saving to {output_dir}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path
import time

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
import tyro


@dataclass
class Args:
    num_episodes: int = 10000
    env_name: str = "coinrun"
    min_episode_length: int = 50


args = tyro.cli(Args)
output_dir = Path(f"data/{args.env_name}_episodes")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Generate episodes ---
i = 0
metadata = []
times= []
while i < args.num_episodes:
    seed = np.random.randint(0, 10000)
    env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
    dataseq = []

    # --- Run episode ---
    print(f"Generating episode {i}...")
    start_time = time.time()
    for j in range(1000):
        env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        dataseq.append(obs["rgb"])
        if first:
            break

    # --- Save episode ---
    if len(dataseq) >= args.min_episode_length:
        episode_data = np.concatenate(dataseq, axis=0)
        episode_path = output_dir / f"episode_{i}.npy"
        np.save(episode_path, episode_data.astype(np.uint8))
        metadata.append({"path": str(episode_path), "length": len(dataseq)})
        # print time per episode
        times.append(time.time() - start_time)
        print(f"Episode {i} completed, length: {len(dataseq)}, time: {time.time() - start_time}")
        i += 1
    else:
        print(f"Episode too short ({len(dataseq)}), resampling...")


# --- Save metadata ---
np.save(output_dir / "metadata.npy", metadata)
print(f"Dataset generated with {len(metadata)} valid episodes")
print(f"Average time per episode: {np.mean(times)}")

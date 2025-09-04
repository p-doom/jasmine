"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
import tyro
import pickle
import json
from array_record.python.array_record_module import ArrayRecordWriter 



@dataclass
class Args:
    num_episodes: int = 10000
    output_dir: str = "data/coinrun_episodes"
    min_episode_length: int = 50


args = tyro.cli(Args)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# --- Generate episodes ---
i = 0
episode_metadata = []
while i < args.num_episodes:
    seed = np.random.randint(0, 10000)
    env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
    observations_seq = []

    # --- Run episode ---
    for j in range(1000):
        env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        observations_seq.append(obs["rgb"])
        if first:
            break

    # --- Save episode ---
    if len(observations_seq) >= args.min_episode_length:
        observations_data = np.concatenate(observations_seq, axis=0).astype(np.uint8)
        episode_path = output_dir / f"episode_{i}.array_record"  

        # --- Save as ArrayRecord ---
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        record = {"raw_video": observations_data.tobytes(), "sequence_length": len(observations_seq)}
        writer.write(pickle.dumps(record))
        writer.close()

        episode_metadata.append({"path": str(episode_path), "length": len(observations_seq)})
        print(f"Episode {i} completed, length: {len(observations_seq)}")
        i += 1
    else:
        print(f"Episode too short ({len(observations_seq)}), resampling...")

# --- Save metadata ---
metadata = {
    "env": "coinrun",
    "num_episodes": args.num_episodes,
    "avg_episode_len": np.mean([ep["length"] for ep in episode_metadata]),
    "episode_metadata": episode_metadata,
}
with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"Dataset generated with {len(episode_metadata)} valid episodes")

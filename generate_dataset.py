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
    num_episodes_train: int = 10000
    num_episodes_val: int = 500
    num_episodes_test: int = 500
    output_dir: str = "data/coinrun_episodes"
    min_episode_length: int = 50
    seed: int = 0


args = tyro.cli(Args)

def generate_episodes(num_episodes, split):
    i = 0
    episode_metadata = []
    while i < num_episodes:
        seed = np.random.randint(0, 10000)
        env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
        observations_seq = []

        # --- Run episode ---
        for _ in range(1000):
            env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
            _ , obs, first = env.observe()
            observations_seq.append(obs["rgb"])
            if first:
                break

        # --- Save episode ---
        if len(observations_seq) >= args.min_episode_length:
            observations_data = np.concatenate(observations_seq, axis=0).astype(np.uint8)
            episode_path = args.output_dir / split / f"episode_{i}.array_record"  

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
    print(f"Done generating {split} split")
    return episode_metadata


def main():
    # Set random seed and create dataset directories
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # --- Generate episodes ---
    train_episode_metadata = generate_episodes(args.num_episodes_train, "train")
    val_episode_metadata = generate_episodes(args.num_episodes_val, "val")
    test_episode_metadata = generate_episodes(args.num_episodes_test, "test")

    # --- Save metadata ---
    metadata = {
        "env": "coinrun",
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": np.mean([ep["length"] for ep in train_episode_metadata]),
        "avg_episode_len_val": np.mean([ep["length"] for ep in val_episode_metadata]),
        "avg_episode_len_test": np.mean([ep["length"] for ep in test_episode_metadata]),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,

    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"Done generating dataset.")

if __name__ == "__main__":
    main()

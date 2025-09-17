"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
import tyro
import json
import os
from utils import save_chunks


@dataclass
class Args:
    num_episodes_train: int = 10000
    num_episodes_val: int = 500
    num_episodes_test: int = 500
    output_dir: str = "data/coinrun_episodes"
    min_episode_length: int = 1000
    max_episode_length: int = 1000
    chunk_size: int = 160
    chunks_per_file: int = 100
    seed: int = 0


args = tyro.cli(Args)
assert args.max_episode_length >= args.min_episode_length, "Maximum episode length must be greater than or equal to minimum episode length."

if args.min_episode_length < args.chunk_size:
    print("Warning: Minimum episode length is smaller than chunk size. Note that episodes shorter than the chunk size will be discarded.")

# --- Generate episodes ---
def generate_episodes(num_episodes, split):
    episode_idx = 0
    episode_metadata = []
    chunks = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    while episode_idx < num_episodes:
        seed = np.random.randint(0, 10000)
        env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
    
        observations_seq = []
        episode_chunks = []

        # --- Run episode ---
        for step_t in range(args.max_episode_length):
            env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
            _, obs, first = env.observe()
            observations_seq.append(obs["rgb"])
            if len(observations_seq) == args.chunk_size:
                episode_chunks.append(observations_seq)
                observations_seq = []
            if first:
                break

        # --- Save episode ---
        if step_t + 1  >= args.min_episode_length:
            if len(observations_seq) < args.chunk_size:
                print(
                    f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq)} frames, "
                    f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                    "This might lead to performance degradation during training."
                )
            episode_chunks.append(observations_seq)
            chunks_data = [np.concatenate(seq, axis=0).astype(np.uint8) for seq in episode_chunks]
            chunks.extend(chunks_data)


            ep_metadata, chunks, file_idx = save_chunks(chunks, file_idx, args.chunks_per_file, output_dir_split)
            episode_metadata.extend(ep_metadata)

            print(f"Episode {episode_idx} completed, length: {step_t + 1}.")
            episode_idx += 1
        else:
            print(f"Episode too short ({step_t + 1}), resampling...")

    if len(chunks) > 0:
        print(f"Warning: Dropping {len(chunks)} chunks for consistent number of chunks per file.",
        "Consider changing the chunk_size and chunks_per_file parameters to prevent data-loss.")

    print(f"Done generating {split} split")
    return episode_metadata


def main():
    # Set random seed and create dataset directories
    np.random.seed(args.seed)
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
        "avg_episode_len_train": np.mean([ep["avg_seq_len"] for ep in train_episode_metadata]),
        "avg_episode_len_val": np.mean([ep["avg_seq_len"] for ep in val_episode_metadata]),
        "avg_episode_len_test": np.mean([ep["avg_seq_len"] for ep in test_episode_metadata]),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,

    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Done generating dataset.")

if __name__ == "__main__":
    main()


"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path

from gym3 import types_np
import os
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
    max_episode_length: int = 1000
    chunk_size: int = 160
    chunks_per_file: int = 100
    seed: int = 0


args = tyro.cli(Args)

def generate_episodes(num_episodes, split):
    episode_idx = 0
    episode_metadata = []
    while episode_idx < num_episodes:
        seed = np.random.randint(0, 10000)
        env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
        observations_seq = []

        # --- Run episode ---
        for _ in range(args.max_episode_length):
            env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
            _ , obs, first = env.observe()
            observations_seq.append(obs["rgb"])
            if first:
                break

        # --- Save episode ---
        if len(observations_seq) >= args.min_episode_length:
            observations_data = np.concatenate(observations_seq, axis=0).astype(np.uint8)

            file_chunks = []
            for start_idx in range(0, observations_data.shape[0] - args.chunk_size + 1, args.chunk_size):
                chunk = observations_data[start_idx : start_idx + args.chunk_size]

                chunk_record = {
                    "raw_video": chunk.tobytes(),
                    "sequence_length": args.chunk_size,
                }

                file_chunks.append(chunk_record)
            # --- Save as ArrayRecord ---
            for file_idx in range(0, len(file_chunks), args.chunks_per_file):
                batch_chunks = file_chunks[file_idx : file_idx + args.chunks_per_file]
                episode_path = os.path.join(args.output_dir, split, f"episode_{episode_idx:04d}_part_{file_idx:04d}.array_record")   
                writer = ArrayRecordWriter(str(episode_path), "group_size:1")
                for chunk in batch_chunks:
                    writer.write(pickle.dumps(chunk))
                writer.close()
                episode_metadata.append({"path": episode_path, "chunk_size": args.chunk_size, "num_chunks": len(batch_chunks)})
                print(f"Created {episode_path} with {len(batch_chunks)} video chunks")
            
            print(f"Episode {episode_idx} completed, length: {len(observations_seq)}. Saved across {len(file_chunks) // args.chunks_per_file} array_records files.")
            episode_idx += 1
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

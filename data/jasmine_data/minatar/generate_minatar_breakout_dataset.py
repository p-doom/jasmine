"""
Generates a dataset of random-action Breakout episodes using MinAtar.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
import numpy as np
import os
import json
import tyro

from minatar import Environment
from utils import save_chunks


@dataclass
class Args:
    num_episodes_train: int = 5000000
    num_episodes_val: int = 25000
    num_episodes_test: int = 25000
    output_dir: str = "data/breakout_episodes"
    min_episode_length: int = 20
    max_episode_length: int = 500
    chunk_size: int = 50
    chunks_per_file: int = 100
    seed: int = 0


args = tyro.cli(Args)

assert (
    args.max_episode_length >= args.min_episode_length
), "Maximum episode length must be >= minimum episode length."
if args.min_episode_length < args.chunk_size:
    print(
        "Warning: Minimum episode length is smaller than chunk size. "
        "Episodes shorter than the chunk size will be discarded."
    )


def _obs_to_rgb(obs):
    # Define a color matrix for each boolean combination
    color_matrix = np.array(
        [
            [0, 0, 0],  # Black for all False
            [128, 0, 0],  # Maroon
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Navy
            [128, 128, 0],  # Olive
            [128, 0, 128],  # Purple
            [0, 128, 128],  # Teal
            [192, 192, 192],  # Silver
            [128, 128, 128],  # Gray
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 255, 255],  # White
        ],
        dtype=np.uint8,
    )

    # Convert boolean array to integer indices
    indices = obs.dot(1 << np.arange(obs.shape[-1] - 1, -1, -1))

    # Map indices to colors using matrix multiplication
    obs = color_matrix[indices]
    return obs


def generate_episodes(num_episodes: int, split: str):
    episode_idx = 0
    episode_metadata = []
    obs_chunks = []
    act_chunks = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    os.makedirs(output_dir_split, exist_ok=True)

    while episode_idx < num_episodes:
        env = Environment("breakout", sticky_action_prob=0.0)  # typical MinAtar setup
        env.reset()
        obs_seq, act_seq = [], []
        episode_obs_chunks, episode_act_chunks = [], []

        step_t = 0
        for step_t in range(args.max_episode_length):
            obs = env.state()  # shape: (10,10,num_channels)
            obs = _obs_to_rgb(obs)
            action = np.random.randint(env.num_actions())
            _, done = env.act(action)
            obs_seq.append(obs.astype(np.uint8))
            act_seq.append(action)

            if len(obs_seq) == args.chunk_size:
                episode_obs_chunks.append(np.stack(obs_seq))
                episode_act_chunks.append(np.array(act_seq))
                obs_seq, act_seq = [], []

            if done:
                break

        if step_t + 1 >= args.min_episode_length:
            if obs_seq:
                if len(obs_seq) < args.chunk_size:
                    print(
                        f"Warning: Inconsistent chunk sizes. Episode has {len(obs_seq)} frames "
                        f"(less than chunk_size {args.chunk_size})."
                    )
                episode_obs_chunks.append(np.stack(obs_seq))
                episode_act_chunks.append(np.array(act_seq))

            obs_chunks_data = episode_obs_chunks
            act_chunks_data = episode_act_chunks
            obs_chunks.extend(obs_chunks_data)
            act_chunks.extend(act_chunks_data)

            ep_metadata, file_idx, obs_chunks, act_chunks = save_chunks(
                file_idx, args.chunks_per_file, output_dir_split, obs_chunks, act_chunks
            )
            episode_metadata.extend(ep_metadata)

            print(f"[{split}] Episode {episode_idx} completed, length: {step_t + 1}")
            episode_idx += 1
        else:
            print(f"Episode too short ({step_t + 1}), resampling...")

    if len(obs_chunks) > 0:
        print(
            f"Warning: Dropping {len(obs_chunks)} leftover chunks. "
            "Consider adjusting chunk_size or chunks_per_file."
        )

    print(f"Done generating {split} split.")
    return episode_metadata


def get_action_space() -> int:
    env = Environment("breakout")
    return env.num_actions()


def main():
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_meta = generate_episodes(args.num_episodes_train, "train")
    val_meta = generate_episodes(args.num_episodes_val, "val")
    test_meta = generate_episodes(args.num_episodes_test, "test")

    metadata = {
        "env": "MinAtar-Breakout",
        "num_actions": get_action_space(),
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": float(
            np.mean([ep["avg_seq_len"] for ep in train_meta])
        ),
        "avg_episode_len_val": float(np.mean([ep["avg_seq_len"] for ep in val_meta])),
        "avg_episode_len_test": float(np.mean([ep["avg_seq_len"] for ep in test_meta])),
        "episode_metadata_train": train_meta,
        "episode_metadata_val": val_meta,
        "episode_metadata_test": test_meta,
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Done generating dataset.")


if __name__ == "__main__":
    main()

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
from data.jasmine_data.utils import save_chunks


@dataclass
class Args:
    env_name: str = "coinrun"
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
assert (
    args.max_episode_length >= args.min_episode_length
), "Maximum episode length must be greater than or equal to minimum episode length."

if args.min_episode_length < args.chunk_size:
    print(
        "Warning: Minimum episode length is smaller than chunk size. Note that episodes shorter than the chunk size will be discarded."
    )


# --- Generate episodes ---
def generate_episodes(num_episodes, split, start_level, env_name):
    episode_idx = 0
    episode_metadata = []
    obs_chunks = []
    act_chunks = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    while episode_idx < num_episodes:
        env = ProcgenGym3Env(
            num=1, env_name=env_name, start_level=start_level + episode_idx
        )

        observations_seq = []
        actions_seq = []
        episode_obs_chunks = []
        episode_act_chunks = []

        # --- Run episode ---
        step_t = 0
        first_obs = True
        for step_t in range(args.max_episode_length):
            _, obs, first = env.observe()
            action = types_np.sample(env.ac_space, bshape=(env.num,))
            env.act(action)
            observations_seq.append(obs["rgb"])
            actions_seq.append(action)
            if len(observations_seq) == args.chunk_size:
                episode_obs_chunks.append(observations_seq)
                episode_act_chunks.append(actions_seq)
                observations_seq = []
                actions_seq = []
            if first and not first_obs:
                break
            first_obs = False

        # --- Save episode ---
        if step_t + 1 >= args.min_episode_length:
            if observations_seq:
                if len(observations_seq) < args.chunk_size:
                    print(
                        f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq)} frames, "
                        f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                        "This might lead to performance degradation during training."
                    )
                episode_obs_chunks.append(observations_seq)
                episode_act_chunks.append(actions_seq)

            obs_chunks_data = [
                np.concatenate(seq, axis=0).astype(np.uint8)
                for seq in episode_obs_chunks
            ]
            act_chunks_data = [
                np.concatenate(act, axis=0) for act in episode_act_chunks
            ]
            obs_chunks.extend(obs_chunks_data)
            act_chunks.extend(act_chunks_data)

            ep_metadata, file_idx, obs_chunks, act_chunks = save_chunks(
                file_idx, args.chunks_per_file, output_dir_split, obs_chunks, act_chunks
            )
            episode_metadata.extend(ep_metadata)

            print(f"Episode {episode_idx} completed, length: {step_t + 1}.")
            episode_idx += 1
        else:
            print(f"Episode too short ({step_t + 1}), resampling...")

    if len(obs_chunks) > 0:
        print(
            f"Warning: Dropping {len(obs_chunks)} chunks for consistent number of chunks per file.",
            "Consider changing the chunk_size and chunks_per_file parameters to prevent data-loss.",
        )

    print(f"Done generating {split} split")
    return episode_metadata


def get_action_space():
    env = ProcgenGym3Env(num=1, env_name=args.env_name, start_level=0)
    return env.ac_space.eltype.n


def main():
    # Set random seed and create dataset directories
    np.random.seed(args.seed)
    train_start_level = np.random.randint(0, 1000)
    val_start_level = train_start_level + args.num_episodes_train
    test_start_level = val_start_level + args.num_episodes_val

    # --- Generate episodes ---
    train_episode_metadata = generate_episodes(
        args.num_episodes_train, "train", train_start_level, args.env_name
    )
    val_episode_metadata = generate_episodes(
        args.num_episodes_val, "val", val_start_level, args.env_name
    )
    test_episode_metadata = generate_episodes(
        args.num_episodes_test, "test", test_start_level, args.env_name
    )

    # --- Save metadata ---
    metadata = {
        "env": args.env_name,
        "num_actions": get_action_space(),
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": np.mean(
            [ep["avg_seq_len"] for ep in train_episode_metadata]
        ),
        "avg_episode_len_val": np.mean(
            [ep["avg_seq_len"] for ep in val_episode_metadata]
        ),
        "avg_episode_len_test": np.mean(
            [ep["avg_seq_len"] for ep in test_episode_metadata]
        ),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Done generating dataset.")


if __name__ == "__main__":
    main()

"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from typing import Sequence

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
import tyro
import json
import os
import pickle
from array_record.python.array_record_module import ArrayRecordWriter


@dataclass
class Args:
    env_name: str = "coinrun"
    num_episodes_train: int = 10000
    num_episodes_val: int = 500
    num_episodes_test: int = 500
    output_dir: str = "data/coinrun_episodes"
    min_episode_length: int = 1000
    max_episode_length: int = 1000
    seed: int = 0


args = tyro.cli(Args)
assert (
    args.max_episode_length >= args.min_episode_length
), "Maximum episode length must be greater than or equal to minimum episode length."


# --- Generate episodes ---
def generate_episodes(num_episodes, split, start_level, env_name):
    episode_idx = 0
    episode_metadata = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    os.makedirs(output_dir_split, exist_ok=True)

    total_sequence_length = 0
    while episode_idx < num_episodes:
        env = ProcgenGym3Env(
            num=1, env_name=env_name, start_level=start_level + episode_idx
        )

        observations_seq = []
        actions_seq = []

        # --- Run episode ---
        step_t = 0
        first_obs = True
        for step_t in range(args.max_episode_length):
            _, obs, first = env.observe()
            action = types_np.sample(env.ac_space, bshape=(env.num,))
            env.act(action)
            observations_seq.append(obs["rgb"])
            actions_seq.append(action)
            if first and not first_obs:
                break
            first_obs = False

        if step_t + 1 >= args.min_episode_length:
            episode_data = np.concatenate(observations_seq, axis=0).astype(np.uint8)

            # save as array record
            episode_path = os.path.join(
                output_dir_split, f"episode_{episode_idx}.array_record"
            )
            writer = ArrayRecordWriter(str(episode_path), "group_size:1")
            writer.write(
                pickle.dumps(
                    {"raw_video": episode_data.tobytes(), "sequence_length": step_t + 1}
                )
            )
            total_sequence_length += step_t + 1
            writer.close()

            # save episode metadata
            episode_metadata.append({"path": episode_path, "length": step_t + 1})
            episode_idx += 1

    print(f"Done generating {split} split")
    return total_sequence_length


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
    train_total_sequence_length = generate_episodes(
        args.num_episodes_train, "train", train_start_level, args.env_name
    )
    val_total_sequence_length = generate_episodes(
        args.num_episodes_val, "val", val_start_level, args.env_name
    )
    test_total_sequence_length = generate_episodes(
        args.num_episodes_test, "test", test_start_level, args.env_name
    )

    # --- Save metadata ---
    metadata = {
        "env": args.env_name,
        "num_actions": get_action_space(),
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "total_sequence_length_train": train_total_sequence_length,
        "total_sequence_length_val": val_total_sequence_length,
        "total_sequence_length_test": test_total_sequence_length,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Done generating dataset.")


if __name__ == "__main__":
    main()

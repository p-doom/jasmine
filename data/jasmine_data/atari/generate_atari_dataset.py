import json
import os
from dataclasses import dataclass
from typing import List, Optional

import gin
import numpy as np
import tyro
from dopamine.discrete_domains import atari_lib
from dopamine.jax.agents.rainbow.rainbow_agent import JaxRainbowAgent
from jasmine_data.utils import save_chunks


@dataclass
class Args:
    gin_files: Optional[List[str]] = None
    gin_bindings: Optional[List[str]] = None

    capture_dataset: bool = True
    num_transitions_train: int = 10_000_000
    num_transitions_val: int = 500_000
    num_transitions_test: int = 500_000
    output_dir: str = "data/atari_episodes"
    min_episode_length: int = 1
    chunk_size: int = 160
    chunks_per_file: int = 100
    stop_on_complete: bool = True
    seed: int = 0


if __name__ == "__main__":
    args = tyro.cli(Args)

    gin_files = [] if args.gin_files is None else list(args.gin_files)
    gin_bindings = [] if args.gin_bindings is None else list(args.gin_bindings)

    assert gin_files, "No gin files provided"
    gin.parse_config_files_and_bindings(
        gin_files, bindings=gin_bindings, skip_unknown=False
    )

    env = atari_lib.create_atari_environment()
    rng = np.random.RandomState(args.seed)
    num_actions = env.action_space.n
    agent = JaxRainbowAgent(num_actions=num_actions, seed=args.seed)

    split_targets: dict[str, int] = {
        "train": args.num_transitions_train,
        "val": args.num_transitions_val,
        "test": args.num_transitions_test,
    }
    splits_in_order = [s for s in ["train", "val", "test"] if split_targets[s] > 0]
    transitions_captured_per_split: dict[str, int] = {
        s: 0 for s in ["train", "val", "test"]
    }
    file_idx_by_split: dict[str, int] = {s: 0 for s in ["train", "val", "test"]}
    episode_metadata_by_split: dict[str, list[dict]] = {
        s: [] for s in ["train", "val", "test"]
    }

    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []

    if args.capture_dataset and splits_in_order:
        os.makedirs(os.path.join(args.output_dir, splits_in_order[0]), exist_ok=True)

    current_split_idx = 0 if splits_in_order else -1
    current_split = splits_in_order[current_split_idx] if splits_in_order else None
    split_dir = (
        os.path.join(args.output_dir, current_split)
        if current_split is not None
        else None
    )

    def should_continue_capturing() -> bool:
        return any(
            transitions_captured_per_split[s] < split_targets[s]
            for s in splits_in_order
        )

    while True:
        if (
            args.capture_dataset
            and args.stop_on_complete
            and not should_continue_capturing()
        ):
            break

        observation = env.reset()
        observations_seq: list[np.ndarray] = []
        actions_seq: list[int] = []
        total_reward = 0.0

        action = agent.begin_episode(observation)

        while True:
            if args.capture_dataset and current_split is not None:
                frame = np.asarray(observation, dtype=np.uint8)
                assert (
                    frame.ndim == 3
                ), f"Frame has {frame.ndim} dimensions, expected a trailing singleton dimension"
                observations_seq.append(frame)
                actions_seq.append(int(action))

            next_obs, reward, terminal, _info = env.step(int(action))
            total_reward += float(reward)

            if terminal:
                agent.end_episode(reward=float(reward), terminal=True)
                break

            action = agent.step(float(reward), next_obs)
            observation = next_obs

        if (
            args.capture_dataset
            and current_split is not None
            and should_continue_capturing()
        ):
            current_len = len(observations_seq)
            if current_len >= args.min_episode_length:
                frames = np.stack(observations_seq, axis=0).astype(
                    np.uint8
                )  # (T, H, W, 1)
                acts = np.asarray(actions_seq, dtype=np.int8)  # (T,)
                # Convert grayscale to 3-channel RGB by repeating values
                frames_rgb = np.repeat(frames, 3, axis=-1)  # (T, H, W, 3)

                episode_obs_chunks = []
                episode_act_chunks = []
                start_idx = 0
                while start_idx < current_len:
                    end_idx = min(start_idx + args.chunk_size, current_len)
                    episode_obs_chunks.append(frames_rgb[start_idx:end_idx])
                    episode_act_chunks.append(acts[start_idx:end_idx])
                    start_idx = end_idx

                obs_chunks.extend([seq.astype(np.uint8) for seq in episode_obs_chunks])
                act_chunks.extend([act for act in episode_act_chunks])

                assert split_dir is not None
                os.makedirs(split_dir, exist_ok=True)
                (
                    ep_metadata,
                    file_idx_by_split[current_split],
                    obs_chunks,
                    act_chunks,
                ) = save_chunks(
                    file_idx_by_split[current_split],
                    args.chunks_per_file,
                    split_dir,
                    obs_chunks,
                    act_chunks,
                )
                episode_metadata_by_split[current_split].extend(ep_metadata)
                transitions_captured_per_split[current_split] += current_len

                if (
                    transitions_captured_per_split[current_split]
                    >= split_targets[current_split]
                ):
                    obs_chunks = []
                    act_chunks = []
                    if current_split_idx + 1 < len(splits_in_order):
                        current_split_idx += 1
                        current_split = splits_in_order[current_split_idx]
                        split_dir = os.path.join(args.output_dir, current_split)
                        os.makedirs(split_dir, exist_ok=True)
            # else: episode too short; skip

    # Write metadata
    if args.capture_dataset:
        os.makedirs(args.output_dir, exist_ok=True)
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
        else:
            metadata = {}

        try:
            game_name = gin.query_parameter(
                "atari_lib.create_atari_environment.game_name"
            )
        except Exception:
            game_name = None
        metadata.setdefault("num_actions", int(num_actions))
        for split in ["train", "val", "test"]:
            metadata.setdefault(f"num_transitions_{split}", 0)
            metadata.setdefault(f"avg_episode_len_{split}", 0.0)
            metadata.setdefault(f"episode_metadata_{split}", [])

        for split_key in splits_in_order:
            ep_meta_list = episode_metadata_by_split[split_key]
            if ep_meta_list:
                metadata[f"episode_metadata_{split_key}"].extend(ep_meta_list)
                metadata[f"num_transitions_{split_key}"] = int(
                    np.sum(
                        [
                            ep["avg_seq_len"]
                            for ep in metadata[f"episode_metadata_{split_key}"]
                        ]
                    )
                )
                metadata[f"avg_episode_len_{split_key}"] = float(
                    np.mean(
                        [
                            ep["avg_seq_len"]
                            for ep in metadata[f"episode_metadata_{split_key}"]
                        ]
                    )
                )

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

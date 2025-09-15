import os
import math
from dataclasses import dataclass

import tyro
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from tqdm import tqdm


@dataclass
class DownloadDQNReplayPNGs:
    """CLI options for downloading frames from the RLU DQN Replay (Atari) dataset.

    The dataset name follows: rlu_atari_checkpoints_ordered/{game}_run_{run_number}
    """

    game: str = "Pong"
    run_number: int = 1
    output_dir: str = "data/dqn_replay_pngs"

    # Percentage of episodes to load per split (0 < percent <= 100)
    data_percent: float = 100.0

    # Save every Nth frame (1 = save all frames)
    frame_stride: int = 1

    # Limit total episodes processed across all splits (-1 for no limit)
    max_episodes: int = -1

    # Overwrite existing PNG files if they exist
    overwrite: bool = False


def _normalize_image_to_uint8(image: tf.Tensor) -> tf.Tensor:
    """Converts an image tensor to uint8 [H, W, C]."""
    # Ensure rank 2 or 3
    if image.shape.rank == 2:
        image = tf.expand_dims(image, axis=-1)
    # Some datasets might store a trailing singleton channel; keep as-is
    if image.dtype != tf.uint8:
        image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    return image


def _build_split_string(builder: tfds.core.DatasetBuilder, percent: float) -> str:
    """Constructs a tfds split string covering all available splits.

    Applies a per-split episode percentage if percent < 100.
    """
    splits = list(builder.info.splits.keys())
    if not splits:
        raise RuntimeError("No splits found for the specified dataset.")

    if percent >= 100.0:
        return "+".join(splits)

    split_parts = []
    for split in splits:
        num_episodes = builder.info.splits[split].num_examples or 0
        take_n = max(1, int(math.floor((percent / 100.0) * num_episodes)))
        split_parts.append(f"{split}[:{take_n}]")
    return "+".join(split_parts)


def download_pngs(
    game: str,
    run_number: int,
    output_dir: str,
    data_percent: float,
    frame_stride: int,
    max_episodes: int,
    overwrite: bool,
) -> None:
    dataset_name = f"rlu_atari_checkpoints_ordered/{game}_run_{run_number}"

    base_output_dir = os.path.join(output_dir, f"{game}_run_{run_number}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Discover available splits and construct split selection string
    builder = tfds.builder(dataset_name)
    split_str = _build_split_string(builder, data_percent)

    # Load the dataset of episodes (RLDS format)
    episodes = tfds.load(
        dataset_name,
        split=split_str,
        shuffle_files=True,
        read_config=tfds.ReadConfig(enable_ordering_guard=False),
    )

    episodes_processed = 0
    frames_saved = 0

    episode_bar = tqdm(desc="Episodes", unit="ep")

    for episode in episodes:
        if max_episodes > 0 and episodes_processed >= max_episodes:
            break

        episode_dir = os.path.join(base_output_dir, f"episode_{episodes_processed:06d}")
        os.makedirs(episode_dir, exist_ok=True)

        step_bar = tqdm(desc=f"Frames e{episodes_processed:06d}", unit="f", leave=False)
        step_idx = 0
        # Iterate steps within an episode
        for step in episode[rlds.STEPS]:
            if frame_stride > 1 and (step_idx % frame_stride) != 0:
                step_idx += 1
                continue

            obs = step[rlds.OBSERVATION]
            img = _normalize_image_to_uint8(obs)

            # Ensure shape is [H, W, C]
            if img.shape.rank == 2:
                img = tf.expand_dims(img, axis=-1)

            file_path = os.path.join(episode_dir, f"frame_{step_idx:06d}.png")
            if not overwrite and os.path.exists(file_path):
                step_idx += 1
                step_bar.update(1)
                continue

            png_bytes = tf.io.encode_png(img)
            tf.io.write_file(file_path, png_bytes)
            frames_saved += 1
            step_idx += 1
            step_bar.update(1)

        step_bar.close()
        episodes_processed += 1
        episode_bar.update(1)

    episode_bar.close()
    print(
        f"Done. Episodes processed: {episodes_processed}, frames saved: {frames_saved}. Output: {base_output_dir}"
    )


if __name__ == "__main__":
    args = tyro.cli(DownloadDQNReplayPNGs)

    print(f"Game: {args.game}")
    print(f"Run number: {args.run_number}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data percent: {args.data_percent}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Overwrite: {args.overwrite}")

    download_pngs(
        game=args.game,
        run_number=args.run_number,
        output_dir=args.output_dir,
        data_percent=args.data_percent,
        frame_stride=args.frame_stride,
        max_episodes=args.max_episodes,
        overwrite=args.overwrite,
    )

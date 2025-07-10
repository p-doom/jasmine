from dataclasses import dataclass

import tensorflow as tf
import concurrent.futures
import numpy as np
import logging
import tyro
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


@dataclass
class Args:
    source_data_dir: str = "data/coinrun_episodes"
    output_tfrecords_dir: str = "data/coinrun_tfrecords"
    num_shards: int = 50


args = tyro.cli(Args)


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord_example(episode_numpy_array):
    feature = {
        "height": _int64_feature(episode_numpy_array.shape[1]),
        "width": _int64_feature(episode_numpy_array.shape[2]),
        "channels": _int64_feature(episode_numpy_array.shape[3]),
        "sequence_length": _int64_feature(episode_numpy_array.shape[0]),
        "raw_video": _bytes_feature(episode_numpy_array.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def process_shard(shard_idx, episode_paths, output_filename):
    """Process a single shard: load episodes, write to one TFRecord file."""
    with tf.io.TFRecordWriter(output_filename) as writer:
        for npy_path in tqdm(
            episode_paths,
            desc=f"Shard {shard_idx:03d}",
            leave=False,
        ):
            try:
                episode_data = np.load(npy_path)
                tf_example = create_tfrecord_example(episode_data)
                writer.write(tf_example.SerializeToString())
            except Exception as e:
                logging.error(f"Shard {shard_idx}: Skipping {npy_path} due to error: {e}")

def main_preprocess(data_dir_str, output_dir_str, num_shards):
    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = np.load(data_dir / "metadata.npy", allow_pickle=True)
    episode_source_paths = [Path(item["path"]) for item in metadata]
    num_total_episodes = len(episode_source_paths)

    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if num_shards > num_total_episodes:
        logging.warning(
            f"Warning: num_shards ({num_shards}) is greater than total episodes ({num_total_episodes}). "
            f"Setting num_shards to {num_total_episodes}."
        )
        num_shards = num_total_episodes

    logging.info(
        f"Preparing to write {num_total_episodes} episodes to {num_shards} TFRecord shards in {output_dir}..."
    )

    output_filenames = [
        str(output_dir / f"shard-{i:05d}-of-{num_shards:05d}.tfrecord")
        for i in range(num_shards)
    ]

    # Split episode paths into shards
    shards = [[] for _ in range(num_shards)]
    for idx, npy_path in enumerate(episode_source_paths):
        shards[idx % num_shards].append(npy_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_shards) as executor:
        futures = []
        for shard_idx, (shard_paths, out_fname) in enumerate(zip(shards, output_filenames)):
            futures.append(
                executor.submit(process_shard, shard_idx, shard_paths, out_fname)
            )
        for f in tqdm(concurrent.futures.as_completed(futures), total=num_shards, desc="Shards"):
            f.result()  # Propagate exceptions

    logging.info(
        f"TFRecord sharding complete. {num_shards} shards written to {output_dir}."
    )
    logging.info("Generated shard files:")
    for fname in output_filenames:
        logging.info(f"  {fname}")


if __name__ == "__main__":
    if (
        not Path(args.source_data_dir).exists()
        or not (Path(args.source_data_dir) / "metadata.npy").exists()
    ):
        logging.error(f"Please generate data in '{args.source_data_dir}' first.")
    else:
        main_preprocess(
            args.source_data_dir, args.output_tfrecords_dir, args.num_shards
        )

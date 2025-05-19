import tensorflow as tf
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord_example(episode_numpy_array):
    feature = {
        'height': _int64_feature(episode_numpy_array.shape[1]),
        'width': _int64_feature(episode_numpy_array.shape[2]),
        'channels': _int64_feature(episode_numpy_array.shape[3]),
        'sequence_length': _int64_feature(episode_numpy_array.shape[0]),
        'raw_video': _bytes_feature(episode_numpy_array.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

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
        logging.warning(f"Warning: num_shards ({num_shards}) is greater than total episodes ({num_total_episodes}). "
              f"Setting num_shards to {num_total_episodes}.")
        num_shards = num_total_episodes


    logging.info(f"Preparing to write {num_total_episodes} episodes to {num_shards} TFRecord shards in {output_dir}...")

    output_filenames = [
        str(output_dir / f"shard-{i:05d}-of-{num_shards:05d}.tfrecord")
        for i in range(num_shards)
    ]
    writers = [tf.io.TFRecordWriter(filename) for filename in output_filenames]

    writer_idx_for_episode = 0
    try:
        for i, npy_path in enumerate(episode_source_paths):
            if i % 100 == 0 and i > 0:
                logging.info(f"  Processed {i}/{num_total_episodes} episodes...")
            try:
                episode_data = np.load(npy_path)
                tf_example = create_tfrecord_example(episode_data)

                current_writer = writers[writer_idx_for_episode]
                current_writer.write(tf_example.SerializeToString())

                writer_idx_for_episode = (writer_idx_for_episode + 1) % num_shards

            except Exception as e:
                logging.error(f"Skipping {npy_path} due to error: {e}")
    finally:
        for writer in writers:
            writer.close()
        logging.info(f"TFRecord sharding complete. {num_shards} shards written to {output_dir}.")
        logging.info("Generated shard files:")
        for fname in output_filenames:
            logging.info(f"  {fname}")

if __name__ == '__main__':
    source_data_dir = "data/coinrun_episodes"
    output_tfrecords_dir = "data_tfrecords"
    NUM_SHARDS = 50

    if not Path(source_data_dir).exists() or not (Path(source_data_dir) / "metadata.npy").exists():
        logging.error(f"Please generate data in '{source_data_dir}' first.")
    else:
        main_preprocess(source_data_dir, output_tfrecords_dir, NUM_SHARDS)
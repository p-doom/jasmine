import functools
import jax

import tensorflow as tf


# --- TensorFlow function for processing: slicing, normalization ---
def _tf_process_episode(episode_tensor, seq_len, image_h, image_w, image_c):
    """
    Processes a raw episode tensor in TensorFlow.
    Takes a full episode, extracts a random sequence, and normalizes it.
    Args:
        episode_tensor: A TensorFlow tensor representing a full video episode.
                        Expected shape: (dynamic_length, image_h, image_w, image_c)
                        Expected dtype: e.g., tf.uint8 (raw pixel values)
        seq_len: The desired length of the sub-sequence to extract.
        image_h: The height of each frame.
        image_w: The width of each frame.
        image_c: The number of channels in each frame.
    Returns:
        A TensorFlow tensor representing the processed video sequence.
        Shape: (seq_len, image_h, image_w, image_c)
        Dtype: tf.float32 (normalized pixel values)
    """
    current_episode_len = tf.shape(episode_tensor)[0]

    max_start_idx = current_episode_len - seq_len

    start_idx = tf.random.uniform(
        shape=(), minval=0, maxval=max_start_idx + 1, dtype=tf.int32
    )

    seq = episode_tensor[start_idx : start_idx + seq_len]

    seq = tf.cast(seq, tf.float32) / 255.0

    # Ensure the final shape is statically known for batching.
    # tf.reshape is robust, but tf.ensure_shape or set_shape can also be used if confident.
    processed_sequence = tf.reshape(seq, [seq_len, image_h, image_w, image_c])

    return processed_sequence


def _parse_tfrecord_fn(example_proto, image_h, image_w, image_c):
    feature_description = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "sequence_length": tf.io.FixedLenFeature([], tf.int64),
        "raw_video": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    video_shape = (example["sequence_length"], image_h, image_w, image_c)

    episode_tensor = tf.io.decode_raw(example["raw_video"], out_type=tf.uint8)
    episode_tensor = tf.reshape(episode_tensor, video_shape)

    episode_tensor = tf.ensure_shape(episode_tensor, [None, image_h, image_w, image_c])
    return episode_tensor


def get_dataloader(
    tfrecord_paths: list[str],  # List of TFRecord file paths
    seq_len: int,
    global_batch_size: int,
    image_h: int,
    image_w: int,
    image_c: int,
    shuffle_buffer_size: int = 1000,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    cache_processed_data: bool = True,
    seed: int = 42,
):
    """
    Creates a tf.data.Dataset pipeline from TFRecord files.
    """
    if not tfrecord_paths:
        raise ValueError("tfrecord_paths list cannot be empty.")

    process_id = jax.process_index()
    num_processes = jax.process_count()

    assert (
        global_batch_size % num_processes == 0
    ), "Global batch size {global_batch_size} \
        must be divisible by the number of JAX processes {num_processes} for proper sharding."
    per_process_batch_size = global_batch_size // num_processes

    dataset = tf.data.TFRecordDataset(
        tfrecord_paths, num_parallel_reads=tf.data.AUTOTUNE
    )

    dataset = dataset.shard(num_shards=num_processes, index=process_id)

    # (f.srambical) NOTE: For TFRecords, it's often good to have a large shuffle buffer.
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )
    parse_fn = functools.partial(
        _parse_tfrecord_fn, image_h=image_h, image_w=image_w, image_c=image_c
    )
    dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)

    dataset = dataset.cache() if cache_processed_data else dataset

    tf_process_fn = functools.partial(
        _tf_process_episode,
        seq_len=seq_len,
        image_h=image_h,
        image_w=image_w,
        image_c=image_c,
    )
    dataset = dataset.map(tf_process_fn, num_parallel_calls=num_parallel_calls)

    dataset = dataset.repeat(None)
    dataset = dataset.batch(per_process_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset.as_numpy_iterator()

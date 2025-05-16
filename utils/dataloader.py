from pathlib import Path
import functools

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# --- Helper function to be wrapped by tf.numpy_function ---
def _load_and_process_numpy(path_bytes, seq_len):
    path = path_bytes.decode('utf-8')
    episode = np.load(path)
    
    if len(episode) < seq_len:
        raise ValueError(f"Episode length {len(episode)} is less than seq_len {seq_len}. Consider adapting your data generation process or filtering the dataset.")

    start_idx = np.random.randint(0, len(episode) - seq_len + 1)
    seq = episode[start_idx : start_idx + seq_len]
    return (seq.astype(np.float32) / 255.0)


def get_tf_dataloader(data_dir_str: str, seq_len: int, global_batch_size: int,
                        image_h: int, image_w: int, image_c: int,
                        shuffle_buffer_size: int = 1000, seed: int = 42):
    """
    Creates a tf.data.Dataset pipeline for loading video sequences.

    Args:
        data_dir_str: Path to the data directory.
        seq_len: Length of the video sequences.
        global_batch_size: Total batch size across all devices.
        image_h: Height of the image.
        image_w: Width of the image.
        image_c: Number of image channels.
        shuffle_buffer_size: Size of the buffer used for shuffling file paths.
        seed: Random seed for shuffling.

    Returns:
        An iterator that yields batches of video sequences as JAX arrays.
    """
    data_dir = Path(data_dir_str)
    metadata = np.load(data_dir / "metadata.npy", allow_pickle=True)
    
    episode_paths = [item["path"] for item in metadata]

    py_func_wrapper = functools.partial(
        _load_and_process_numpy,
        seq_len=seq_len,
    )

    def _tf_load_and_process(path_tensor):
        output_signature = tf.TensorSpec(shape=(seq_len, image_h, image_w, image_c), dtype=tf.float32)
        
        processed_sequence = tf.numpy_function(
            func=py_func_wrapper,
            inp=[path_tensor],
            Tout=output_signature
        )
        # FIXME: do we need this?
        processed_sequence.set_shape((seq_len, image_h, image_w, image_c))
        return processed_sequence

    dataset = tf.data.Dataset.from_tensor_slices(episode_paths)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(_tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache() # FIXME: check whether this fits
    # FIXME: Repeat the dataset for multiple epochs if necessary
    # dataset = dataset.repeat()
    dataset = dataset.batch(global_batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset.as_numpy_iterator()

    
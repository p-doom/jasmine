import jax
import numpy as np
import grain
from typing import Any, Optional
from array_record.python.array_record_module import ArrayRecordWriter
import tensorflow as tf
import os
from pathlib import Path
import pickle
import multiprocessing as mp
from functools import partial



def _convert_single_tfrecord(
    tfrecord_file: Path,
    output_folder: str,
    feature_description: dict,
) -> str:
    """
    Convert a single TFRecord file to ArrayRecord format.
    
    Args:
        tfrecord_file: Path to the TFRecord file
        output_folder: Output folder for the ArrayRecord file
        feature_description: Dictionary describing TFRecord features
    
    Returns:
        Path to the created ArrayRecord file
    """
    output_filename = tfrecord_file.stem + ".array_record"
    output_file = os.path.join(output_folder, output_filename)
    
    dataset = tf.data.TFRecordDataset(str(tfrecord_file))
    
    def parse_tfrecord(example_proto):
        """Parse a single TFRecord example."""
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        raw_video_bytes = parsed_features['raw_video'].numpy()
        sequence_length = int(parsed_features['sequence_length'].numpy())
        
        return {
            'raw_video': raw_video_bytes,
            'sequence_length': sequence_length,
        }
    
    record_count = 0
    writer = ArrayRecordWriter(output_file, "group_size:1")
    for record in dataset:
        parsed_record = parse_tfrecord(record)
        writer.write(pickle.dumps(parsed_record))
        record_count += 1
    writer.close()
    
    print(f"Converted {tfrecord_file.name} -> {output_filename}: {record_count} records")
    return output_file


def convert_tfrecords_to_arrayrecords(
    tfrecord_folder: str,
    output_folder: str,
    feature_description: Optional[dict] = None,
    num_workers: Optional[int] = None,
):
    """
    Converts TFRecord files to ArrayRecord format for use with Grain.
    Creates one ArrayRecord file per TFRecord file using multiprocessing.
    
    Args:
        tfrecord_folder: Path to folder containing TFRecord files
        output_folder: Path to output folder for ArrayRecord files
        feature_description: Dictionary describing TFRecord features. If None,
                           uses default description for video data.
        num_workers: Number of worker processes. If None, uses CPU count.
    
    Returns:
        List of paths to created ArrayRecord files
    """
    if feature_description is None:
        feature_description = {
            'raw_video': tf.io.FixedLenFeature([], tf.string),
            'sequence_length': tf.io.FixedLenFeature([], tf.int64),
        }
    
    os.makedirs(output_folder, exist_ok=True)
    
    tfrecord_files = list(Path(tfrecord_folder).glob("*.tfrecord"))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_folder}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(tfrecord_files))
    
    print(f"Using {num_workers} worker processes for conversion")
    
    convert_func = partial(
        _convert_single_tfrecord,
        output_folder=output_folder,
        feature_description=feature_description
    )
    
    with mp.Pool(processes=num_workers) as pool:
        arrayrecord_files = pool.map(convert_func, tfrecord_files)
    
    print(f"Conversion complete! Created {len(arrayrecord_files)} ArrayRecord files")
    return arrayrecord_files


class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """
    A Grain Transformation that combines parsing, slicing, and normalizing.
    """

    def __init__(self, seq_len: int, image_h: int, image_w: int, image_c: int):
        """Initializes the transformation with processing parameters."""
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

    def random_map(self, element: dict, rng: np.random.Generator) -> Any:
        """
        Processes a single raw episode from the data source.

        Args:
            element: A dictionary representing one record from the DataSource.
                     Expected to contain 'raw_video' (bytes) and 'sequence_length' (int)
            rng: A per-record random number generator provided by the Grain sampler.

        Returns:
            A processed video sequence as a NumPy array with shape
            (seq_len, height, width, channels) and dtype float32.
        """
        assert isinstance(element, bytes)
        element = pickle.loads(element)
        
        video_shape = (
            element["sequence_length"],
            self.image_h,
            self.image_w,
            self.image_c,
        )
        episode_tensor = np.frombuffer(element["raw_video"], dtype=np.uint8)
        episode_tensor = episode_tensor.reshape(video_shape)

        current_episode_len = episode_tensor.shape[0]
        if current_episode_len < self.seq_len:
             raise ValueError(f"An episode has length {current_episode_len}, which is "
                              f"shorter than the requested sequence length {self.seq_len}.")
        
        max_start_idx = current_episode_len - self.seq_len
        
        start_idx = rng.integers(0, max_start_idx + 1)

        seq = episode_tensor[start_idx : start_idx + self.seq_len]

        processed_sequence = seq.astype(np.float32) / 255.0

        return processed_sequence


def get_dataloader(
    array_record_paths: list[str],
    seq_len: int,
    global_batch_size: int,
    image_h: int,
    image_w: int,
    image_c: int,
    num_workers: int = 1,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
):
    """
    Creates a data loading pipeline using Grain.
    """
    if not array_record_paths:
        raise ValueError("array_record_paths list cannot be empty.")

    num_processes = jax.process_count()

    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes} for proper sharding."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = grain.sources.ArrayRecordDataSource(array_record_paths)
    
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        # FIXME: check whether the global shuffle is the reason why the dataloader is so slow
        shuffle=True,
        num_epochs=100, # FIXME: is there an equivalent to tf.data.repeat(None)?
        seed=seed,
    )

    operations = [
        ProcessEpisodeAndSlice(
            seq_len=seq_len, image_h=image_h, image_w=image_w, image_c=image_c
        ),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]

    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        # FIXME: `If the data is already loaded in memory, we recommend setting this to 0 to
        # avoid Python GIL contention by multiple threads.`
        num_threads=1,
    )
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        # FIXME: think about whether we should tune this
        worker_buffer_size=1,
        read_options=read_options,
    )

    return iter(dataloader)


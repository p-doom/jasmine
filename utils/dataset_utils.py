import os
import pickle
import multiprocessing as mp
import tensorflow as tf
import numpy as np
from typing import Optional
from array_record.python.array_record_module import ArrayRecordWriter, ArrayRecordReader
from pathlib import Path
from functools import partial

tf.config.experimental.set_visible_devices([], "GPU")


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
        raw_video_bytes = parsed_features["raw_video"].numpy()
        sequence_length = int(parsed_features["sequence_length"].numpy())

        return {
            "raw_video": raw_video_bytes,
            "sequence_length": sequence_length,
        }

    record_count = 0
    writer = ArrayRecordWriter(output_file, "group_size:1")
    for record in dataset:
        parsed_record = parse_tfrecord(record)
        writer.write(pickle.dumps(parsed_record))
        record_count += 1
    writer.close()

    print(
        f"Converted {tfrecord_file.name} -> {output_filename}: {record_count} records"
    )
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
            "raw_video": tf.io.FixedLenFeature([], tf.string),
            "sequence_length": tf.io.FixedLenFeature([], tf.int64),
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
        feature_description=feature_description,
    )

    with mp.Pool(processes=num_workers) as pool:
        arrayrecord_files = pool.map(convert_func, tfrecord_files)

    print(f"Conversion complete! Created {len(arrayrecord_files)} ArrayRecord files")
    return arrayrecord_files


def _reprocess_single_arrayrecord(
    arrayrecord_file: Path,
    output_folder: str,
    chunk_size: int,
    videos_per_file: int,
    image_h: int,
    image_w: int,
    image_c: int,
    file_index: int,
) -> list[str]:
    """
    Reprocess a single ArrayRecord file by splitting videos into chunks.

    Args:
        arrayrecord_file: Path to the ArrayRecord file
        output_folder: Output folder for the chunked files
        chunk_size: Number of frames per video chunk
        videos_per_file: Number of video chunks per output file
        image_h: Image height in pixels
        image_w: Image width in pixels
        image_c: Number of image channels
        file_index: Index for naming output files

    Returns:
        List of paths to created ArrayRecord files
    """
    print(f"Processing {arrayrecord_file.name}...")
    reader = ArrayRecordReader(str(arrayrecord_file))

    all_record_bytes = reader.read_all()
    file_chunks = []

    for record_bytes in all_record_bytes:
        record = pickle.loads(record_bytes)
        video_data = record["raw_video"]
        sequence_length = record["sequence_length"]

        video_tensor = np.frombuffer(video_data, dtype=np.uint8)

        assert (
            video_tensor.shape[0] == sequence_length * image_h * image_w * image_c
        ), f"Video tensor shape {video_tensor.shape} does not match expected shape {sequence_length * image_h * image_w * image_c}"
        video_tensor = video_tensor.reshape(sequence_length, image_h, image_w, image_c)

        current_episode_len = video_tensor.shape[0]
        if current_episode_len < chunk_size:
            print(
                f"Warning: Video has {current_episode_len} frames, skipping (need {chunk_size})"
            )
            continue

        for start_idx in range(0, current_episode_len - chunk_size + 1, chunk_size):
            chunk = video_tensor[start_idx : start_idx + chunk_size]

            # FIXME: currently no way of correlating the chunk with the original video
            chunk_record = {
                "raw_video": chunk.tobytes(),
                "sequence_length": chunk_size,
            }

            file_chunks.append(chunk_record)

    reader.close()

    # Write chunks to output files
    output_files = []
    for i in range(0, len(file_chunks), videos_per_file):
        batch_chunks = file_chunks[i : i + videos_per_file]
        output_filename = (
            f"chunked_videos_{file_index:04d}_{i//videos_per_file:04d}.array_record"
        )
        output_file = os.path.join(output_folder, output_filename)

        writer = ArrayRecordWriter(output_file, "group_size:1")
        for chunk in batch_chunks:
            writer.write(pickle.dumps(chunk))
        writer.close()

        output_files.append(output_file)
        print(f"Created {output_filename} with {len(batch_chunks)} video chunks")

    print(
        f"Processed {arrayrecord_file.name}: {len(file_chunks)} chunks -> {len(output_files)} files"
    )
    return output_files


def reprocess_arrayrecords_to_chunks(
    arrayrecord_folder: str,
    output_folder: str,
    chunk_size: int = 160,
    videos_per_file: int = 100,
    image_h: int = 90,
    image_w: int = 160,
    image_c: int = 3,
    num_workers: Optional[int] = None,
):
    """
    Reprocesses ArrayRecord files by splitting videos into chunks and creating new files.

    This function:
    1. Reads existing ArrayRecord files in parallel
    2. Splits each video into chunks of specified size
    3. Creates new ArrayRecord files with specified number of videos per file

    Args:
        arrayrecord_folder: Path to folder containing input ArrayRecord files
        output_folder: Path to output folder for new ArrayRecord files
        chunk_size: Number of frames per video chunk (default: 160)
        videos_per_file: Number of video chunks per output file (default: 100)
        image_h: Image height in pixels (default: 90)
        image_w: Image width in pixels (default: 160)
        image_c: Number of image channels (default: 3 for RGB)
        num_workers: Number of worker processes. If None, uses CPU count.

    Returns:
        List of paths to created ArrayRecord files
    """
    os.makedirs(output_folder, exist_ok=True)

    arrayrecord_files = list(Path(arrayrecord_folder).glob("*.array_record"))
    if not arrayrecord_files:
        raise ValueError(f"No ArrayRecord files found in {arrayrecord_folder}")

    print(f"Found {len(arrayrecord_files)} ArrayRecord files")

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(arrayrecord_files))

    print(f"Using {num_workers} worker processes for reprocessing")

    # Create a list of arguments for each file processing task
    process_args = []
    for i, file_path in enumerate(arrayrecord_files):
        process_args.append(
            (
                file_path,
                output_folder,
                chunk_size,
                videos_per_file,
                image_h,
                image_w,
                image_c,
                i,
            )
        )

    # Process files in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(_reprocess_single_arrayrecord, process_args)

    # Flatten the results
    all_output_files = []
    for result in results:
        all_output_files.extend(result)

    print(
        f"Reprocessing complete! Created {len(all_output_files)} new ArrayRecord files"
    )
    return all_output_files

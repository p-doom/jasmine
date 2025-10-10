import os
import pickle
import hashlib
import tyro
import numpy as np
import json
from collections import defaultdict
from array_record.python.array_record_module import ArrayRecordReader
import multiprocessing
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class Args:
    data_dir: str = "data/coinrun_episodes"
    output_dir: str = "data/jasmine_data/metadata_duplicates"


def hash_byte_data(bytes_data: bytes) -> str:
    """Calculates the SHA256 hash of a byte string."""
    return hashlib.sha256(bytes_data).hexdigest()


def hash_numpy_frame(frame: np.ndarray) -> str:
    """Calculates the SHA256 hash of a numpy array."""
    return hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()


def get_episode_level_hash(video_path: str) -> tuple[str, str] | None:
    """
    Reads a single array_record file, extracts the video, hashes it,
    and returns the (hash, video_path) tuple.
    """
    try:
        reader = ArrayRecordReader(video_path)
        record_data = reader.read()
        record_unpickled = pickle.loads(record_data)
        video_bytes = record_unpickled["raw_video"]
        video_hash = hash_byte_data(video_bytes)
        return video_hash, video_path
    except Exception as e:
        print(f"Error processing file {video_path}: {e}")
        return None


def get_frame_level_hashes(video_path: str) -> tuple[list[str], str] | None:
    """
    Reads a single array_record file, extracts the frames, hashes each frame,
    and returns the (list of hashes, video_path) tuple.
    """
    try:
        reader = ArrayRecordReader(video_path)
        record_data = pickle.loads(reader.read())

        # video shape (seq_len, 64, 64, 3)
        video_shape = (record_data["sequence_length"], 64, 64, 3)
        episode_tensor = np.frombuffer(record_data["raw_video"], dtype=np.uint8)
        episode_tensor = episode_tensor.reshape(video_shape)

        frame_hashes = [hash_numpy_frame(frame) for frame in episode_tensor]

        return frame_hashes, video_path
    except Exception as e:
        print(f"Error processing file {video_path}: {e}")
        return None


def get_array_record_files(dir):
    return [
        os.path.join(dir, x) for x in os.listdir(dir) if x.endswith(".array_record")
    ]


def main(args: Args):

    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    val_dir = os.path.join(args.data_dir, "val")

    array_record_files = get_array_record_files(args.data_dir)
    train_array_record_files = get_array_record_files(train_dir)
    test_array_record_files = get_array_record_files(test_dir)
    val_array_record_files = get_array_record_files(val_dir)

    array_record_files = (
        train_array_record_files + test_array_record_files + val_array_record_files
    )

    print(f"Detecting duplicates in: {args.data_dir}")
    print(f"Found {len(array_record_files)} files to process.")

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} worker processes.")

    # --- Episode level duplicates ---
    duplicate_episode = defaultdict(list)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(get_episode_level_hash, array_record_files)

        print("\nProcessing files and calculating hashes...")
        for result in tqdm(results, total=len(array_record_files)):
            if result:  # Ensure the worker didn't return None due to an error
                video_hash, video_path = result
                duplicate_episode[video_hash].append(video_path)

    print("\nAggregation complete. Finding duplicates...")
    duplicates = {h: paths for h, paths in duplicate_episode.items() if len(paths) > 1}

    total_episodes = len(array_record_files)
    num_duplicate_episodes = len(duplicates)
    percentage_duplicate_episodes = num_duplicate_episodes / len(array_record_files)

    print(f"Total episodes: {total_episodes}")
    print(f"Number of duplicate episodes: {num_duplicate_episodes}")
    print(f"Percentage of duplicate episodes: {percentage_duplicate_episodes:.2%}")

    print(f"Saving duplicates to episode_duplicates.json to {args.output_dir}...")
    with open(os.path.join(args.output_dir, "episode_duplicates.json"), "w") as f:
        json.dump(duplicates, f)

    # --- Frame level duplicates ---
    frame_dup_dict = defaultdict(list)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(get_frame_level_hashes, array_record_files)

        print("\nProcessing files and calculating frame hashes...")
        for result in tqdm(results, total=len(array_record_files)):
            if result:
                frame_hashes, video_path = result
                for frame_idx, frame_hash in enumerate(frame_hashes):
                    frame_dup_dict[frame_hash].append((video_path, frame_idx))

    print("\nAggregation complete. Finding duplicate frames...")
    duplicate_frames = {
        hash: location for hash, location in frame_dup_dict.items() if len(location) > 1
    }

    total_frames = sum(len(locations) for locations in frame_dup_dict.values())
    num_duplicate_frames = len(duplicate_frames)
    percentage = num_duplicate_frames / total_frames
    print(f"Total frames: {total_frames}")
    print(f"Number of duplicate frames: {num_duplicate_frames}")
    print(f"Percentage of duplicate frames: {percentage:.2%}")

    print(f"Saving duplicates to frame_duplicates.json to {args.output_dir}...")
    with open(os.path.join(args.output_dir, "frame_duplicates.json"), "w") as f:
        json.dump(duplicate_frames, f)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

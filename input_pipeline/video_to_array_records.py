import ffmpeg
import numpy as np
import os
import tyro
import multiprocessing as mp
from dataclasses import dataclass
import json
import pickle
from array_record.python.array_record_module import ArrayRecordWriter

"""
This file processes video files by converting them into array records.
It splits videos into chunks of a specified size and saves them in a specified output folder.
The script uses multiprocessing to handle multiple videos concurrently and generates metadata for the processed videos.
"""


@dataclass
class Args:
    input_path: str
    output_path: str
    env_name: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    target_width: int = 160
    target_height: int = 90
    target_fps: int = 10
    chunk_size: int = 160
    chunks_per_file: int = 100


def _chunk_and_save_video(
    video_tensor,
    video_file_name: str,
    output_folder: str,
    chunk_size: int,
    chunks_per_file: int,
    file_index: int,
) -> list[str]:
    """
    Reprocess a single ArrayRecord file by splitting videos into chunks.

    Args:
        video_file_name: Name of the video file 
        output_folder: Output folder for the chunked files
        chunk_size: Number of frames per video chunk
        chunks_per_file: Number of video chunks per output file
        file_index: Index for naming output files

    Returns:
        List of paths to created ArrayRecord files
    """
    file_chunks = []

    current_episode_len = video_tensor.shape[0]
    if current_episode_len < chunk_size:
        print(
            f"Warning: Video has {current_episode_len} frames, skipping (need {chunk_size})"
        )
        return [{"path": "", "length": 0, "video_file_name": video_file_name}]

    for start_idx in range(0, current_episode_len - chunk_size + 1, chunk_size):
        chunk = video_tensor[start_idx : start_idx + chunk_size]

        chunk_record = {
            "raw_video": chunk.tobytes(),
            "sequence_length": chunk_size,
            "video_file_name": video_file_name,
        }

        file_chunks.append(chunk_record)

    # Write chunks to output files
    output_files = []
    for i in range(0, len(file_chunks), chunks_per_file):
        batch_chunks = file_chunks[i : i + chunks_per_file]
        output_filename = (
            f"chunked_videos_{file_index:04d}_{i//chunks_per_file:04d}.array_record"
        )
        output_file = os.path.join(output_folder, output_filename)

        writer = ArrayRecordWriter(output_file, "group_size:1")
        for chunk in batch_chunks:
            writer.write(pickle.dumps(chunk))
        writer.close()

        output_files.append({"path": output_file, "length": chunk_size, "video_file_name": video_file_name})
        print(f"Created {output_filename} with {len(batch_chunks)} video chunks")

    print(
        f"Processed {video_file_name}: {len(file_chunks)} chunks -> {len(output_files)} files"
    )
    return output_files



def preprocess_video(
    idx, in_filename, output_path, target_width, target_height, target_fps, chunk_size, chunks_per_file
):
    """
    Preprocess a video file by reading it, resizing, changing its frame rate, 
    and then chunking it into smaller segments to be saved as ArrayRecord files.

    Args:
        idx (int): Index of the video being processed.
        in_filename (str): Path to the input video file.
        output_path (str): Directory where the output ArrayRecord files will be saved.
        target_width (int): The target width for resizing the video frames.
        target_height (int): The target height for resizing the video frames.
        target_fps (int): The target frames per second for the output video.
        chunk_size (int): Number of frames per chunk.
        chunks_per_file (int): Number of chunks to be saved in each ArrayRecord file.

    Returns:
        list: A list of dictionaries containing metadata about the created ArrayRecord files.
    """

    print(f"Processing video {idx}, Filename: {in_filename}")
    try:
        out, _ = (
            ffmpeg.input(in_filename)
            .filter("fps", fps=target_fps, round="up")
            .filter("scale", target_width, target_height)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
        )

        frame_size = target_height * target_width * 3
        n_frames = len(out) // frame_size
        frames = np.frombuffer(out, np.uint8).reshape(
            n_frames, target_height, target_width, 3
        )

        result = _chunk_and_save_video(video_tensor=frames,
                                        video_file_name=in_filename, 
                                        output_folder=output_path, 
                                        chunk_size=chunk_size, 
                                        chunks_per_file=chunks_per_file, 
                                        file_index=idx)
        return result
    except Exception as e:
        print(f"Error processing video {idx} ({in_filename}): {e}")
        return [{"path": "", "length": 0, "video_file_name": in_filename}]

def save_split(pool_args):
    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")
    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.starmap(preprocess_video, pool_args):
            results.extend(result)
    return results

def main():
    args = tyro.cli(Args)

    print(f"Output path: {args.output_path}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert np.isclose(total_ratio, 1.0), "Ratios must sum to 1.0"


    print("Converting video to array_record files...")
    input_files = [os.path.join(args.input_path, in_filename) 
                    for in_filename in os.listdir(args.input_path)
                    if in_filename.endswith(".mp4") or in_filename.endswith(".webm") 
                    ]
    n_total = len(input_files)
    n_train = round(n_total * args.train_ratio)
    n_val = round(n_total * args.val_ratio)

    np.random.shuffle(input_files)
    file_splits = {
        "train": input_files[:n_train],
        "val": input_files[n_train:n_train + n_val],
        "test": input_files[n_train + n_val :]
    }

    pool_args = dict()
    for split in file_splits.keys():
        pool_args[split] = []
        os.makedirs(os.path.join(args.output_path, split), exist_ok=True)
        for idx, in_filename in enumerate(file_splits[split]):
            pool_args[split].append((
                idx,
                in_filename,
                os.path.join(args.output_path, split),
                args.target_width,
                args.target_height,
                args.target_fps,
                args.chunk_size,
                args.chunks_per_file
            ))
        
    train_episode_metadata = save_split(pool_args["train"])
    val_episode_metadata = save_split(pool_args["val"])
    test_episode_metadata = save_split(pool_args["test"])

    print("Done converting video to array_record files")

    results = train_episode_metadata + val_episode_metadata + test_episode_metadata
    # count the number of short and failed videos
    failed_videos = [result for result in results if result["length"] == 0]
    num_successful_videos = len(results) - len(failed_videos)
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of successful videos: {num_successful_videos}")
    print(f"Number of total files: {len(input_files)}")
    print(f"Number of total chunks: {len(results)}")

    metadata = {
        "env": args.env_name,
        "total_chunks": len(results),
        "total_videos": len(input_files),
        "num_successful_videos": len(input_files) - len(failed_videos),
        "num_failed_videos": len(failed_videos),
        "avg_episode_len_train": np.mean([ep["length"] for ep in train_episode_metadata]),
        "avg_episode_len_val": np.mean([ep["length"] for ep in val_episode_metadata]),
        "avg_episode_len_test": np.mean([ep["length"] for ep in test_episode_metadata]),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()

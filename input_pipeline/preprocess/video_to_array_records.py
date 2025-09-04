import ffmpeg
import numpy as np
import os
import tyro
import multiprocessing as mp
from dataclasses import dataclass
import json
import pickle
from array_record.python.array_record_module import ArrayRecordWriter


@dataclass
class Args:
    input_path: str
    output_path: str
    env_name: str
    target_width: int = 160
    target_height: int = 90
    target_fps: int = 10


def preprocess_video(
    idx, in_filename, output_path, target_width, target_height, target_fps
):
    print(f"Processing video {idx}, Filename: {in_filename}")
    try:
        out, _ = (
            ffmpeg.input(in_filename)
            .filter("fps", fps=target_fps, round="up")
            .filter("scale", target_width, target_height)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, quiet=True)
        )

        output_path = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(in_filename))[0] + ".array_record",
        )

        writer = ArrayRecordWriter(str(output_path), "group_size:1")

        frame_size = target_height * target_width * 3
        n_frames = len(out) // frame_size
        frames = np.frombuffer(out, np.uint8).reshape(
            n_frames, target_height, target_width, 3
        )

        print(f"Saving video {idx} to {output_path}")
        record = {"raw_video": frames.tobytes(), "sequence_length": n_frames}
        writer.write(pickle.dumps(record))
        writer.close()

        return {"path": in_filename, "length": n_frames}
    except Exception as e:
        print(f"Error processing video {idx} ({in_filename}): {e}")
        return {"path": in_filename, "length": 0}


def main():
    args = tyro.cli(Args)

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")

    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")

    print("Converting video to array_record files...")
    pool_args = []
    for idx, in_filename in enumerate(os.listdir(args.input_path)):
        if in_filename.endswith(".mp4") or in_filename.endswith(".webm"):
            pool_args.append((
                idx,
                os.path.join(args.input_path, in_filename),
                args.output_path,
                args.target_width,
                args.target_height,
                args.target_fps,
            ))
        else:
            print(f"Warning: {in_filename} is not a supported video format. Skipping...")

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.starmap(preprocess_video, pool_args):
            results.append(result)
    print("Done converting video to array_record files")

    # count the number of failed videos
    failed_videos = [result for result in results if result["length"] == 0]
    short_videos = [result for result in results if result["length"] < 1600]
    num_successful_videos = len(results) - len(failed_videos) - len(short_videos)
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of short videos: {len(short_videos)}")
    print(f"Number of successful videos: {num_successful_videos}")
    print(f"Number of total videos: {len(results)}")

    metadata = {
        "env": args.env_name,
        "total_videos": len(results),
        "num_successful_videos": len(results) - len(failed_videos) - len(short_videos),
        "num_failed_videos": len(failed_videos),
        "num_short_videos": len(short_videos),
        "avg_episode_len": np.mean([ep["length"] for ep in results]),
        "episode_metadata": results,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()

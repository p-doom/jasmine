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
    target_width, target_height = 160, 90
    target_fps = 10
    input_path: str = "data/minecraft_videos"
    output_path: str = "data/minecraft_arrayrecords"


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

        return in_filename, n_frames
    except Exception as e:
        print(f"Error processing video {idx} ({in_filename}): {e}")
        return in_filename, 0


def main():
    args = tyro.cli(Args)

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")

    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")

    print("Converting video to array_record files...")
    pool_args = [
        (
            idx,
            os.path.join(args.input_path, in_filename),
            args.output_path,
            args.target_width,
            args.target_height,
            args.target_fps,
        )
        for idx, in_filename in enumerate(os.listdir(args.input_path))
        if in_filename.endswith(".mp4") or in_filename.endswith(".webm")
    ]

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.starmap(preprocess_video, pool_args):
            results.append(result)
    print("Done converting video to array_record files")

    # count the number of failed videos
    failed_videos = [result for result in results if result[1] == 0]
    short_episodes = [result for result in results if result[1] < 1600]
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of short episodes: {len(short_episodes)}")
    print(
        f"Number of successful videos: {len(results) - len(failed_videos) - len(short_episodes)}"
    )
    print(f"Number of total videos: {len(results)}")

    with open(os.path.join(args.output_path, "meta_data.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

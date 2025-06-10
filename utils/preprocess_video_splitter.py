import os
import ffmpeg
import tyro
from dataclasses import dataclass
import multiprocessing as mp


@dataclass
class Args:
    input_path: str = "/hkfs/work/workspace/scratch/tum_ind3695-jafar_ws/data/knoms"
    output_path: str = (
        "/hkfs/work/workspace/scratch/tum_ind3695-jafar_ws/data/knoms_clips"
    )
    clip_duration: int = 16


def split_video_ffmpeg_python(input_file_path, output_path, clip_duration):

    # TODO: this is purely for logging and sanity checking
    print(f"Splitting {input_file_path}...")
    filename, ext = os.path.splitext(os.path.basename(input_file_path))
    output_dir = f"{output_path}/{filename}_clips"
    os.makedirs(output_dir, exist_ok=True)

    # Probe video duration
    probe = ffmpeg.probe(input_file_path)
    duration = float(probe["format"]["duration"])

    start = 0
    while start < duration:
        # Calculate end time, but don't exceed video length
        end = min(start + clip_duration, duration)
        out_file = os.path.join(output_dir, f"{filename}_clip_{int(start):04d}{ext}")

        (
            ffmpeg.input(input_file_path, ss=start, t=(end - start))
            .output(out_file, c="copy")
            .overwrite_output()
            .run(quiet=True)
        )

        start += clip_duration

    print(f"Clips saved in '{output_dir}'")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # split_video_ffmpeg_python(args.input_path, args.output_path, args.clip_duration)

    files = [
        os.path.join(args.input_path, file)
        for file in os.listdir(args.input_path)
        if file.endswith(".mp4") or file.endswith(".webm")
    ]

    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")

    print(f"Splitting {len(files)} videos...")
    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(
            split_video_ffmpeg_python,
            [(file, args.output_path, args.clip_duration) for file in files],
        )

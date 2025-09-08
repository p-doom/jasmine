import os
import numpy as np
from PIL import Image
import tyro
from dataclasses import dataclass
import pickle
import json
import multiprocessing as mp
from array_record.python.array_record_module import ArrayRecordWriter


@dataclass
class Args:
    input_path: str
    output_path: str
    env_name: str
    original_fps: int = 60
    target_fps: int = 10
    target_width: int = 64 

def preprocess_pngs(input_dir, output_path, original_fps, target_fps, target_width=None):
    print(f"Processing PNGs in {input_dir}")
    try:
        png_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith('.png')
        ], key=lambda x: int(os.path.splitext(x)[0]))

        if not png_files:
            print(f"No PNG files found in {input_dir}")
            return {"path": input_dir, "length": 0} 

        # Downsample indices
        n_total = len(png_files)
        if original_fps == target_fps:
            selected_indices = np.arange(n_total)
        else:
            n_target = int(np.floor(n_total * target_fps / original_fps))
            selected_indices = np.linspace(0, n_total-1, n_target, dtype=int)

        selected_files = [png_files[i] for i in selected_indices]

        # Load images
        frames = []
        for fname in selected_files:
            img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            if target_width is not None:
                w, h = img.size  # PIL gives (width, height)
                if w != target_width:
                    target_height = int(round(h * (target_width / float(w))))
                    resample_filter = Image.LANCZOS
                    img = img.resize((target_width, target_height), resample=resample_filter)
            frames.append(np.array(img))

        frames = np.stack(frames, axis=0)  # (n_frames, H, W, 3)
        environment = os.path.basename(os.path.dirname(input_dir)) 
        episode_id = os.path.basename(input_dir)
        # Write to array_record
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(
            output_path,
            f"{environment}_{episode_id}.array_record"
        )
        writer = ArrayRecordWriter(str(out_file), "group_size:1")
        record = {"raw_video": frames.tobytes(), 
                  "environment": environment,
                  "sequence_length": frames.shape[0]}
        writer.write(pickle.dumps(record))
        writer.close()
        print(f"Saved {frames.shape[0]} frames to {out_file}")
        return {"path": input_dir, "length": frames.shape[0]}
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")
        return {"path": input_dir, "length": 0}

def main():
    args = tyro.cli(Args)
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")

    games = [
        os.path.join(args.input_path, d)
        for d in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, d))
    ]
    episodes = [
        os.path.join(game, d)
        for game in games
        for d in os.listdir(game)
    ]

    results = []
    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")
    pool_args = [
        (episode, args.output_path, args.original_fps, args.target_fps, args.target_width)
        for episode in episodes
    ]
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.starmap(preprocess_pngs, pool_args):
            results.append(result)

    print("Done converting png to array_record files")

    # count the number of failed videos
    failed_videos = [result for result in results if result["length"] == 0]
    num_successful_videos = len(results) - len(failed_videos)
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of successful videos: {num_successful_videos}")
    print(f"Number of total videos: {len(results)}")

    metadata = {
        "env": args.env_name,
        "total_videos": len(results),
        "num_successful_videos": len(results) - len(failed_videos),
        "num_failed_videos": len(failed_videos),
        "avg_episode_len": np.mean([ep["length"] for ep in results]),
        "episode_metadata": results,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Done.")

if __name__ == "__main__":
    main()
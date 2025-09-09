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
    multigame: bool = False
    original_fps: int = 60
    target_fps: int = 10
    target_width: int = 64
    chunk_size: int = 160
    chunks_per_file: int = 100

def _save_chunks(chunks, file_idx, chunks_per_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    while len(chunks) >= chunks_per_file:
        chunk_batch = chunks[:chunks_per_file]
        chunks = chunks[chunks_per_file:]
        episode_path = os.path.join(output_dir, f"data_{file_idx:04d}.array_record")  
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        seq_lens = []
        for chunk in chunk_batch:
            seq_len = chunk.shape[0]
            seq_lens.append(seq_len)
            chunk_record = {
                "raw_video": chunk.tobytes(),
                "sequence_length": seq_len,
            }
            writer.write(pickle.dumps(chunk_record))
        writer.close()
        file_idx += 1
        metadata.append({"path": episode_path, "num_chunks": len(chunk_batch), "avg_seq_len": np.mean(seq_lens)})
        print(f"Created {episode_path} with {len(chunk_batch)} video chunks")

    return metadata

def preprocess_pngs(input_dir, original_fps, target_fps, chunk_size, target_width):
    print(f"Processing PNGs in {input_dir}")
    try:
        png_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith('.png')
        ], key=lambda x: int(os.path.splitext(x)[0]))

        if not png_files:
            print(f"No PNG files found in {input_dir}")
            return [] 

        # Downsample indices
        n_total = len(png_files)
        if original_fps == target_fps:
            selected_indices = np.arange(n_total)
        else:
            n_target = int(np.floor(n_total * target_fps / original_fps))
            selected_indices = np.linspace(0, n_total-1, n_target, dtype=int)

        selected_files = [png_files[i] for i in selected_indices]

        # Load images
        chunks = []
        frames = []
        for fname in selected_files:
            img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            w, h = img.size  # PIL gives (width, height)
            if w != target_width:
                target_height = int(round(h * (target_width / float(w))))
                resample_filter = Image.LANCZOS
                img = img.resize((target_width, target_height), resample=resample_filter)
            frames.append(np.array(img))
            if len(frames) == chunk_size:
                chunks.append(frames)
                frames = []

        chunks = [np.stack(chunk, axis=0) for chunk in chunks]

        return chunks
    except Exception as e:
        print(f"Error processing {input_dir}: {e}")
        return []

def main():
    args = tyro.cli(Args)
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")

    directories = [
        os.path.join(args.input_path, d)
        for d in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, d))
    ]
    if args.multigame:
        episodes = [
            os.path.join(game, d)
            for game in directories 
            for d in os.listdir(game)
        ]
    else: 
        episodes = directories

    num_processes = mp.cpu_count()
    print(f"Number of processes: {num_processes}")
    pool_args = [
        (
            episode, 
            args.original_fps, 
            args.target_fps, 
            args.chunk_size, 
            args.target_width, 
        )
        for episode in episodes
    ]

    chunks = []
    file_idx = 0
    results = []
    for bucket_idx in range(0, len(pool_args), num_processes):
        args_batch = pool_args[bucket_idx : bucket_idx + num_processes]
        with mp.Pool(processes=num_processes) as pool:
            for episode_chunks in pool.starmap(preprocess_pngs, args_batch):
                chunks.extend(episode_chunks)
        results_batch = _save_chunks(chunks, file_idx, args.chunks_per_file, args.output_path) 
        results.extend(results_batch)

    print("Done converting png to array_record files")

    # count the number of failed videos
    failed_videos = [result for result in results if result["length"] == 0]
    num_successful_videos = len(results) - len(failed_videos)
    print(f"Number of failed videos: {len(failed_videos)}")
    print(f"Number of successful videos: {num_successful_videos}")
    print(f"Number of total videos: {len(results)}")

    metadata = {
        "env": args.env_name,
        "total_chunks": len(results),
        "avg_episode_len": np.mean([ep["avg_seq_len"] for ep in results]),
        "episode_metadata": results,
    }

    with open(os.path.join(args.output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("Done.")

if __name__ == "__main__":
    main()
"""
Generates a dataset of random-action CoinRun episodes.
Episodes are saved individually as memory-mapped files for efficient loading.
"""

from dataclasses import dataclass
from pathlib import Path

from gym3 import types_np
import numpy as np
from procgen import ProcgenGym3Env
import tyro
import pickle
import json
from array_record.python.array_record_module import ArrayRecordWriter 



@dataclass
class Args:
    num_episodes: int = 10000
    output_dir: str = "data/coinrun_episodes"
    min_episode_length: int = 1000
    max_episode_length: int = 1000
    chunk_size: int = 160
    chunks_per_file: int = 100


args = tyro.cli(Args)
assert args.max_episode_length >= args.min_episode_length, "Maximum episode length must be greater than or equal to minimum episode length."

if args.min_episode_length < args.chunk_size:
    print("Warning: Minimum episode length is smaller than chunk size. Note that episodes shorter than the chunk size will be discarded.")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

def _save_chunks(chunks, file_idx):
    ep_metadata = []
    while len(chunks) >= args.chunks_per_file:
        chunk_batch = chunks[:args.chunks_per_file]
        chunks = chunks[args.chunks_per_file:]
        episode_path = output_dir / f"coinrun_episodes_{file_idx:04d}.array_record"  
        # --- Save as ArrayRecord ---
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

        ep_metadata.append({"path": str(episode_path), "avg_seq_len": np.mean(seq_lens), "num_chunks": len(chunk_batch)})
        print(f"Created {episode_path} with {len(chunk_batch)} video chunks")
    return ep_metadata, chunks, file_idx

# --- Generate episodes ---
episode_idx = 0
episode_metadata = []
chunks = []
file_idx = 0
while episode_idx < args.num_episodes:
    seed = np.random.randint(0, 10000)
    env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=seed)
    
    observations_seq = []
    episode_chunks = []

    # --- Run episode ---
    for step_t in range(args.max_episode_length):
        env.act(types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        observations_seq.append(obs["rgb"])
        if len(observations_seq) == args.chunk_size:
            episode_chunks.append(observations_seq)
            observations_seq = []
        if first:
            break

    # --- Save episode ---
    if step_t + 1  >= args.min_episode_length:
        if len(observations_seq) < args.chunk_size:
            print(
                f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq)} frames, "
                f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                "This might lead to performance degradation during training."
            )
        episode_chunks.append(observations_seq)
        chunks_data = [np.concatenate(seq, axis=0).astype(np.uint8) for seq in episode_chunks]
        chunks.extend(chunks_data)

        ep_metadata, chunks, file_idx = _save_chunks(chunks, file_idx)
        episode_metadata.extend(ep_metadata)

        print(f"Episode {episode_idx} completed, length: {step_t + 1}.")
        episode_idx += 1
    else:
        print(f"Episode too short ({step_t + 1}), resampling...")
ep_metadata, chunks, file_idx = _save_chunks(chunks, file_idx)
episode_metadata.extend(ep_metadata)
# --- Save metadata ---
metadata = {
    "env": "coinrun",
    "num_episodes": args.num_episodes,
    "avg_episode_len": np.mean([ep["avg_seq_len"] for ep in episode_metadata]),
    "episode_metadata": episode_metadata,
}
with open(output_dir / "metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"Dataset generated with {args.num_episodes} episodes, saved over {len(episode_metadata)} files")

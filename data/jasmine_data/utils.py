import os
import pickle
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter


def save_chunks(file_idx, chunks_per_file, output_dir, obs_chunks, act_chunks=None):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    while len(obs_chunks) >= chunks_per_file:
        chunk_batch = obs_chunks[:chunks_per_file]
        obs_chunks = obs_chunks[chunks_per_file:]
        act_chunk_batch = None
        if act_chunks:
            act_chunk_batch = act_chunks[:chunks_per_file]
            act_chunks = act_chunks[chunks_per_file:]
        episode_path = os.path.join(output_dir, f"data_{file_idx:04d}.array_record")
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        seq_lens = []
        for idx, chunk in enumerate(chunk_batch):
            seq_len = chunk.shape[0]
            seq_lens.append(seq_len)
            chunk_record = {
                "raw_video": chunk.tobytes(),
                "sequence_length": seq_len,
            }
            if act_chunk_batch:
                assert len(chunk) == len(
                    act_chunk_batch[idx]
                ), f"Observation data length and action sequence length do not match: {len(chunk)} != {len(act_chunk_batch[idx])}"
                chunk_record["actions"] = act_chunk_batch[idx]
            writer.write(pickle.dumps(chunk_record))
        writer.close()
        file_idx += 1
        metadata.append(
            {
                "path": episode_path,
                "num_chunks": len(chunk_batch),
                "avg_seq_len": np.mean(seq_lens),
            }
        )
        print(f"Created {episode_path} with {len(chunk_batch)} video chunks")

    return metadata, file_idx, obs_chunks, act_chunks

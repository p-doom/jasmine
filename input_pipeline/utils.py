import os
import pickle
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

def save_chunks(chunks, file_idx, chunks_per_file, output_dir):
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

    return metadata, chunks, file_idx


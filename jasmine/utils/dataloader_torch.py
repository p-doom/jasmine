# file copied from https://raw.githubusercontent.com/FLAIROx/jafar/refs/heads/main/utils/dataloader.py
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, data_dir, seq_len):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.metadata = np.load(self.data_dir / "metadata.npy", allow_pickle=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        episode = np.load(self.metadata[idx]["path"])
        start_idx = np.random.randint(0, len(episode) - self.seq_len + 1)
        seq = episode[start_idx : start_idx + self.seq_len]
        return seq.astype(np.float32) / 255.0


def collate_fn(batch):
    """Convert batch of numpy arrays to JAX array"""
    return {"videos": jnp.array(np.stack(batch))}


def get_dataloader(data_dir, seq_len, batch_size):
    dataset = VideoDataset(data_dir, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

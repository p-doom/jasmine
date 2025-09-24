import jax
import numpy as np
import grain
from typing import Any
import pickle


class EpisodeLengthFilter(grain.transforms.Filter):
    """
    A Grain Filter that keeps only episodes with sufficient length.
    """

    def __init__(self, seq_len: int, image_h: int, image_w: int, image_c: int):
        """Initializes the filter with sequence length requirements."""
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

    def filter(self, element: Any) -> bool:
        """
        Filters episodes based on length.

        Args:
            element: A dictionary representing one record from the DataSource.
                     Expected to contain 'raw_video' (bytes) and 'sequence_length' (int)

        Returns:
            True if the episode has sufficient length, False otherwise.
        """
        assert isinstance(element, bytes)
        element = pickle.loads(element)

        current_episode_len = element["sequence_length"]
        if current_episode_len < self.seq_len:
            print(
                f"Filtering out episode with length {current_episode_len}, which is "
                f"shorter than the requested sequence length {self.seq_len}."
            )
            return False

        return True


class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """
    A Grain Transformation that combines parsing, slicing, and normalizing.
    """

    def __init__(self, seq_len: int, image_h: int, image_w: int, image_c: int):
        """Initializes the transformation with processing parameters."""
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

    def random_map(self, element: dict, rng: np.random.Generator) -> Any:
        """
        Processes a single raw episode from the data source.

        Args:
            element: A dictionary representing one record from the DataSource.
                     Expected to contain 'raw_video' (bytes) and 'sequence_length' (int)
            rng: A per-record random number generator provided by the Grain sampler.

        Returns:
            A processed video sequence as a NumPy array with shape
            (seq_len, height, width, channels) and dtype float32.
        """
        assert isinstance(element, bytes)
        element = pickle.loads(element)

        video_shape = (
            element["sequence_length"],
            self.image_h,
            self.image_w,
            self.image_c,
        )
        episode_tensor = np.frombuffer(element["raw_video"], dtype=np.uint8)
        episode_tensor = episode_tensor.reshape(video_shape)

        current_episode_len = episode_tensor.shape[0]
        if current_episode_len < self.seq_len:
            raise ValueError(
                f"Episode length {current_episode_len} is shorter than "
                f"requested sequence length {self.seq_len}. This should "
                f"have been filtered out."
            )

        max_start_idx = current_episode_len - self.seq_len

        start_idx = rng.integers(0, max_start_idx + 1)

        seq = episode_tensor[start_idx : start_idx + self.seq_len]

        data_dict = {"videos": seq}
        if "actions" in element.keys():
            actions_tensor = np.array(element["actions"])
            actions = actions_tensor[start_idx : start_idx + self.seq_len]
            data_dict["actions"] = actions

        return data_dict


def get_dataloader(
    array_record_paths: list[str],
    seq_len: int,
    global_batch_size: int,
    image_h: int,
    image_w: int,
    image_c: int,
    num_workers: int = 1,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
):
    """
    Creates a data loading pipeline using Grain.
    """
    if not array_record_paths:
        raise ValueError("array_record_paths list cannot be empty.")

    num_processes = jax.process_count()

    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes} for proper sharding."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = grain.sources.ArrayRecordDataSource(array_record_paths)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        EpisodeLengthFilter(
            seq_len=seq_len, image_h=image_h, image_w=image_w, image_c=image_c
        ),
        ProcessEpisodeAndSlice(
            seq_len=seq_len, image_h=image_h, image_w=image_w, image_c=image_c
        ),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]

    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=1,
    )
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=1,
        read_options=read_options,
    )

    return dataloader

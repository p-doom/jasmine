import unittest
import numpy as np
import tensorflow as tf
import tempfile
from pathlib import Path

from utils.dataloader import get_dataloader
from tests.data.generate_dummy_tfrecord import generate_dummy_tfrecord


class DataloaderReproducibilityTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._temp_dir_manager = tempfile.TemporaryDirectory()
        self.test_data_dir = Path(self._temp_dir_manager.name)
        self.addCleanup(self._temp_dir_manager.cleanup)
        self.dummy_tfrecord_path = self.test_data_dir / "dummy_test_shard.tfrecord"

        self.num_episodes = 5
        self.episode_length = 16
        self.image_height = 64
        self.image_width = 64
        self.image_channels = 3
        generate_dummy_tfrecord(
            self.dummy_tfrecord_path,
            num_episodes=self.num_episodes,
            episode_length=self.episode_length,
            height=self.image_height,
            width=self.image_width,
            channels=self.image_channels,
        )
        self.tfrecord_files = [str(self.dummy_tfrecord_path)]

        self.fixed_seed = 42

    def test_dataloader_yields_reproducible_batches(self):
        seq_len = 8
        batch_size = 2

        dataloader1 = get_dataloader(
            self.tfrecord_files,
            seq_len,
            batch_size,
            self.image_height,
            self.image_width,
            self.image_channels,
            seed=self.fixed_seed,
        )
        batches1 = [next(dataloader1) for _ in range(3)]

        dataloader2 = get_dataloader(
            self.tfrecord_files,
            seq_len,
            batch_size,
            self.image_height,
            self.image_width,
            self.image_channels,
            seed=self.fixed_seed,
        )
        batches2 = [next(dataloader2) for _ in range(3)]

        for i, (b1, b2) in enumerate(zip(batches1, batches2)):
            np.testing.assert_array_equal(b1, b2, err_msg=f"Batch {i} is not reproducible")  # type: ignore


if __name__ == "__main__":
    unittest.main()
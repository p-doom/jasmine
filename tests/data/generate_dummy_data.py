import tyro
import tensorflow as tf
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Args:
    data_dir: str = "data_tfrecords/dummy"
    num_episodes: int = 5
    episode_length: int = 16



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tfrecord_example(episode_numpy_array):
    """Creates a TFRecord example from a numpy array video."""
    feature = {
        "height": _int64_feature(episode_numpy_array.shape[1]),
        "width": _int64_feature(episode_numpy_array.shape[2]),
        "channels": _int64_feature(episode_numpy_array.shape[3]),
        "sequence_length": _int64_feature(episode_numpy_array.shape[0]),
        "raw_video": _bytes_feature(episode_numpy_array.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_dummy_tfrecord(
    output_path, num_episodes=5, episode_length=16, height=90, width=160, channels=3
):
    """Generates a dummy TFRecord file with synthetic video data."""
    print(f"Generating dummy TFRecord file at {output_path}")
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for i in range(num_episodes):
            np.random.seed(i)  # Seed per episode for some variation, but deterministic
            dummy_video = np.random.randint(
                0, 256, size=(episode_length, height, width, channels), dtype=np.uint8
            )
            tf_example = create_tfrecord_example(dummy_video)
            writer.write(tf_example.SerializeToString())
    print("Dummy TFRecord generation complete.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    temp_dir = Path(args.data_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = temp_dir / "dummy_test_shard.tfrecord"
    generate_dummy_tfrecord(dummy_file, num_episodes=args.num_episodes, episode_length=args.episode_length)
    print(f"Generated dummy file: {dummy_file}")
import pickle
import numpy as np
from pathlib import Path
from array_record.python.array_record_module import ArrayRecordWriter


def generate_dummy_arrayrecord(
    output_path: Path,
    num_videos: int = 5,
    episode_length: int = 16,
    height: int = 64,
    width: int = 64,
    channels: int = 3,
    action_vector_dim: int = 11,  # default dimension of the action token vector from minerl
    vocab_size: int = 68,  # default vocab size from minerl
    seed: int = 42,
):
    """Generates a dummy ArrayRecord file with synthetic video data for testing.

    Dimension keys:
        E: episode length
        H: height
        W: width
        C: channels
        V: vocab size
        A: action vector dimension
    """

    print(f"Generating dummy ArrayRecord file at {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = ArrayRecordWriter(str(output_path), "group_size:1")
    try:
        for i in range(num_videos):
            np.random.seed(seed + i)
            dummy_video_EHWC = np.random.randint(
                0, 256, size=(episode_length, height, width, channels), dtype=np.uint8
            )

            actions_VEA = np.random.randint(
                0,
                vocab_size,
                size=(episode_length, action_vector_dim),
                dtype=np.uint8,
            )

            record = {
                "raw_video": dummy_video_EHWC.tobytes(),
                "sequence_length": episode_length,
                "actions": actions_VEA,
            }

            writer.write(pickle.dumps(record))
    finally:
        writer.close()

    print("Dummy ArrayRecord generation complete.")


if __name__ == "__main__":
    test_dir = Path("tests/data/dummy_arrayrecord")
    test_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = test_dir / "dummy_test_shard.array_record"

    generate_dummy_arrayrecord(
        dummy_file, num_videos=5, episode_length=16, height=64, width=64, channels=3
    )
    print(f"Generated dummy file: {dummy_file}")

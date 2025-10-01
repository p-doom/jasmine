import os
import math
import argparse
import pickle
from typing import Optional, List

import numpy as np

from PIL import Image, ImageDraw

from array_record.python.array_record_module import ArrayRecordReader


def infer_hw_from_bytes(
    total_elements: int, seq_len: int, channels: int
) -> tuple[int, int]:
    assert seq_len > 0 and channels > 0, "sequence_length and channels must be positive"
    base = total_elements // (seq_len * channels)
    side = int(math.isqrt(base))
    if side * side != base:
        raise ValueError(
            f"Could not infer square HxW from buffer. elements={total_elements}, seq_len={seq_len}, channels={channels}"
        )
    return side, side


def load_one_record(input_path: str) -> dict:
    assert (
        ArrayRecordReader is not None
    ), "array_record is not installed. pip install array-record"
    reader = ArrayRecordReader(str(input_path))

    raw = reader.read()
    if raw is None:
        raise RuntimeError(f"No record could be read from {input_path}")

    return pickle.loads(raw)


def get_action_meanings(env_id: Optional[str]) -> Optional[List[str]]:
    if env_id is None:
        return None
    try:
        import gymnasium as gym
        import ale_py

        gym.register_envs(ale_py)
        env = gym.make(env_id)
        try:
            meanings = list(env.unwrapped.get_action_meanings())  # type: ignore[attr-defined]
        finally:
            env.close()
        return meanings
    except Exception:
        print(f"Error getting action meanings for {env_id}")
        return None


def save_frames_with_actions(
    record: dict,
    output_dir: str,
    channels: int,
    height: Optional[int],
    width: Optional[int],
    action_meanings: Optional[List[str]],
    fps: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    seq_len: int = int(record["sequence_length"])
    raw_video: bytes = record["raw_video"]
    actions: Optional[np.ndarray] = record.get("actions")
    if actions is not None:
        actions = np.asarray(actions).reshape(-1)

    # Decode frames
    arr = np.frombuffer(raw_video, dtype=np.uint8)
    total_elements = arr.size

    if height is not None and width is not None:
        h, w = int(height), int(width)
        expected = seq_len * h * w * channels
        assert (
            expected == total_elements
        ), f"Expected {expected} elements, got {total_elements}"
    else:
        h, w = infer_hw_from_bytes(total_elements, seq_len, channels)

    frames = arr.reshape(seq_len, h, w, channels)

    if actions is not None:
        assert (
            actions.shape[0] == seq_len
        ), f"Expected {seq_len} actions, got {actions.shape[0]}"

    # Save an actions.txt index for quick inspection
    if actions is not None:
        with open(os.path.join(output_dir, "actions.txt"), "w") as f:
            for t in range(seq_len):
                a = int(actions[t])
                name = (
                    action_meanings[a]
                    if action_meanings is not None and 0 <= a < len(action_meanings)
                    else None
                )
                f.write(f"{t}\t{a}" + (f"\t{name}" if name is not None else "") + "\n")

    # Build frames with overlays and save as GIF
    duration_ms = max(1, int(1000 / max(1, fps)))
    imgs: List[Image.Image] = []
    for t in range(seq_len):
        img_np_rgb = frames[t]
        img = Image.fromarray(img_np_rgb, mode="RGB")
        draw = ImageDraw.Draw(img)
        if actions is not None:
            a = int(actions[t])
            name = (
                action_meanings[a]
                if action_meanings is not None and 0 <= a < len(action_meanings)
                else None
            )
            text = f"{a}" if name is None else f"{a} {name}"
        else:
            text = "?"
        draw.text((2, 2), text, fill=(255, 255, 255))
        imgs.append(img)

    if not imgs:
        raise RuntimeError("No frames to render.")

    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "sequence.gif")

    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
    )

    print(
        f"Saved GIF to {gif_path} with {seq_len} frames (H={h}, W={w}, C={channels}, fps={fps})."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a random sequence from an ArrayRecord by saving frames with actions."
    )
    parser.add_argument("--input", required=True, help="Path to .array_record file")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output images and actions.txt",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of channels in frames (default 3 for RGB)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=84,
        help="Frame height (if mismatch, will be inferred)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=84,
        help="Frame width (if mismatch, will be inferred)",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default=None,
        help="Gymnasium env id to map action indices to names",
    )
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second")

    args = parser.parse_args()

    assert args.channels == 3, "Only 3 channels are currently supported"

    record = load_one_record(args.input)

    # Print quick summary
    seq_len = int(record.get("sequence_length", -1))
    has_actions = "actions" in record and record["actions"] is not None
    print(
        f"Loaded record with sequence_length={seq_len}, actions_present={has_actions}"
    )

    action_meanings = get_action_meanings(args.env_id)

    save_frames_with_actions(
        record=record,
        output_dir=args.output_dir,
        channels=int(args.channels),
        height=int(args.height) if args.height else None,
        width=int(args.width) if args.width else None,
        action_meanings=action_meanings,
        fps=int(args.fps),
    )


if __name__ == "__main__":
    main()

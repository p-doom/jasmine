"""
Generates a dataset of random-action ViZDoom episodes.
Episodes are saved individually as memory-mapped files (via jasmine_data.utils.save_chunks)
for efficient loading (same save convention as the CoinRun example you provided).

Notes / assumptions:
- Uses a ViZDoom .cfg scenario file. By default it will try to use the bundled
  `basic.cfg` from the installed `vizdoom` package (`vizdoom.scenarios_path`).
- Observations are the raw RGB `screen_buffer` from ViZDoom (converted to HWC uint8).
- Actions are binary vectors of length `num_buttons` (0/1 per button) stored as uint8.

Make sure you have `vizdoom` and `jasmine_data` installed in your environment.
"""

from dataclasses import dataclass
import os
import json
import numpy as np
import tyro
import vizdoom as vzd
from vizdoom import DoomGame, ScreenResolution, ScreenFormat, Mode
from jasmine_data.utils import save_chunks


@dataclass
class Args:
    num_episodes_train: int = 1000
    num_episodes_val: int = 100
    num_episodes_test: int = 100
    output_dir: str = "data/vizdoom_episodes"
    scenario_cfg: str = (
        ""  # path to .cfg; if empty, use vizdoom.scenarios_path/basic.cfg
    )
    min_episode_length: int = 100
    max_episode_length: int = 1000
    chunk_size: int = 160
    chunks_per_file: int = 100
    seed: int = 0
    screen_resolution: ScreenResolution = ScreenResolution.RES_160X120
    screen_format: ScreenFormat = ScreenFormat.RGB24
    window_visible: bool = False


args = tyro.cli(Args)
assert (
    args.max_episode_length >= args.min_episode_length
), "Maximum episode length must be greater than or equal to minimum episode length."

if args.min_episode_length < args.chunk_size:
    print(
        "Warning: Minimum episode length is smaller than chunk size. Note that episodes shorter than the chunk size will be discarded."
    )


# --- Helpers for ViZDoom ---
def make_game(cfg_path: str, seed: int):
    """Create and initialize a DoomGame from cfg_path. Returns an initialized DoomGame.
    We intentionally load and init a fresh game per call (simple and robust).
    """
    game = DoomGame()
    # If cfg_path is empty, use the bundled basic.cfg from the vizdoom installation
    if not cfg_path:
        cfg_path = os.path.join(vzd.scenarios_path, "basic.cfg")
    game.load_config(cfg_path)

    # override screen buffers if desired (these must be set before init)
    game.set_screen_resolution(args.screen_resolution)
    game.set_screen_format(args.screen_format)
    game.set_window_visible(args.window_visible)
    game.set_mode(Mode.PLAYER)

    # Seed controls determinism of the episode generation
    game.set_seed(int(seed))

    game.init()
    return game


# --- Generate episodes ---
def generate_episodes(num_episodes, split):
    episode_idx = 0
    episode_metadata = []
    obs_chunks = []
    act_chunks = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    os.makedirs(output_dir_split, exist_ok=True)

    while episode_idx < num_episodes:
        seed = np.random.randint(0, 2**31 - 1)
        game = make_game(args.scenario_cfg, seed)

        observations_seq = []
        actions_seq = []
        episode_obs_chunks = []
        episode_act_chunks = []

        # --- Run episode ---
        step_t = 0
        game.new_episode()  # ensure a fresh episode

        num_buttons = game.get_available_buttons_size()

        for step_t in range(args.max_episode_length):
            if game.is_episode_finished():
                break

            state = game.get_state()
            if state is None or state.screen_buffer is None:
                # no more frames
                break

            # VizDoom returns screen_buffer in CHW uint8 (channels, H, W)
            # convert to HWC for common ML pipelines
            frame_hwc = state.screen_buffer
            # append as a 1-frame batch so concatenation later matches coinrun style
            observations_seq.append(np.expand_dims(frame_hwc, axis=0).astype(np.uint8))

            # sample a random binary action vector (0/1 per available button)
            action = np.random.randint(0, 2, size=(num_buttons,), dtype=np.uint8)
            action_idx = action.dot(1 << np.arange(action.shape[-1] - 1, -1, -1))
            actions_seq.append(np.expand_dims(action_idx, axis=0))

            # perform the action in the game
            # make_action accepts a list of ints/bools
            game.make_action(action.tolist())

            # chunking
            if len(observations_seq) == args.chunk_size:
                episode_obs_chunks.append(observations_seq)
                episode_act_chunks.append(actions_seq)
                observations_seq = []
                actions_seq = []

        # --- Save episode ---
        ep_len = step_t + 1 if step_t is not None else 0
        if ep_len >= args.min_episode_length:
            # if there is an unfinished partial chunk, keep it (but warn)
            if observations_seq:
                if len(observations_seq) < args.chunk_size:
                    print(
                        f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq)} frames, "
                        f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                        "This might lead to performance degradation during training."
                    )
                episode_obs_chunks.append(observations_seq)
                episode_act_chunks.append(actions_seq)

            obs_chunks_data = [
                np.concatenate(seq, axis=0).astype(np.uint8)
                for seq in episode_obs_chunks
            ]
            act_chunks_data = [
                np.concatenate(act, axis=0).astype(np.uint8)
                for act in episode_act_chunks
            ]
            obs_chunks.extend(obs_chunks_data)
            act_chunks.extend(act_chunks_data)

            ep_metadata, file_idx, obs_chunks, act_chunks = save_chunks(
                file_idx, args.chunks_per_file, output_dir_split, obs_chunks, act_chunks
            )
            episode_metadata.extend(ep_metadata)

            print(f"Episode {episode_idx} completed, length: {ep_len}.")
            episode_idx += 1
        else:
            print(f"Episode too short ({ep_len}), resampling...")

        # cleanup
        try:
            game.close()
        except Exception:
            pass

    if len(obs_chunks) > 0:
        print(
            f"Warning: Dropping {len(obs_chunks)} chunks for consistent number of chunks per file.",
            "Consider changing the chunk_size and chunks_per_file parameters to prevent data-loss.",
        )

    print(f"Done generating {split} split")
    return episode_metadata


def get_action_space():
    # inspect sample game to get button-count
    sample_game = make_game(args.scenario_cfg, seed=0)
    num_buttons = sample_game.get_available_buttons_size()
    sample_game.close()
    return 2**num_buttons


def main():
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_episode_metadata = generate_episodes(args.num_episodes_train, "train")
    val_episode_metadata = generate_episodes(args.num_episodes_val, "val")
    test_episode_metadata = generate_episodes(args.num_episodes_test, "test")

    metadata = {
        "env": "vizdoom",
        "scenario_cfg": (
            args.scenario_cfg
            if args.scenario_cfg
            else os.path.join(vzd.scenarios_path, "basic.cfg")
        ),
        "screen_resolution": str(args.screen_resolution),
        "screen_format": str(args.screen_format),
        "num_buttons": get_action_space(),
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": (
            float(np.mean([ep["avg_seq_len"] for ep in train_episode_metadata]))
            if train_episode_metadata
            else 0.0
        ),
        "avg_episode_len_val": (
            float(np.mean([ep["avg_seq_len"] for ep in val_episode_metadata]))
            if val_episode_metadata
            else 0.0
        ),
        "avg_episode_len_test": (
            float(np.mean([ep["avg_seq_len"] for ep in test_episode_metadata]))
            if test_episode_metadata
            else 0.0
        ),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done generating dataset.")


if __name__ == "__main__":
    main()

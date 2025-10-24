# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   /$$    /$$ /$$           /$$$$$$$                                          /$$$$$$$  /$$$$$$$   /$$$$$$     #
#  | $$   | $$|__/          | $$__  $$                                        | $$__  $$| $$__  $$ /$$__  $$    #
#  | $$   | $$ /$$ /$$$$$$$$| $$  \ $$  /$$$$$$   /$$$$$$  /$$$$$$/$$$$       | $$  \ $$| $$  \ $$| $$  \ $$    #
#  |  $$ / $$/| $$|____ /$$/| $$  | $$ /$$__  $$ /$$__  $$| $$_  $$_  $$      | $$$$$$$/| $$$$$$$/| $$  | $$    #
#   \  $$ $$/ | $$   /$$$$/ | $$  | $$| $$  \ $$| $$  \ $$| $$ \ $$ \ $$      | $$____/ | $$____/ | $$  | $$    #
#    \  $$$/  | $$  /$$__/  | $$  | $$| $$  | $$| $$  | $$| $$ | $$ | $$      | $$      | $$      | $$  | $$    #
#     \  $/   | $$ /$$$$$$$$| $$$$$$$/|  $$$$$$/|  $$$$$$/| $$ | $$ | $$      | $$      | $$      |  $$$$$$/    #
#      \_/    |__/|________/|_______/  \______/  \______/ |__/ |__/ |__/      |__/      |__/       \______/     #
#                                                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# FORK OF LEANDRO KIELIGER'S DOOM PPO TUTORIAL: https://lkieliger.medium.com/deep-reinforcement-learning-in-practice-by-playing-doom-part-1-getting-started-618c99075c77

# SCRIPT TO RUN PPO AGENT AND GENERATE DATASET FOR DOOM ENVIRONMENT.

from dataclasses import dataclass
import imageio
from common import envs
import torch
import json
from vizdoom.vizdoom import GameVariable
import os
from PIL import Image

import numpy as np
from train_ppo_parallel import DoomWithBotsCurriculum, game_instance
from stable_baselines3.common.vec_env import (
    VecTransposeImage,
    DummyVecEnv,
    SubprocVecEnv,
)

from loguru import logger
import tyro
from jasmine_data.utils import save_chunks

# To replicate frame_skip in the environment
ACTION_REPEAT = 4


@dataclass
class Args:
    num_episodes_train: int = 1000
    num_episodes_val: int = 100
    num_episodes_test: int = 100
    min_episode_length: int = 100
    max_episode_length: int = 1000
    num_parallel_envs: int = 100
    target_width: int = 320
    target_height: int = 240
    chunk_size: int = 160
    chunks_per_file: int = 100
    agent_path: str = ""
    seed: int = 0
    output_dir: str = "data/vizdoom_episodes"
    generate_gif: bool = False


args = tyro.cli(Args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def dummy_vec_env_with_bots_curriculum(n_envs=1, **kwargs) -> VecTransposeImage:
    """Wraps a Doom game instance in a vectorized environment with shaped rewards and curriculum."""
    scenario = kwargs.pop("scenario")  # Remove 'scenario' from kwargs
    return VecTransposeImage(
        DummyVecEnv(
            [lambda: DoomWithBotsCurriculum(game_instance(scenario), **kwargs)] * n_envs
        )
    )


# TODO move to utils
def downsample_resolution(img):
    if img.shape[:2] != (args.target_height, args.target_width):
        resample_filter = Image.LANCZOS
        img = Image.fromarray(img)
        img = img.resize(
            (args.target_width, args.target_height), resample=resample_filter
        )
        img = np.array(img)
    return img


def make_gif(agent, eval_env_args):
    """Generate a GIF by running the agent in the environment.

    Args:
        agent: The trained PPO agent.
        file_path (str): Path to save the generated GIF.
        eval_env_args (dict): Arguments for the evaluation environment.
        num_episodes (int): Number of episodes to run.

    Returns:
        list: Collected health values for analysis.
    """
    # Set frame_skip to 1 to capture all frames
    eval_env_args["frame_skip"] = 1
    env = dummy_vec_env_with_bots_curriculum(1, **eval_env_args)

    images = []
    actions = []
    health_values = []
    current_action = None
    frame_counter = 0

    obs = env.reset()

    done = False
    while not done and frame_counter < args.max_episode_length:
        if frame_counter % ACTION_REPEAT == 0:
            current_action, _ = agent.predict(obs)

        obs, _, done, _ = env.step(current_action)

        # Get the raw screen buffer from the Doom game instance
        screen = env.venv.envs[0].game.get_state().screen_buffer
        screen = downsample_resolution(screen)

        # Get the current health value
        health = env.venv.envs[0].game.get_game_variable(GameVariable.HEALTH)
        health_values.append(health)  # Store the health value

        actions.append(current_action)
        images.append(screen)

        frame_counter += 1

    print("Health values:", health_values)
    print("Number of health values:", len(health_values))
    print("Number of actions:", len(actions))
    print("Number of images:", len(images))

    # Save only the first 1000 frames to avoid large file size
    output_path = os.path.join(args.output_dir, "output.gif")
    imageio.mimsave(output_path, images, fps=20)
    env.close()
    logger.info(f"GIF saved to {args.output_dir}")

    return health_values


def make_array_records_dataset(agent, eval_env_args, num_episodes, split):
    """Generate a dataset by running the agent in the environment and saving the data as array record files.

    Args:
        agent: The trained PPO agent.
        output_dir (str): Directory to save the array record files.
        eval_env_args (dict): Arguments for the evaluation environment.
        num_episodes (int): Number of episodes to run.
    """
    # Set frame_skip to 1 to capture all frames
    eval_env_args["frame_skip"] = 1
    env = dummy_vec_env_with_bots_curriculum(args.num_parallel_envs, **eval_env_args)

    current_action_B = None
    episode_idx = 0
    episode_metadata = []
    obs_chunks = []
    act_chunks = []
    file_idx = 0
    output_dir_split = os.path.join(args.output_dir, split)
    os.makedirs(output_dir_split, exist_ok=True)
    env.venv.render_mode = "rgb_array"

    while episode_idx < num_episodes // args.num_parallel_envs:
        obs = env.reset()
        done = np.array(False)
        frame_counter = 0

        observations_seq_TBHWC = []
        actions_seq_TB = []
        health_values_seq_TB = []
        episode_obs_chunks_NTBHWC = []
        episode_act_chunks_NTB = []

        # --- Run episode ---
        while not done.any() and frame_counter < args.max_episode_length:
            screen_BHWC = [
                downsample_resolution(env_i.game.get_state().screen_buffer)
                for env_i in env.venv.envs
            ]
            health_B = [
                env_i.game.get_game_variable(GameVariable.HEALTH)
                for env_i in env.venv.envs
            ]
            if frame_counter % ACTION_REPEAT == 0:
                current_action_B, _ = agent.predict(obs)

            obs, _, done, _ = env.step(current_action_B)

            observations_seq_TBHWC.append(screen_BHWC)
            actions_seq_TB.append(current_action_B)
            health_values_seq_TB.append(health_B)

            while len(observations_seq_TBHWC) >= args.chunk_size:
                episode_obs_chunks_NTBHWC.append(
                    observations_seq_TBHWC[: args.chunk_size]
                )
                episode_act_chunks_NTB.append(actions_seq_TB[: args.chunk_size])
                observations_seq_TBHWC = observations_seq_TBHWC[args.chunk_size :]
                actions_seq_TB = actions_seq_TB[args.chunk_size :]

            frame_counter += 1

        # --- Save episode ---
        if frame_counter >= args.min_episode_length:
            if observations_seq_TBHWC:
                if len(observations_seq_TBHWC) < args.chunk_size:
                    print(
                        f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq_TBHWC)} frames, "
                        f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                        "This might lead to performance degradation during training."
                    )
                episode_obs_chunks_NTBHWC.append(observations_seq_TBHWC)
                episode_act_chunks_NTB.append(actions_seq_TB)
            episode_obs_chunks_NBTHWC = [
                np.transpose(seq, (1, 0, 2, 3, 4)).astype(np.uint8)
                for seq in episode_obs_chunks_NTBHWC
            ]
            obs_chunks_data = [
                chunk for batch in episode_obs_chunks_NBTHWC for chunk in batch
            ]
            episode_act_chunks_NBT = [
                np.transpose(seq).astype(np.uint8) for seq in episode_act_chunks_NTB
            ]
            act_chunks_data = [
                chunk for batch in episode_act_chunks_NBT for chunk in batch
            ]
            obs_chunks.extend(obs_chunks_data)
            act_chunks.extend(act_chunks_data)

            ep_metadata, file_idx, obs_chunks, act_chunks = save_chunks(
                file_idx, args.chunks_per_file, output_dir_split, obs_chunks, act_chunks
            )
            episode_metadata.extend(ep_metadata)

            print(f"Episode {episode_idx} completed, length: {frame_counter}.")
            print(
                f"Total number of frames until now: {file_idx * args.chunk_size * args.chunks_per_file}"
            )
            episode_idx += 1
        else:
            print(f"Episode too short ({frame_counter}), resampling...")
    env.close()
    return episode_metadata


def main():
    assert (
        args.num_episodes_train % args.num_parallel_envs == 0
        and args.num_episodes_train >= args.num_parallel_envs
    )
    assert (
        args.num_episodes_val % args.num_parallel_envs == 0
        and args.num_episodes_val >= args.num_parallel_envs
    )
    assert (
        args.num_episodes_test % args.num_parallel_envs == 0
        and args.num_episodes_test >= args.num_parallel_envs
    )
    scenario = "deathmatch_simple"

    env_args = {
        "scenario": scenario,
        "frame_skip": 1,
        "frame_processor": envs.default_frame_processor,
        "n_bots": 8,
        "shaping": True,
        "initial_level": 5,
        "max_level": 5,
        "rolling_mean_length": 10,
    }

    eval_env_args = dict(env_args)
    new_env = dummy_vec_env_with_bots_curriculum(1, **env_args)
    agent = envs.load_model(
        args.agent_path,
        new_env,
    )

    if args.generate_gif:
        make_gif(agent, eval_env_args)
        return

    train_episode_metadata = make_array_records_dataset(
        agent,
        num_episodes=args.num_episodes_train,
        eval_env_args=eval_env_args,
        split="train",
    )
    val_episode_metadata = make_array_records_dataset(
        agent,
        num_episodes=args.num_episodes_val,
        eval_env_args=eval_env_args,
        split="val",
    )
    test_episode_metadata = make_array_records_dataset(
        agent,
        num_episodes=args.num_episodes_test,
        eval_env_args=eval_env_args,
        split="test",
    )
    # --- Save metadata ---
    metadata = {
        "env": "coinrun",
        "num_actions": 18,  # TODO mihir
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": np.mean(
            [ep["avg_seq_len"] for ep in train_episode_metadata]
        ),
        "avg_episode_len_val": np.mean(
            [ep["avg_seq_len"] for ep in val_episode_metadata]
        ),
        "avg_episode_len_test": np.mean(
            [ep["avg_seq_len"] for ep in test_episode_metadata]
        ),
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Done generating dataset.")


if __name__ == "__main__":
    main()

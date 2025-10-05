import typing as t

import cv2
import numpy as np
import vizdoom
from gym import Env
from gym import spaces
from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import ppo, policies
from vizdoom import GameVariable

from common.models import init_model
from common.utils import get_available_actions

from tqdm import tqdm
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

import torch
import os
import sys

Frame = np.ndarray

DOOM_ENV_WITH_BOTS_ARGS = """
    -host 1 
    -deathmatch 
    +viz_nocheat 0 
    +cl_run 1 
    +name AGENT 
    +colorset 0 
    +sv_forcerespawn 1 
    +sv_respawnprotect 1 
    +sv_nocrouch 1 
    +sv_noexit 1
    """


class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(
        self, game: vizdoom.DoomGame, frame_processor: t.Callable, frame_skip: int = 4
    ):
        super().__init__()

        # Determine action space
        self.action_space = spaces.Discrete(game.get_available_buttons_size())

        # Determine observation space
        h, w, c = (
            game.get_screen_height(),
            game.get_screen_width(),
            game.get_screen_channels(),
        )
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8
        )

        # Assign other variables
        self.game = game
        self.possible_actions = np.eye(
            self.action_space.n
        ).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor

        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame

    def step(self, action: int) -> t.Tuple[Frame, int, bool, t.Dict]:
        """Apply an action to the environment.

        Args:
            action:

        Returns:
            A tuple containing:
                - A numpy ndarray containing the current environment state.
                - The reward obtained by applying the provided action.
                - A boolean flag indicating whether the episode has ended.
                - An empty info dict.
        """
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self) -> Frame:
        """Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self._get_frame()

        return self.state

    def close(self) -> None:
        self.game.close()

    def render(self, mode="human"):
        pass

    def _get_frame(self, done: bool = False) -> Frame:
        return (
            self.frame_processor(self.game.get_state().screen_buffer)
            if not done
            else self.empty_frame
        )


class DoomWithBots(DoomEnv):

    def __init__(self, game, frame_processor, frame_skip, n_bots):
        super().__init__(game, frame_processor, frame_skip)
        self.n_bots = n_bots
        self.last_frags = 0
        self._reset_bots()

        # Redefine the action space using combinations.
        self.possible_actions = get_available_actions(
            np.array(game.get_available_buttons())
        )
        self.action_space = spaces.Discrete(len(self.possible_actions))

    def step(self, action):
        self.game.make_action(self.possible_actions[action], self.frame_skip)

        # Compute rewards.
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = frags - self.last_frags
        self.last_frags = frags

        # Check for episode end.
        self._respawn_if_dead()
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self):
        self._reset_bots()
        self.last_frags = 0

        return super().reset()

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()

    def _reset_bots(self):
        # Make sure you have the bots.cfg file next to the program entry point.
        self.game.send_game_command("removebots")
        for i in range(self.n_bots):
            self.game.send_game_command("addbot")

    def _print_state(self):
        server_state = self.game.get_server_state()
        player_scores = list(
            zip(
                server_state.players_names,
                server_state.players_frags,
                server_state.players_in_game,
            )
        )
        player_scores = sorted(player_scores, key=lambda tup: tup[1])

        # print('*** DEATHMATCH RESULTS ***')
        # for player_name, player_score, player_ingame in player_scores:
        #     if player_ingame:
        #         print(f' - {player_name}: {player_score}')


def default_frame_processor(frame: Frame) -> Frame:
    return cv2.resize(
        frame[40:, 4:-4], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
    )


def create_env(scenario: str, **kwargs) -> DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f"scenarios/{scenario}.cfg")
    game.set_window_visible(False)
    game.init()

    # Wrap the game with the Gym adapter.
    return DoomEnv(game, **kwargs)


def create_env_with_bots(scenario: str, **kwargs) -> DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f"scenarios/{scenario}.cfg")
    game.add_game_args(DOOM_ENV_WITH_BOTS_ARGS)
    game.set_window_visible(False)
    game.init()

    return DoomWithBots(game, **kwargs)


def create_vec_env(n_envs: int = 1, **kwargs) -> VecTransposeImage:
    return VecTransposeImage(
        SubprocVecEnv([lambda: create_env(**kwargs) for _ in range(n_envs)])
    )


def vec_env_with_bots(n_envs: int = 1, **kwargs) -> VecTransposeImage:
    return VecTransposeImage(
        SubprocVecEnv([lambda: create_env_with_bots(**kwargs) for _ in range(n_envs)])
    )


def create_eval_vec_env(**kwargs) -> vec_env.VecTransposeImage:
    return create_vec_env(n_envs=1, **kwargs)


def solve_env(
    env: vec_env.VecTransposeImage,
    eval_env: vec_env.VecTransposeImage,
    scenario: str,
    agent_args: t.Dict,
    resume: bool = False,
    load_path: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if resume:
        # Load the existing model
        if load_path != "":
            agent = ppo.PPO.load(
                load_path, env=env, tensorboard_log="logs/tensorboard", **agent_args
            )
            print(f"Resumed training from {load_path}")
        else:
            print("Resume selected but no path provided")
            sys.exit()
    else:
        # Create a new agent
        agent = ppo.PPO(
            policies.ActorCriticCnnPolicy,
            env,
            tensorboard_log="logs/tensorboard",
            seed=0,
            **agent_args,
        )
        init_model(agent)

    agent.policy.to(device)

    # Create callbacks.
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=4000,
        log_path=f"logs/evaluations/{scenario}",
        best_model_save_path=f"logs/models/{scenario}",
    )

    # Set up progress bar
    total_timesteps = 10_000_000
    pbar = tqdm(total=total_timesteps, desc="Training Progress")

    class ProgressBarCallback(BaseCallback):
        def __init__(self, pbar):
            super().__init__()
            self.pbar = pbar
            self.last_time = time.time()
            self.last_timesteps = 0

        def _on_step(self):
            current_timesteps = self.num_timesteps - self.last_timesteps
            self.pbar.update(current_timesteps)
            self.last_timesteps = self.num_timesteps

            current_time = time.time()
            steps_per_second = current_timesteps / (current_time - self.last_time)
            self.pbar.set_postfix({"steps/s": f"{steps_per_second:.2f}"})
            self.last_time = current_time

            return True

        def on_training_end(self):
            self.pbar.close()

    progress_callback = ProgressBarCallback(pbar)

    # Start the training process.
    try:
        agent.learn(
            total_timesteps=10_000_000,
            tb_log_name=scenario,
            callback=[eval_callback, progress_callback],
            reset_num_timesteps=not resume,  # Don't reset timesteps if resuming
        )
    finally:
        pbar.close()
        env.close()
        eval_env.close()

    return agent


def save_model(agent: ppo.PPO, scenario: str):
    """Save the trained model."""
    save_path = f"logs/models/{scenario}/final_model"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    agent.save(save_path)
    print(f"Model saved to {save_path}")


def load_model(load_path: str, env: vec_env.VecTransposeImage) -> ppo.PPO:
    """Load a trained model."""
    if not os.path.exists(os.path.dirname(load_path)):
        os.makedirs(os.path.dirname(load_path))
    agent = ppo.PPO.load(load_path, env=env)
    print(f"Model loaded from {load_path}")
    return agent

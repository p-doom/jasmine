import itertools
import typing as t

import numpy as np
from vizdoom import Button

# Buttons that cannot be used together
MUTUALLY_EXCLUSIVE_GROUPS = [
    [Button.MOVE_RIGHT, Button.MOVE_LEFT],
    [Button.TURN_RIGHT, Button.TURN_LEFT],
    [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
]

# Buttons that can only be used alone.
EXCLUSIVE_BUTTONS = [Button.ATTACK]


def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
    exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)

    # Flag actions that have more than 1 active button among exclusive list.
    return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (
        np.sum(actions, axis=-1) > 1
    )


def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
    # Create mask of shape (n_mutual_exclusion_groups, n_available_buttons), marking location of excluded pairs.
    mutual_exclusion_mask = np.array(
        [
            np.isin(buttons, excluded_group)
            for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS
        ]
    )

    # Flag actions that have more than 1 button active in any of the mutual exclusion groups.
    return np.any(
        np.sum(
            # Resulting shape (n_actions, n_mutual_exclusion_groups, n_available_buttons)
            (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
            axis=-1,
        )
        > 1,
        axis=-1,
    )


def get_available_actions(buttons: np.array) -> t.List[t.List[float]]:
    # Create list of all possible actions of size (2^n_available_buttons x n_available_buttons)
    action_combinations = np.array(
        [list(seq) for seq in itertools.product([0.0, 1.0], repeat=len(buttons))]
    )

    # Build action mask from action combinations and exclusion mask
    illegal_mask = has_excluded_pair(
        action_combinations, buttons
    ) | has_exclusive_button(action_combinations, buttons)

    possible_actions = action_combinations[~illegal_mask]
    possible_actions = possible_actions[
        np.sum(possible_actions, axis=1) > 0
    ]  # Remove no-op

    print(
        "Built action space of size {} from buttons {}".format(
            len(possible_actions), buttons
        )
    )
    return possible_actions.tolist()

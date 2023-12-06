"""Grid containing blue (-1) or red (1) squares."""

import enum
import functools

import numpy as np
import scipy.signal

import agent as agent_lib


class Side(enum.IntEnum):
    """Gives a semantic meaning to the values in the grid."""

    RED = 1
    BLUE = -1


class Grid(np.ndarray):
    """A numpy array with an extra update function, to update the values."""

    def __init__(self, *args, dtype: type(np.int8) = None, **kwargs):
        super().__init__(*args, **kwargs)
        if dtype != np.int8:
            raise ValueError(
                f"The grid should have the type np.int8. Got {dtype}."
            )

    def update_swap(
        self,
        actions: np.ndarray,
        agent_types: np.ndarray,
        agent_positions: np.ndarray,
    ) -> None:
        """Updates the agent positions on the grid using their actions."""
        for action, agent_type, (x, y) in zip(
            actions, agent_types, agent_positions
        ):
            if action == agent_lib.Action.SWAP:
                self[x, y] += int(agent_type)
                self[x, y] = np.clip(self[x, y], -1, 1)

    def update_gol(self) -> None:
        """Updates the grid, following the rule of the game of life."""
        full_neighbors = functools.partial(
            scipy.signal.convolve2d,
            mode="same",
            boundary="wrap",
            in2=np.ones((3, 3)),
        )
        red_neighbors = full_neighbors(self > 0) - (self > 0)
        blue_neighbors = full_neighbors(self < 0) - (self < 0)
        red_update = (red_neighbors == 3) | ((self > 0) & (red_neighbors == 2))
        blue_update = (blue_neighbors == 3) | (
            (self < 0) & (blue_neighbors == 2)
        )
        new_grid = red_update.astype(int) - blue_update.astype(int)
        self[:, :] = new_grid

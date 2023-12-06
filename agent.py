"""Defines the agent class."""

import enum

import numpy as np


class Action(enum.IntEnum):
    """Actions agents can take."""

    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SWAP = 5


def egocentric_views(
    agent_positions: np.ndarray,
    # Not using the grid_lib.Grid type here as that would could circular deps.
    grid: np.ndarray,
    size: int,
) -> np.ndarray:
    """Returns the aggregated egocentric views of agents."""
    views = np.zeros((len(agent_positions), size**2))
    for i, (x, y) in enumerate(agent_positions):
        flattened_view = grid[
            x - size // 2 : x + size // 2 + 1, y - size // 2 : y + size // 2 + 1
        ].flatten()
        # Cutting the inputs outside the screen.
        views[i][: len(flattened_view)] = flattened_view
    return views

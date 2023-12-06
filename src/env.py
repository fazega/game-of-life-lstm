"""Defines the environment state and its update."""

import dataclasses

import numpy as np

import src.agent as agent_lib
import src.grid as grid_lib


EGOCENTRIC_SIZE = 5


@dataclasses.dataclass
class EnvironmentState:
    """Container for the environment state.

    Attributes:
        agent_sides: Which side is the agent competing for: blue or red. Shape
            (num_agents,), type np.int8 (values in {-1, 1}). See grid_lib.Side.
        agent_positions: Positions of the agents in the environment. Shape
            (num_agents, 2), type np.int32.
        agent_states: LSTM states of the agents. Internal shape
            (num_agents, lstm_size), type np.float32.
    """

    grid: grid_lib.Grid
    agent_types: np.ndarray
    agent_positions: np.ndarray
    agent_states: np.ndarray

    def update_agent_positions(self, actions: np.ndarray) -> None:
        """Updates agent positions in place given actions."""
        n = len(self.grid)
        for i, (action, (x, y)) in enumerate(
            zip(actions, self.agent_positions)
        ):
            match action:
                case agent_lib.Action.UP:
                    self.agent_positions[i] = ((x + 1) % n, y)
                case agent_lib.Action.DOWN:
                    self.agent_positions[i] = ((x - 1) % n, y)
                case agent_lib.Action.LEFT:
                    self.agent_positions[i] = (x, (y - 1) % n)
                case agent_lib.Action.RIGHT:
                    self.agent_positions[i] = (x, (y + 1) % n)

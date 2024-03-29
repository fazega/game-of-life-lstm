"""Main file to launch the program."""

import random

import haiku as hk
import jax
import numpy as np
import wandb

import src.env as env_lib
import src.grid as grid_lib
import src.network as network_lib
import src.training as training_lib
import src.visualization as vis_lib


def main_train(num_steps: int):
    """Trains agents in the game of life, starting with random."""
    wandb.init(project="game-of-life-lstm")

    grid_size = 60
    grid = np.random.randint(-1, 2, (grid_size, grid_size), dtype=np.int8).view(
        grid_lib.Grid
    )
    num_agents = 10
    agent_positions = np.array(
        [
            (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            for _ in range(num_agents)
        ]
    )
    agent_types = np.array(
        [int(random.choice(list(grid_lib.Side))) for _ in range(num_agents)]
    )
    agent_states = hk.LSTMState(
        hidden=np.zeros((num_agents, network_lib.LSTM_SIZE)),
        cell=np.zeros((num_agents, network_lib.LSTM_SIZE)),
    )
    env_state = env_lib.EnvironmentState(
        grid=grid,
        agent_positions=agent_positions,
        agent_types=agent_types,
        agent_states=agent_states,
    )
    model = hk.transform(network_lib.policy)
    model_params = model.init(
        jax.random.PRNGKey(0),
        np.zeros((env_lib.EGOCENTRIC_SIZE**2,)),
        hk.LSTMState(hidden=agent_states.hidden[0], cell=agent_states.cell[0]),
    )

    return training_lib.train(
        env_state=env_state,
        model_params=model_params,
        train_config=training_lib.TrainConfig(
            batch_size=32,
            learning_rate=1e-4,
        ),
        num_steps=num_steps,
    )


def train_and_visualize():
    """Trains some agents and visualize them in the environment, once trained."""
    # Reduce the number of training steps to visualize the game quickly.
    env_state, model_params = main_train(num_steps=10_000)
    vis_lib.visualize(
        env_state=env_state,
        model_params=model_params,
    )


if __name__ == "__main__":
    train_and_visualize()

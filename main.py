"""Main file to launch the program."""

import copy
import dataclasses
import enum
import numpy as np
import random
import optax
import pygame
import time
from scipy import signal

import jax
import jax.nn as jnn
import jax.numpy as jnp
import haiku as hk
import tree

import agent as agent_lib
import grid as grid_lib
import network as network_lib


_SCREEN_SIZE = 600
_EGOCENTRIC_SIZE = 5


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
        n = len(self.grid)
        for i, (action, (x, y)) in enumerate(zip(actions, self.agent_positions)):
            match action:
                case agent_lib.Action.UP:
                    self.agent_positions[i] = ((x + 1) % n, y)
                case agent_lib.Action.DOWN:
                    self.agent_positions[i] = ((x - 1) % n, y)
                case agent_lib.Action.LEFT:
                    self.agent_positions[i] = (x, (y - 1) % n)
                case agent_lib.Action.RIGHT:
                    self.agent_positions[i] = (x, (y + 1) % n)


@dataclasses.dataclass
class TrainConfig:
    """Config for the training script.
    
    Attributes:
        batch_size: Batch size to use for training.
        learning_rate: Used by Adam.
        score_delay: How long to wait before the score is actually used for
            training. The agent train with the score at t+score_delay at
            timestep t.
    """

    batch_size: int
    learning_rate: float
    score_delay: int = 10


def train_simul(
    env_state: EnvironmentState,
    model_params: hk.Params,
    train_config: TrainConfig,
    num_steps: int = 100,
) -> None:
    """Trains the model using some agents in the environment.
    
    We make one gradient pass per environment step, once the training data
    pool is big enough.

    Args:
        env_state: Initial environment state, with the grid, agent positions,
            types, and LSTM states.
        model_params: Initial weights of the LSTM.
        train_config: Various arguments for training like lr.
        num_steps: Number of steps to do in the environment.
    """
    rng = np.random.default_rng(1)
    # Set up the update function.
    model = hk.transform(network_lib.policy)
    optimizer = optax.adam(learning_rate=train_config.learning_rate)
    optimizer_state = optimizer.init(model_params)
    loss_fn = network_lib.make_loss_fn(model)
    grad_fn = jax.value_and_grad(loss_fn)
    
    @jax.jit
    def update_fn(model_params, optimizer_state, inputs, states, normalized_future_scores, actions):
        """Updates some parameters using gradients on the passed data."""
        loss, grads = grad_fn(
            model_params, inputs, states, normalized_future_scores, actions
        )
        new_params, new_optimizer_state = optimizer.update(
            model_params, optimizer_state, grads
        )
        return loss, new_params, new_optimizer_state

    last_red_scores, last_blue_scores = [], []
    train_data = []
    for step in range(num_steps):
        # Get the egocentric views of the agents.
        views = agent_lib.egocentric_views(
            agent_positions=env_state.agent_positions,
            grid=env_state.grid,
            size=_EGOCENTRIC_SIZE,
        )

        # Retrieve the policies and sample random actions.
        log_probs, new_states = model.apply(
            model_params, None, views, env_state.agent_states
        )
        probs = np.exp(log_probs)
        oh_actions = rng.multinomial(n=1, pvals=probs, size=(len(log_probs),))
        actions = np.argmax(oh_actions, axis=-1)

        # Update training data with inputs and action taken.
        for i in range(len(env_state.agent_positions)):
            train_data.append(
                [views[i],
                tree.map_structure(lambda x: x[i], env_state.agent_states),
                0.,
                actions[i]]
            )
        
        # Update the environment with the actions.
        env_state.update_agent_positions(actions)
        env_state.grid.update_swap(actions, env_state.agent_types, env_state.agent_positions)
        env_state.grid.update_gol()

        # Retrieving the scores.
        red_sum = float(np.sum(env_state.grid == 1))
        blue_sum = float(np.sum(env_state.grid == -1))
        red_score = red_sum / (red_sum + blue_sum)
        blue_score = blue_sum / (red_sum + blue_sum)
        total_delay = train_config.score_delay * len(env_state.agent_positions)
        if len(train_data) >= total_delay:
            # Update the scores in the training data.
            normalized_red_score = (red_score - 0.5) * 2
            normalized_blue_score = (blue_score - 0.5) * 2
            for i, agent_type in enumerate(env_state.agent_types):
                index = -total_delay + i
                if agent_type == grid_lib.Side.RED:
                    train_data[index][2] = normalized_red_score
                else:
                    train_data[index][2] = normalized_blue_score

        # Actual training.
        loss = None
        if len(train_data[:-total_delay]) >= train_config.batch_size:
            batch = random.choices(
                train_data[:-total_delay],
                k=train_config.batch_size,
            )
            inputs, states, normalized_future_scores, actions = zip(*batch)
            inputs = np.stack(inputs, axis=0)
            states = hk.LSTMState(hidden=np.stack([x.hidden for x in states]),
                                  cell=np.stack([x.cell for x in states]))
            normalized_future_scores = np.array(normalized_future_scores)
            # Function to put the scores between -infinity and +infinity.
            normalized_future_scores = 5 * np.tan(
                np.clip(normalized_future_scores, -0.999, 0.999) * (np.pi/2),
            )
            actions = np.array(actions, dtype=int)
            loss, model_params, optimizer_state = update_fn(
                model_params, optimizer_state, inputs, states,
                normalized_future_scores, actions
            )

        # Logging.
        if step % 100 == 0:
            print(f'Step: {step}, Scores: {red_score*100:.1f}%'
                  f'-{blue_score*100:.1f}%, Loss: {loss}')
        
        # Stop the loop when one of the two teams has no square left.
        if red_score == 0. or blue_score == 0.:
            break
        
    return (env_state, model_params)


def visualize_simul(
    env_state: EnvironmentState,
    model_params: hk.Params,
) -> None:
    """Displays some agents interacting in the environment."""
    rng = np.random.default_rng(1)
    model = hk.transform(network_lib.policy)
    size_grid = len(env_state.grid)

    # Initialize the pygame screen.
    pygame.init()
    screen = pygame.display.set_mode(
        (_SCREEN_SIZE, _SCREEN_SIZE))
    pygame.display.set_caption("Life game")
    my_font = pygame.font.SysFont('Arial', 16, True)

    # Used for logging.
    last_red_scores, last_blue_scores = [], []

    running = True
    while running:
        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        gridsize = (_SCREEN_SIZE//size_grid)
        for i in range(size_grid):
            for j in range(size_grid):
                pixel = env_state.grid[i, j]
                if pixel == 0:
                    color = (0, 0, 0)
                elif pixel == grid_lib.Side.RED:
                    color = (255, 0, 0)
                elif pixel == grid_lib.Side.BLUE:
                    color = (0, 0, 255)
                pygame.draw.rect(screen, color, (j*gridsize,i*gridsize, gridsize, gridsize))

        for agent_type, agent_pos in zip(env_state.agent_types, env_state.agent_positions):
            x, y = agent_pos
            color = (255, 0, 0) if agent_type == grid_lib.Side.RED else (0, 0, 255)
            pygame.draw.circle(screen, color, (y*gridsize+gridsize/2, x*gridsize+gridsize/2), gridsize//2, 2)

        # Update.
        # Get the egocentric views of the agents.
        views = agent_lib.egocentric_views(
            agent_positions=env_state.agent_positions,
            grid=env_state.grid,
            size=_EGOCENTRIC_SIZE,
        )

        # Retrieve the policies and sample random actions.
        log_probs, new_states = model.apply(
            model_params, None, views, env_state.agent_states
        )
        probs = np.exp(log_probs)
        oh_actions = rng.multinomial(n=1, pvals=probs, size=(len(log_probs),))
        actions = np.argmax(oh_actions, axis=-1)
        
        # Update the environment with the actions.
        env_state.update_agent_positions(actions)
        env_state.grid.update_swap(actions, env_state.agent_types, env_state.agent_positions)
        env_state.grid.update_gol()

        # Display the scores.
        red_sum = float(np.sum(env_state.grid == 1))
        blue_sum = float(np.sum(env_state.grid == -1))
        red_score = red_sum / (red_sum + blue_sum)
        blue_score = blue_sum / (red_sum + blue_sum)
        max_scores = 30
        delta = 4
        height = 60
        last_red_scores.append(red_score)
        if len(last_red_scores) >= max_scores:
            last_red_scores.pop(0)
        last_blue_scores.append(blue_score)
        if len(last_blue_scores) >= max_scores:
            last_blue_scores.pop(0)

        pygame.draw.rect(screen, (0, 0, 0), (_SCREEN_SIZE - 250, 10, delta * max_scores, height))
        pygame.draw.rect(screen, (255, 255, 255), (_SCREEN_SIZE - 250, 10, delta * max_scores, height), width=2)
        if len(last_red_scores) >= 2:
            red_scores_coords = [(_SCREEN_SIZE - 250 + i * delta, 10 + height * (1 - score)) for i, score in enumerate(last_red_scores)]
            pygame.draw.lines(screen, (255, 0, 0), False, red_scores_coords, width=2)
        if len(last_blue_scores) >= 2:
            blue_scores_coords = [(_SCREEN_SIZE - 250 + i * delta, 10 + height * (1 - score)) for i, score in enumerate(last_blue_scores)]
            pygame.draw.lines(screen, (0, 0, 255), False, blue_scores_coords, width=2)
        red_text = my_font.render(f'Red: {red_score*100:.1f}%', False, (255, 255, 255))
        screen.blit(red_text, (_SCREEN_SIZE - 100, 10))
        blue_text = my_font.render(f'Blue: {blue_score*100:.1f}%', False, (255, 255, 255))
        screen.blit(blue_text, (_SCREEN_SIZE - 100, 40))
        pygame.display.update()

        # Cap the frame rate.
        pygame.time.Clock().tick(20)

    # Clean up.
    pygame.quit()


def main_train(num_steps: int):
    """Trains agents in the game of life, starting with random."""
    N = 60
    grid = np.random.randint(-1, 2, (N, N), dtype=np.int8).view(grid_lib.Grid)
    num_agents = 10
    agent_positions = np.array([(random.randint(0, N-1), random.randint(0, N-1)) for _ in range(num_agents)])
    agent_types = np.array([int(random.choice(list(grid_lib.Side))) for _ in range(num_agents)])
    zero_state = hk.LSTMState(hidden=np.zeros([network_lib.LSTM_SIZE]), cell=np.zeros([network_lib.LSTM_SIZE]))
    agent_states = tree.map_structure(lambda x: np.stack([x]*num_agents), zero_state)
    env_state = EnvironmentState(
        grid=grid,
        agent_positions=agent_positions,
        agent_types=agent_types,
        agent_states=agent_states,
    )
    model = hk.transform(network_lib.policy)
    model_params = model.init(jax.random.PRNGKey(0), np.zeros((_EGOCENTRIC_SIZE**2,)), zero_state)

    return train_simul(
        env_state=env_state,
        model_params=model_params,
        train_config=TrainConfig(batch_size=32, learning_rate=1e-4),
        num_steps=num_steps,
    )


# Train agents and then visualize.
env_state, model_params = main_train(num_steps=5)
visualize_simul(
    env_state=env_state,
    model_params=model_params,
)



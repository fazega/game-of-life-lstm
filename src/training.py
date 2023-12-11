"""Training function for the agents."""

import copy
import dataclasses
import operator
import random

import jax
import haiku as hk
import numpy as np
import optax
import tree
import wandb

import src.agent as agent_lib
import src.env as env_lib
import src.grid as grid_lib
import src.network as network_lib


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


def train(
    env_state: env_lib.EnvironmentState,
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
    init_env_state = copy.deepcopy(env_state)
    rng = np.random.default_rng(1)
    # Set up the update function.
    model = hk.transform(network_lib.policy)
    optimizer = optax.adam(learning_rate=train_config.learning_rate)
    optimizer_state = optimizer.init(model_params)
    loss_fn = network_lib.make_loss_fn(model)
    grad_fn = jax.value_and_grad(loss_fn)

    @jax.jit
    def update_fn(
        model_params,
        optimizer_state,
        inputs,
        states,
        normalized_future_scores,
        actions,
    ):
        """Updates some parameters using gradients on the passed data."""
        loss, grads = grad_fn(
            model_params, inputs, states, normalized_future_scores, actions
        )
        new_params, new_optimizer_state = optimizer.update(
            model_params, optimizer_state, grads
        )
        return loss, new_params, new_optimizer_state

    train_data = []
    for step in range(num_steps):
        # Get the egocentric views of the agents.
        views = agent_lib.egocentric_views(
            agent_positions=env_state.agent_positions,
            grid=env_state.grid,
            size=env_lib.EGOCENTRIC_SIZE,
        )

        # Retrieve the policies and sample random actions.
        log_probs, env_state.agent_states = model.apply(
            model_params, None, views, env_state.agent_states
        )
        probs = np.exp(log_probs)
        mean_entropy = np.mean(-probs * log_probs)
        oh_actions = rng.multinomial(n=1, pvals=probs, size=(len(log_probs),))
        actions = np.argmax(oh_actions, axis=-1)

        # Update training data with inputs and action taken.
        for i in range(len(env_state.agent_positions)):
            train_data.append(
                [
                    views[i],
                    tree.map_structure(
                        operator.itemgetter(i),
                        env_state.agent_states,
                    ),
                    0.0,
                    actions[i],
                ]
            )

        # Update the environment with the actions.
        env_state.update_agent_positions(actions)
        env_state.grid.update_swap(
            actions, env_state.agent_types, env_state.agent_positions
        )
        env_state.grid.update_gol()

        # Retrieving the scores.
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

        # Actual training.
        loss = None
        if len(train_data[:-total_delay]) >= train_config.batch_size:
            batch = random.choices(
                train_data[:-total_delay],
                k=train_config.batch_size,
            )
            inputs, states, normalized_future_scores, actions = zip(*batch)
            inputs = np.stack(inputs, axis=0)
            states = hk.LSTMState(
                hidden=np.stack([x.hidden for x in states]),
                cell=np.stack([x.cell for x in states]),
            )
            normalized_future_scores = np.array(normalized_future_scores)
            # Function to put the scores between -infinity and +infinity.
            normalized_future_scores = 5 * np.tan(
                np.clip(normalized_future_scores, -0.999, 0.999) * (np.pi / 2),
            )
            actions = np.array(actions, dtype=int)
            loss, model_params, optimizer_state = update_fn(
                model_params,
                optimizer_state,
                inputs,
                states,
                normalized_future_scores,
                actions,
            )

        # Logging.
        if step % 100 == 0:
            print(
                f"Step: {step}, Scores: {red_score*100:.1f}%"
                f"-{blue_score*100:.1f}%, Loss: {loss}, Entropy: {mean_entropy}"
            )
            wandb.log(
                {
                    "blue_score": blue_score,
                    "red_score": red_score,
                    "loss": loss,
                    "entropy": mean_entropy,
                }
            )

        # Reset the env if one team has no squares left.
        if red_score == 0.0 or blue_score == 0.0:
            env_state = copy.deepcopy(init_env_state)

    wandb.finish()

    return (env_state, model_params)

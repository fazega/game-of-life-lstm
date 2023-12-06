"""Functions defining the network, and how it's updated."""

from typing import Any

import jax
import jax.nn as jnn
import jax.numpy as jnp
import haiku as hk

import agent as agent_lib


LSTM_SIZE = 64


def policy(inputs: jax.Array, state: hk.LSTMState) -> jax.Array:
    """Returns the log-probabilities of actions."""
    h, new_state = hk.LSTM(LSTM_SIZE)(inputs, state)
    h = hk.Linear(64)(h)
    h = jnn.gelu(h)
    h = hk.Linear(64)(h)
    h = jnn.gelu(h)
    logits = hk.Linear(len(agent_lib.Action))(h)
    return jnn.log_softmax(logits, axis=-1), new_state


def make_loss_fn(model: hk.Transformed) -> Any:
    """Returns a loss function given a haiku model."""

    def loss_fn(
        params: hk.Params,
        inputs: jnp.ndarray,
        states: hk.LSTMState,
        normalized_future_scores: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jax.Array:
        """Returns a classical policy gradient (REINFORCE) loss."""
        # normalized means -1 for 100% blue, 1 for 100% red, 0 in the middle
        log_probs, _ = model.apply(params, None, inputs, states)
        action_log_probs = jnp.take_along_axis(
            log_probs, actions[:, None], axis=1
        )[:, 0]
        return -jnp.mean(normalized_future_scores * action_log_probs)

    return loss_fn

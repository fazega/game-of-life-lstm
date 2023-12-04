# LSTMs in Game of Life

The goal of this project is to try to train agents online in a modified game of
life environment with two teams (blue and red), in which the goal is to
maximize the proportion of the map of your color.

## Central parameters, decentralized states

The parameters of the LSTM are the same for all agents. The only thing that
differentiates them are their LSTM states.

## Observation, action and reward

The observation is an egocentric view of each agent, also seeing itself. The
actions are stay, up, down, left, right and a 'swap' action, which:
- if the tile the agent is on is of the opposite color as the agent's team, turn
it to black (i.e., kill it)
- otherwise, do nothing
Finally, there is no reward but rather a score G (sum of future rewards), which
is the relative proportion (in percentage) of the map controlled by the agent's
team. Note that we use a tangent transformation before passing to the agents to
get a score between -infinity and +infinity.

## Loss function and training

We train using an Adam optimizer, and a very simply policy gradient loss without
advantage (i.e. REINFORCE).

## Usage

Launch a training + visualization with `python3 main.py`.

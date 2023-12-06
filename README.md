# LSTMs in Game of Life

The goal of this project is to train agents online in a modified game of
life environment with two teams (blue and red), in which the goal is to
maximize the proportion of the map of your color.

## Rules of the game

### Game of life

The game of life usually happens on a binary grid of dimensions MxN. A cell with
0 is called *dead* and a cell with 1 is called *alive*. The game starts
with an initial state, and uses the following evolution rules:
- If a cell is dead, then it becomes alive if and only if it has exactly 3 alive
neighbour cells (a cell has 8 neighbours, we count the diagonals too).
- If a cell is alive, then it stays alive if and only if it has exactly 2 or 3
alive neighbour cells.

Note that the map has a torus topology: the edges flip over.

### Competitive GoL

In this project, we use a modified version with two teams, red and blue. The
grid is now not binary, but ternary. A cell with 0 is still called *dead*, a
cell with 1 is called *red* and a cell with -1 is called *blue*. The evolution
rules are the same for red and blue (replace red or blue by *alive*),
independently, but with one extra interaction rule:
- If a *dead* cell has exactly 3 *blue* neighbours and 3 *red* neighbours,
nothing happens.
It means in particular that alive cells from a given team have 'priority': if a
*red* cell has 2 or 3 *red* neighbours but also 2 or 3 *blue* neighbours, it
still stays *red*. Similarly, If a *blue* cell has 2 or 3 *blue* neighbours but
also 2 or 3 *red* neighbours, it still stays *blue*.

### Agents

Agents are added on top of the grid, and can freely move on it. They are part of
one of the two teams (*red* or *blue*). They have 5 actions to move: stay, up,
down, left, right. They also have a 6th crucial extra action: swap. This action,
if used, does the following: if the agent is *red*, add +1 to the current cell,
and if the agent is *blue*, add -1 to the current cell, and clip such that the
value of the cell is always in {-1, 0, 1}. The effect is that an
agent can kill a cell from the other team, or make any dead cell alive, with its
team's color. If the action is applied on an alive cell of the agent's color,
nothing happens.

## Observation, action and reward

The observation is an egocentric view of each agent, of only the grid (the
agent itself and other agents are **not included** in the observation). The
actions are described in the 'agents' section above.
There is no explicit reward but rather a score G (sum of future rewards), which
is the relative proportion (in percentage) of the map controlled by the agent's
team. Note that we use a tangent transformation before passing to the agents to
get a score between -infinity and +infinity.
The reward can be easily obtained by computing G_(t+1) - G_t.

## Central parameters, decentralized states

The parameters of the LSTM are the same for all agents. The only thing that
differentiates the agents are their LSTM states.

## Loss function and training

We train using an Adam optimizer, and a very simply policy gradient loss without
advantage (i.e. REINFORCE).

## Usage

Launch a training + visualization with `python3 main.py`.

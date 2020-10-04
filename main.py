"""Main file to launch the program."""
import numpy as np
import time
from grid import Grid

def launch_simul(
    size_grid: int, init_data: np.ndarray, n_steps: int = 1000,
    soft_strength: float = np.inf, min_n: int = 2, max_n: int = 3
) -> None:
    grid = Grid(size_grid)
    grid.set_data(init_data)

    for k in range(n_steps):
        grid.draw()
        time.sleep(0.2)
        grid.update(soft_strength=soft_strength, min_n=min_n, max_n=max_n)


def launch_game_of_life():
    """Launches a game of life, starting from the plane."""
    N = 30
    M = np.zeros((N, N))
    M[20, 22:25] = 1
    M[21, 24] = 1
    M[22, 23] = 1
    launch_simul(N, M)

def launch_random():
    """Launches a random recurrent convolutional simulation."""
    N = 30
    M = np.random.uniform(0, 1, (N, N))
    launch_simul(N, M, soft_strength=5.)


launch_game_of_life()

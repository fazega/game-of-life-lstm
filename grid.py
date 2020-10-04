"""Dynamic grid."""
import numpy as np
import time
import pygame
import random
from scipy import signal

def sigmoid(x, s):
    return 1/(1 + np.exp(-s*x))

class Grid:
    def __init__(self, n: int):
        self.n = n
        self.data = np.random.uniform(0, 1, (n,n))
        self.already_init_draw = False

        # Filtre aléatoire
        # self.theta = np.random.normal(0, 5, (3,3,))

        # Filtre donnant la somme des voisins
        self.theta = np.matrix([[1,1,1], [1,0,1], [1,1,1]])

    def set_data(self, data: np.ndarray):
        self.data = data

    def update(self, soft_strength: float = np.inf, min_n: float=2, max_n: float=3) -> None:
        """Update the values of the grid.

        This function applies a convolution filter to the grid, then an
        activation function. This activation can either be a heaviside or a
        sigmoid, depending on the value of 'soft'.
        Args:
            soft_strength: if np.inf, use heaviside, else use sigmoid with the
                given strength
            min_n: the minimum number of neighbours needed to stay alive
            max_n: the maximum number of neighbours needed to stay alive
        """
        self.data = signal.convolve2d(self.data, self.theta, mode="same", boundary="wrap")
        self.data = self._activation(self.data, soft=soft_strength, min_n=min_n, max_n=max_n)

    def _activation(self, v, soft, min_n, max_n):
        f = (lambda x: np.heaviside(x, 1)) if soft is np.inf else (lambda x: sigmoid(x, soft))
        return f(v-min_n) * (1 - f(v-max_n))

    def init_draw(self):
        self.screen_size = 1000
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Life game")
        self.already_init_draw = True

    def draw(self):
        if(self.already_init_draw == False):
            self.init_draw()

        gridsize = (self.screen_size//self.n)
        for i in range(self.n):
            for j in range(self.n):
                color = (self.data[i,j]*255,self.data[i,j]*255,self.data[i,j]*255)
                pygame.draw.rect(self.screen, color, (j*gridsize,i*gridsize, gridsize, gridsize))
        pygame.display.update()

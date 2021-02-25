from builtins import range
import time
from PerlinNoiseFactory import PerlinNoiseFactory
import numpy as np
import math
import random


class XMLenv:
    def __init__(self, max_episode_steps, size=201, obs_size=5, flat_word=False, debug=False):
        self.size = size
        self.debug = debug
        self.obs_size = obs_size
        self.flat_world = flat_word
        self.terrain_array = self.getTerrain()
        self.center = self.size//2
        self.max_episode_steps = max_episode_steps
        i = math.floor(random.random()*size)
        j = math.floor(random.random()*size)

        # coordinate in the form of (x, y, z)
        # FLOOR AGENTS X AND Z COORDINATES TO CHECK IF ITS AT THE END COORDINATE (floor -)
        self.end_coordinate = (
            i-self.center, self.terrain_array[j + self.obs_size, i + self.obs_size]+1, j-self.center)
        self.goal = np.array(
            [self.end_coordinate[0], self.end_coordinate[2]])
        self.start_coordinate = (
            0, 0)

    def getGoal(self):
        return self.goal

    def getTerrain(self):
        if self.flat_world:
            a = np.array([[5 for j in range(self.size)]
                          for i in range(self.size)])
        else:
            p = PerlinNoiseFactory(2, 4)
            a = np.array([[p(i/self.size, j/self.size)
                           for j in range(self.size)] for i in range(self.size)])
            a = np.abs((a*50).astype(int)) + 5

        a = np.pad(a, self.obs_size, constant_values=80)
        if self.debug:
            print("Terrain Map:", a)
        return a

    def inGoal(self, position):
        return (position[0] == self.goal[0]) and (position[1] == self.goal[1])

    def isEnd(self, position, steps):
        return self.inGoal(position) or steps >= self.max_episode_steps

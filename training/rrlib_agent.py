# Rllib docs: https://docs.ray.io/en/latest/rllib.html


import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import random
import gym
import ray
from gym.spaces import Discrete, Box
from ray.rllib.agents import ppo
from XMLenv import XMLenv
from Observations import DiscreteObservation
import os

# for custom model
import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from MyModel import MyModel
from Agent import Agent


class MoogleMap(gym.Env):

    def __init__(self, env_config):
        try:
            os.mkdir("./data")
            os.mkdir("./data/trajectory_graphs")
        except:
            pass

        try:
            os.mkdir("./data/models")
        except: pass


        try:
            os.mkdir("./data/trajectory_graphs/")
        except:
            pass

        # Static Parameters
        self.world_size = 51
        self.obs_size = 9

        self.move_reward_scale = 5  # norm 2
        self.end_reward = 100

        self.reward_dict = {
            0: -1,  # Move one block forward
            1: -1,  # Turn 90 degrees to the right
            2: -1,  # Turn 90 degrees to the left
            3: -2
        }

        self.max_episode_steps = 50
        self.log_frequency = 100
        self.flatland = False
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'jumpmove 1'
        }

        self.debug_obs = False
        self.debug_turn = False

        # Rllib Parameters
        # self.action_space = Box(-1,1,shape=(3,), dtype=np.float32)
        self.action_space = Discrete(len(self.action_dict))
        self.obseravtion = DiscreteObservation(self.obs_size, self.world_size)
        self.observation_space = self.obseravtion.getBox()

        # DiamondCollector Parameters
        self.obs = None
        self.prev_position = np.array([0, 0])
        self.environment = XMLenv(
            self.max_episode_steps, self.world_size, self.obs_size, flat_word=self.flatland)

        self.agent = Agent(self.environment.start_coordinate,self.world_size, self.obs_size)

        # self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

        self.episode_dist_return = 0
        self.dist_returns = []
        self.average_dist = self.get_average_dist()

        self.start_time = time.time()
        self.times = []

        # for ploting the agent's trajectory
        self.coordinates = []
        self.graph_num = 0

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """
        # Reset Malmo and the envrionment

        self.returns.append(self.episode_return)
        self.dist_returns.append(self.episode_dist_return)
        self.times.append(time.time() - self.start_time)

        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
                len(self.returns) % self.log_frequency == 0:
            self.log_returns()
            self.log_dist_return()
            self.log_time_graph()
            try:
                self.draw_agent_trajectory()
            except: pass

        self.environment = XMLenv(
            self.max_episode_steps, self.world_size, self.obs_size, flat_word=self.flatland)
        self.agent.reset(self.environment.start_coordinate)

        # Reset Variables
        self.episode_dist_return = 0
        self.episode_return = 0
        self.episode_step = 0
        self.coordinates.clear()

        # Get Observation
        self.obs, self.prev_position = self.get_observation()

        self.start_time = time.time()

        return self.obs

    def step(self, action):
        """
        Take an action in the environment and return the results.

        Args
            action: <int> index of the action to take

        Returns
            observation: <np.array> flattened array of obseravtion
            reward: <int> reward from taking action
            done: <bool> indicates terminal state
            info: <dict> dictionary of extra information
        """

        # For Discrete Actions

        command = self.action_dict[action]
        self.agent.doAction(command, self.environment.terrain_array)
        if self.debug_turn:
            print("[TURN DEBUG] Command:", command)
        self.episode_step += 1

        # Get Observation

        self.obs, pos = self.get_observation()

        # Get Done
        done = self.environment.isEnd(self.agent.getPosition(), self.episode_step)

        # Get Reward
        reward = 0
        if not done:
            reward += (np.sum(np.abs(self.prev_position - self.environment.getGoal())) - np.sum(
                np.abs(pos - self.environment.getGoal())))*self.move_reward_scale  # L1 for discrete

            self.episode_dist_return += reward / (self.average_dist * self.move_reward_scale)
        else:
            if self.environment.inGoal(self.agent.getPosition()):
                reward += self.end_reward
                self.episode_dist_return += 1/self.average_dist

        # reward = np.clip(reward,-self.move_reward_scale,self.move_reward_scale)

        if self.debug_turn:
            print("[TURN DEBUG] Distance Reward:", reward)

        reward += self.reward_dict[action]
        if self.debug_turn:
            print("[TURN DEBUG] Total Reward:", reward)
        self.prev_position = pos

        if not done:
            self.coordinates.append(pos)

        self.episode_return += reward
        # print("reward received:",reward)

        #time.sleep(60) ####

        return self.obs, reward, done, dict()

    def get_observation(self):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent.
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
        """

        obs = self.obseravtion.getObservation(
            self.environment.terrain_array, self.agent.getPosition()[0], self.agent.getPosition()[1], self.agent.getYaw(), self.environment.getGoal(), self.debug_obs)
        point = np.copy(self.agent.getPosition())



        return obs, point

    def draw_agent_trajectory(self):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        x = np.linspace(-self.world_size//2,
                        self.world_size//2, self.world_size)
        y = np.linspace(-self.world_size//2,
                        self.world_size//2, self.world_size)
        X, Y = np.meshgrid(x, y)
        Z = self.environment.terrain_array[self.obs_size:self.world_size +
                                           self.obs_size, self.obs_size:self.world_size+self.obs_size]
        plt.contourf(X, Y, Z, cmap='RdGy')
        plt.colorbar()

        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.scatter(
            self.environment.start_coordinate[0], self.environment.start_coordinate[1], c="b")
        ax.scatter(
            self.environment.end_coordinate[0], self.environment.end_coordinate[2], c="m")

        plt.xlim([-self.world_size/2, self.world_size/2])
        plt.ylim([-self.world_size/2, self.world_size/2])
        xpos, ypos = zip(*self.coordinates)
        ax.plot(xpos, ypos)

        plt.title('Agent coordinate')
        plt.savefig(
            f'./data/trajectory_graphs/AgentCoords_{self.graph_num}.png')
        plt.close()

        self.graph_num += 1

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """

        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        # print(f"debug for self.")
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Moogle Map')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('./data/returns.png')

        with open('./data/returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))

    def log_dist_return(self):

        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.dist_returns[1:], box, mode='same')
        plt.clf()
        # plt.ylim(top=1)
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Moogle Map')
        plt.ylabel('Distance Value')
        plt.xlabel('Steps')
        plt.savefig('./data/returns_dist.png')

        with open('./data/returns_dist.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.dist_returns[1:]):
                f.write("{}\t{}\n".format(step, value))

    def log_time_graph(self):

        box = np.ones(self.log_frequency) / self.log_frequency

        divlist = list(t/d if d != 0 else float("inf")
                       for (d, t) in zip(self.dist_returns[1:], self.times[1:]))

        returns_smooth = np.convolve(divlist, box, mode='same')
        plt.clf()
        # plt.ylim(top=1)
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Moogle Map')
        plt.ylabel('Time Spent / Distance Value')
        plt.xlabel('Steps')
        plt.savefig('./data/time_per_distance_graph.png')

        with open('./data/returns_time.txt', 'w') as f:
            for step, value in zip(self.steps[1:], divlist):
                f.write("{}\t{}\n".format(step, value))

    def get_average_dist(self):
        a = np.zeros((self.world_size, self.world_size))
        for i in range(self.world_size):
            for j in range(self.world_size):
                a[i, j] = abs(i-self.world_size//2) + abs(j-self.world_size//2)
        return (np.sum(a) / (a.shape[0] * a.shape[1]))


if __name__ == '__main__':
    ModelCatalog.register_custom_model('my_model', MyModel)

    ray.init()
    trainer = ppo.PPOTrainer(env=MoogleMap, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0,            # We aren't using parallelism
        'model': {
            'custom_model': 'my_model',
            'custom_model_config': {
            }
        }

    })

    trainer.restore('./checkpoint_2751/checkpoint-2751')

    i = 0
    while True:
        print("episode_reward_mean:",trainer.train()['episode_reward_mean'])
        if i % 50 == 0:
            checkpoint = trainer.save("./data/models")
            print("checkpoint saved at", checkpoint)
        i+=1

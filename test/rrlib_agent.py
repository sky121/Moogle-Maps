# Rllib docs: https://docs.ray.io/en/latest/rllib.html

try:
    from malmo import MalmoPython
except:
    import MalmoPython

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


class MoogleMap(gym.Env):

    def __init__(self, env_config):
        # Static Parameters
        self.world_size = 21
        self.obs_size = 5
        self.move_reward_scale = 5  # norm 2
        self.max_episode_steps = 100
        self.log_frequency = 10
        self.flatland = False
        self.reach_end_reward = 20
        self.action_dict = {
            0: 'move 1',  # Move one block forward
            1: 'turn 1',  # Turn 90 degrees to the right
            2: 'turn -1',  # Turn 90 degrees to the left
            3: 'jumpmove 1'
        }
        self.reward_dict = {
            0: -1,  # Move one block forward
            1: -1,  # Turn 90 degrees to the right
            2: -1,  # Turn 90 degrees to the left
            3: -2
        }

        self.debug_obs = False

        # Rllib Parameters
        #self.action_space = Box(-1,1,shape=(3,), dtype=np.float32)
        self.action_space = Discrete(len(self.action_dict))
        self.obseravtion = DiscreteObservation(self.obs_size, self.world_size)
        self.observation_space = self.obseravtion.getBox()

        # Malmo Parameters
        self.agent_host = MalmoPython.AgentHost()
        try:
            self.agent_host.parse(sys.argv)
        except RuntimeError as e:
            print('ERROR:', e)
            print(self.agent_host.getUsage())
            exit(1)

        # DiamondCollector Parameters
        self.obs = None
        self.prev_position = np.array([0, 0])
        self.environment = XMLenv(
            self.max_episode_steps, self.world_size, self.obs_size, flat_word=self.flatland)

        #self.allow_break_action = False
        self.episode_step = 0
        self.episode_return = 0
        self.returns = []
        self.steps = []

        self.episode_dist_return = 0
        self.dist_returns = []

        # for ploting the agent's trajectory
        self.coordinates = []
        self.graph_num = 0

    def reset(self):
        """
        Resets the environment for the next episode.

        Returns
            observation: <np.array> flattened initial obseravtion
        """

        self.returns.append(self.episode_return)
        self.dist_returns.append(self.episode_dist_return)
        
        self.environment = XMLenv(
            self.max_episode_steps, self.world_size, self.obs_size, flat_word=self.flatland)
        current_step = self.steps[-1] if len(self.steps) > 0 else 0
        self.steps.append(current_step + self.episode_step)

        # Log
        if len(self.returns) > self.log_frequency + 1 and \
                len(self.returns) % self.log_frequency == 0:
            self.log_returns()
            self.log_dist_return()
            self.draw_agent_trajectory()
        

        # Reset Variables
        self.episode_dist_return = 0
        self.episode_return = 0
        self.episode_step = 0
        self.coordinates.clear()

        # Reset Malmo and the envrionment
        world_state = self.init_malmo()

        # Get Observation
        self.obs, self.prev_position = self.get_observation(world_state)

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
        #self.agent_host.sendCommand(command)
        time.sleep(.1)
        # print("STEP:",command)
        self.episode_step += 1

        # Get Observation
        world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:", error.text)
        self.obs, pos = self.get_observation(world_state)

        # Get Done
        done = not world_state.is_mission_running

        # Get Reward
        reward = 0

        # reward += (np.linalg.norm(self.prev_position - self.environment.getGoal()) - np.linalg.norm(pos - self.environment.getGoal()))*self.move_reward_scale #L2 for continuous
        reward += (np.sum(np.abs(self.prev_position - self.environment.getGoal())) - np.sum(
            np.abs(pos - self.environment.getGoal())))*self.move_reward_scale  # L1 for discrete

        self.episode_dist_return += reward
        
        reward += np.allclose(self.environment.getGoal(),
                              pos) * self.reach_end_reward
        reward += self.reward_dict[action]
        #print(reward)
        self.prev_position = pos
        self.coordinates.append(pos)

        self.episode_return += reward
        #print("reward received:",reward)

        #time.sleep(60) ####

        return self.obs, reward, done, dict()

    def get_mission_xml(self):

        return self.environment.generateWorldXML("sandstone")

    def init_malmo(self):
        """
        Initialize new malmo mission.
        """
        my_mission = MalmoPython.MissionSpec(self.get_mission_xml(), True)
        my_mission_record = MalmoPython.MissionRecordSpec()
        my_mission.requestVideo(800, 500)
        my_mission.setViewpoint(1)

        max_retries = 3
        my_clients = MalmoPython.ClientPool()
        # add Minecraft machines here as available
        my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000))

        for retry in range(max_retries):
            try:
                self.agent_host.startMission(
                    my_mission, my_clients, my_mission_record, 0, 'MoogleMap')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:", e)
                    exit(1)
                else:
                    time.sleep(2)

        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:", error.text)

        return world_state

    def get_observation(self, world_state):
        """
        Use the agent observation API to get a flattened 2 x 5 x 5 grid around the agent. 
        The agent is in the center square facing up.

        Args
            world_state: <object> current agent world state

        Returns
            observation: <np.array> the state observation
        """
        obs = np.zeros((self.obs_size*self.obs_size+2,))
        point = np.array([.5, .5])

        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            if len(world_state.errors) > 0:
                raise AssertionError('Could not load grid.')

            if world_state.number_of_observations_since_last_state > 0:
                # First we get the json from the observation API
                msg = world_state.observations[-1].text
                observations = json.loads(msg)

                # Get observation
                obs = self.obseravtion.getObservation(
                    self.environment.terrain_array, observations['XPos'], observations['ZPos'], observations['YPos'], observations['Yaw'], self.environment.getGoal(), self.debug_obs)
                point = np.array([observations['XPos'], observations['ZPos']])

                break

        return obs, point

    def draw_agent_trajectory(self):
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.scatter(
            self.environment.start_coordinate[0], self.environment.start_coordinate[1], c="b")
        ax.scatter(
            self.environment.end_coordinate[0], self.environment.end_coordinate[1], c="r")

        plt.xlim([-self.world_size/2, self.world_size/2])
        plt.ylim([-self.world_size/2, self.world_size/2])
        xpos, ypos = zip(*self.coordinates)
        ax.plot(xpos, ypos)
        plt.title('Agent coordinate')
        plt.savefig(f'./trajectory_graphs/AgentCoords_{self.graph_num}.png')
        self.graph_num += 1

    def log_returns(self):
        """
        Log the current returns as a graph and text file

        Args:
            steps (list): list of global steps after each episode
            returns (list): list of total return of each episode
        """
        #self.draw_agent_trajectory()

        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Moogle Map')
        plt.ylabel('Return')
        plt.xlabel('Steps')
        plt.savefig('returns.png')

        # plot the agent's trajectory

        with open('returns.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.returns[1:]):
                f.write("{}\t{}\n".format(step, value))


    def log_dist_return(self):
        
        box = np.ones(self.log_frequency) / self.log_frequency
        returns_smooth = np.convolve(self.dist_returns[1:], box, mode='same')
        plt.clf()
        plt.plot(self.steps[1:], returns_smooth)
        plt.title('Moogle Map')
        plt.ylabel('Distance Value')
        plt.xlabel('Steps')
        plt.savefig('returns_dist.png')

        # plot the agent's trajectory

        with open('returns_dist.txt', 'w') as f:
            for step, value in zip(self.steps[1:], self.dist_returns[1:]):
                f.write("{}\t{}\n".format(step, value))


if __name__ == '__main__':
    ray.init()
    trainer = ppo.PPOTrainer(env=MoogleMap, config={
        'env_config': {},           # No environment parameters to configure
        'framework': 'torch',       # Use pyotrch instead of tensorflow
        'num_gpus': 0,              # We aren't using GPUs
        'num_workers': 0            # We aren't using parallelism
    })

    while True:
        print(trainer.train())

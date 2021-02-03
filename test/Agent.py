try:
    from malmo import MalmoPython
except:
    import MalmoPython


class Agent:
    def __init__(self):
        self.agent_host = MalmoPython.AgentHost()
        self.action_dict = {
            0: 'move 1',
            1: 'turn 1',
            2: 'turn -1',
            3: 'jump 1'
        }
        self.episode_step = 0
        self.action_space = Discrete(len(self.action_dict))

    def policy_gradient(self):
        pass

    def reward(self):
        #get to destination give u 
        pass

    def train(self):
        pass
    
    def step(self, action):

        command = self.action_dict[action]
        self.agent_host.sendCommand(command)
        time.sleep(.2)
        self.episode_step += 1
        



    



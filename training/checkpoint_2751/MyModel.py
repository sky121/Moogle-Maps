import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2 
import numpy as np
class MyModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kargs):
        TorchModelV2.__init__(self, *args, **kargs)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3)

        '''
        self.dense_layer_1 = nn.Linear(32*5*5+2, 32)
        self.dense_layer_2 = nn.Linear(32,16)
        '''
        self.fc1 = nn.Linear(8*49, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(34, 8)

        self.policy_layer = nn.Linear(8, 4)
        self.value_layer = nn.Linear(8, 1)
        
        self.distance = None
        self.value = None

    def forward(self, input_dict, state, seq_lens):

        #print(input_dict)
        x = input_dict['obs']
        x = x.type(torch.float32)
        #print(x)
        #print(x.size())
        self.distance = x[:,:2]
        x = x[:,2:]
        x = x.reshape(x.shape[0],1,9, 9)


        x = torch.tanh(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.smax(x)

        #print(x.shape)

        x = x.flatten(start_dim=1)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

     
        x = torch.cat((torch.tanh(self.distance), x), 1)

        x = F.relu(self.fc3(x))

        policy = F.log_softmax(self.policy_layer(x), dim=1)
        self.value = self.value_layer(x)

        return policy, state


    def value_function(self):
        return self.value.squeeze(1)
        #return self.value
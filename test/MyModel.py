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

        
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3)


        self.dense_layer = nn.Linear(32*11*11+2, 64)
        self.policy_layer = nn.Linear(64, 4)
        self.value_layer = nn.Linear(64, 1)
        
        self.distance = None
        self.value = None

    def forward(self, input_dict, state, seq_lens):

        #print(input_dict)
        x = input_dict['obs']
        #print(x)
        #print(x.size())
        self.distance = x[:,:2]
        x = x[:,2:]
        x = x.reshape(x.shape[0],1,15, 15)
        x = x.type(torch.float32)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten(start_dim=1)
        x = torch.cat((self.distance,x), 1)

        x = F.relu(self.dense_layer(x))
        policy = self.policy_layer(x)
        self.value = self.value_layer(x)
        
        

        return policy, state

    def value_function(self):
        return self.value.squeeze(1)
        #return self.value
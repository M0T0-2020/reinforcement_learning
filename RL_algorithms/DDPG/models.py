import torch
from torch import nn
from torch import optim
from torch import cuda

import numpy as np

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

def fan_in_uniform_init(tensor):
    """
    Utility function for initializing actor and critic
    fan_in_uniform_init(fc1.weight)
    """
    fan_in = tensor.size(-1)
    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

def init_weights(m):
    if type(m) == nn.Linear:
        fan_in_uniform_init(m.weight)
        fan_in_uniform_init(m.bias)


class Critic(torch.nn.Module):
    def __init__(self, input_size, action_dim, hidden_layers):
        super().__init__()
        self.action_dim = action_dim

        self.fc_1 = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.LeakyReLU(inplace=False)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_layers[0]+self.action_dim, hidden_layers[1]),
            nn.LeakyReLU(inplace=False)
        )

        self.fc_3 = nn.Linear(hidden_layers[1], 1)

        self.fc_1.apply(init_weights)
        self.fc_2.apply(init_weights)
        nn.init.uniform_(self.fc_3.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.fc_3.bias, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
            
    def forward(self, state, action):
        x = self.fc_1(state)
        action = action.view(-1, self.action_dim)
        x = torch.cat((x, action), dim=1)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x


class Actor(torch.nn.Module):
    def __init__(self, input_size, action_dim, hidden_layers):
        super().__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.LeakyReLU(inplace=False),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.LeakyReLU(inplace=False),
        )
        self.fc_2 = nn.Linear(hidden_layers[1], action_dim)
        
        self.fc_1.apply(init_weights)
        nn.init.uniform_(self.fc_2.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.fc_2.bias, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)
        #x = torch.tanh(x)
        x = nn.Sigmoid()(x)
        return x
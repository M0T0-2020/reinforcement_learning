import torch
from torch import nn
from torch import optim
from torch import cuda

class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.LeakyReLU(),

            nn.Linear(hidden_layers[1], hidden_layers[1]),
            nn.LeakyReLU()
        )  

        self.fc_policy = nn.Linear(hidden_layers[1], output_size)
        self.fc_value = nn.Linear(hidden_layers[1], 1)
    def forward(self, x):
        x = self.fc(x)
        p = self.fc_policy(x)
        v = self.fc_value(x)
        return {'policy':p, 'value':v}
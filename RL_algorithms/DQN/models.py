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
            nn.LeakyReLU(),

            nn.Linear(hidden_layers[1], output_size)
        )  
    def forward(self, x):
        x = self.fc(x)
        return x
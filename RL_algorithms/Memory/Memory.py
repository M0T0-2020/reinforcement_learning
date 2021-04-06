import os, sys 
import pandas as pd
import numpy as np

import random
from collections import namedtuple, deque

class Replay_Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.position = 0
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *data):
        data = self.transition(*data)
        self.memory.appendleft(data)
            
    def sample(self, batch_size):
        data = {'transaction':random.sample(self.memory, batch_size)}
        return data

    def __len__(self):
        return len(self.memory)
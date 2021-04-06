import os, sys 
import pandas as pd
import numpy as np
import queue
import copy
import random
from collections import namedtuple, deque

class ProportionalReplayMemory:
    def __init__(self, memory_size, alpha=0.6, beta=0.4):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.priority = deque(maxlen=self.memory_size)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-12
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *data):
        data = self.transition(*data)
        MaxPriority = max(self.priority) if len(self.priority)>0 else 1
        self.memory.appendleft(data)
        self.priority.appendleft(MaxPriority)
            
    def sample(self, batch_size):
        PrioritySum = sum([p**self.alpha for p in self.priority])
        prob = [p**self.alpha/PrioritySum for p in self.priority]

        index = [i for i in range(len(self.memory))]
        index = np.random.choice(index, batch_size, p=prob, replace=False)
        transaction = [self.memory[i] for i in index]
        sample_weight = [1/(batch_size*prob[i]) for i in index]
        sample_weight = [w**self.beta for w in sample_weight]
        sample_weight = [w/max(sample_weight) for w in sample_weight]
        data = {'transaction':transaction, "weight":sample_weight, "index":index}
        return data 

    def updatePriority(self, index, loss):
        for i, idx in enumerate(index):
            self.priority[idx]=float(loss[i])+self.epsilon
        #prioritryの順序index
        sorted_index = np.argsort(self.priority)[::-1]
        self.memory = deque([self.memory[idx] for idx in sorted_index], maxlen=self.memory_size)
        self.priority = deque([self.priority[idx] for idx in sorted_index], maxlen=self.memory_size)

    def __len__(self):
        return len(self.memory)


class RankReplayMemory:
    def __init__(self, memory_size, alpha=0.6, beta=0.4):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.priority = deque(maxlen=self.memory_size)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *data):
        data = self.transition(*data)
        MaxPriority = max(self.priority)
        self.memory.appendleft(data)
        self.priority.appendleft(MaxPriority)
            
    def sample(self, batch_size):
        PrioritySum = sum([p**self.alpha for p in self.priority])
        prob = [p**self.alpha/PrioritySum for p in self.priority]

        index = [i for i in range(len(self.memory))]
        index = np.random.choice(index, batch_size, p=prob, replace=False)
        transaction = [self.memory[i] for i in index]
        sample_weight = [1/(batch_size*prob[i]) for i in index]
        sample_weight = [w**self.beta for w in sample_weight]
        sample_weight = [w/max(sample_weight) for w in sample_weight]
        data = {'transaction':transaction, "weight":sample_weight, "index":index}
        return data 

    def updatePriority(self, index, loss):
        for idx in index:
            self.priority[idx]=float(loss[idx])
        #prioritryの順序index
        sorted_index = np.argsort(self.priority)[::-1]
        self.memory = deque([self.memory[idx] for idx in sorted_index], maxlen=self.memory_size)
        self.priority = deque([self.priority[idx] for idx in sorted_index], maxlen=self.memory_size)

    def __len__(self):
        return len(self.memory)
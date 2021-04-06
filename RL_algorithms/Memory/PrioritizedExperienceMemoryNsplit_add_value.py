import os, sys 
import pandas as pd
import numpy as np
import queue
import copy
import random
from collections import namedtuple, deque

class ProportionalReplayNsplitMemory:
    r"""
    alpha -> uniform sampling
    """
    def __init__(self, memory_size, n=20, alpha=0.6, beta=0.4, maxweight=1):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.priority = deque(maxlen=self.memory_size)

        self.n = n
        self.TemporalStateMemory = deque(maxlen=self.n-1)
        self.TemporalActionMemory = deque(maxlen=self.n-1)
        self.TemporalNextStateMemory = deque(maxlen=self.n-1)
        self.TemporalRewardMemory = deque(maxlen=self.n-1)
        self.TemporalValueMemory = deque(maxlen=self.n-1)

        self.alpha = alpha
        self.beta = beta
        self.maxweight = maxweight
        self.epsilon = 1e-12
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'value'))


    def ResetTemporalMemory(self):
        self.TemporalStateMemory = deque(maxlen=self.n-1)
        self.TemporalActionMemory = deque(maxlen=self.n-1)
        self.TemporalNextStateMemory = deque(maxlen=self.n-1)
        self.TemporalRewardMemory = deque(maxlen=self.n-1)
        self.TemporalValueMemory = deque(maxlen=self.n-1)

    def push(self, *data):
        data = self.transition(*data)
        self.TemporalStateMemory.appendleft(data.state)
        self.TemporalActionMemory.appendleft(data.action)
        self.TemporalNextStateMemory.appendleft(data.next_state)
        self.TemporalRewardMemory.appendleft(data.reward)
        self.TemporalValueMemory.appendleft(data.value)

        if len(self.TemporalNextStateMemory)==self.TemporalNextStateMemory.maxlen:
            data = self.transition(
                self.TemporalStateMemory[-1],
                self.TemporalActionMemory[-1],
                self.TemporalNextStateMemory[0],
                list(self.TemporalRewardMemory),
                list(self.TemporalValueMemory),
            )
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
        if self.maxweight is not None:
            sample_weight = [min(self.maxweight, w) for w in sample_weight]

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


class RankReplayNsplitMemory:
    r"""
    alpha -> uniform sampling
    """
    def __init__(self, memory_size, n=20, alpha=0.6, beta=0.4, maxweight=1):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.priority = deque(maxlen=self.memory_size)

        self.n = n
        self.TemporalStateMemory = deque(maxlen=self.n-1)
        self.TemporalActionMemory = deque(maxlen=self.n-1)
        self.TemporalNextStateMemory = deque(maxlen=self.n-1)
        self.TemporalRewardMemory = deque(maxlen=self.n-1)
        self.TemporalValueMemory = deque(maxlen=self.n-1)

        self.alpha = alpha
        self.beta = beta
        self.maxweight = maxweight
        self.epsilon = 1e-12
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'value'))


    def ResetTemporalMemory(self):
        self.TemporalStateMemory = deque(maxlen=self.n-1)
        self.TemporalActionMemory = deque(maxlen=self.n-1)
        self.TemporalNextStateMemory = deque(maxlen=self.n-1)
        self.TemporalRewardMemory = deque(maxlen=self.n-1)
        self.TemporalValueMemory = deque(maxlen=self.n-1)

    def push(self, *data):
        data = self.transition(*data)
        self.TemporalStateMemory.appendleft(data.state)
        self.TemporalActionMemory.appendleft(data.action)
        self.TemporalNextStateMemory.appendleft(data.next_state)
        self.TemporalRewardMemory.appendleft(data.reward)
        self.TemporalValueMemory.appendleft(data.value)

        if len(self.TemporalNextStateMemory)==self.TemporalNextStateMemory.maxlen:
            data = self.transition(
                self.TemporalStateMemory[-1],
                self.TemporalActionMemory[-1],
                self.TemporalNextStateMemory[0],
                list(self.TemporalRewardMemory),
                list(self.TemporalValueMemory),
            )
            MaxPriority = max(self.priority) if len(self.priority)>0 else 1
            self.memory.appendleft(data)
            self.priority.appendleft(MaxPriority)
            
    def sample(self, batch_size):
        rank = [1/(1+i) for i in range(len(self.priority))]
        PrioritySum = sum([p**self.alpha for p in rank])
        prob = [p**self.alpha/PrioritySum for p in rank]

        index = [i for i in range(len(self.priority))]
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
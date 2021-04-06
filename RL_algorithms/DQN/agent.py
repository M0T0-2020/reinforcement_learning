import torch
from torch import nn
from torch import optim
from torch import cuda

import os, sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import random
from collections import namedtuple

from models import DQN
from optimize_util import optimize_nn, init_opmtimistic, optimize_nn_nSplit
sys.path.append('/Users/kanoumotoharu/Desktop/machine_learning/強化学習/実験コード/RL_algorithms/batcher/')
from Batcher import Batcher

class Agent:
    def __init__(self, action_space, nSplit=False, lr=5e-4, gamma=0.99, weight_decay=1e-4,
                 eps_start=0.99, eps_end=0.25, eps_decay=10000, batch=64, 
                 input_size=2, hidden_layers=[16,32], target_update=50, memory_size=50000):
        # lr
        self.lr = lr
        # discount rate
        self.gamma = gamma
        self.action_space = action_space
        self.batch = batch
        self.steps = 0
        self.gamma = gamma 
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        #self.replay_memory = Replay_Memory(memory_size=memory_size)
        self.batcher = Batcher()
        self.target_update = target_update
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        #self.environment = environment

        if nSplit:
            self.optimize_nn = optimize_nn_nSplit
        else:
            self.optimize_nn = optimize_nn

        self.target_net = DQN(input_size=input_size, output_size=self.action_space, hidden_layers=hidden_layers)
        self.policy_net = DQN(input_size=input_size, output_size=self.action_space, hidden_layers=hidden_layers)
            
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=weight_decay)
        
    def show_eps_threshold(self):
        x = np.arange(0,100000)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * x / self.eps_decay)
        plt.figure(figsize=(30,10))
        plt.plot(eps_threshold)
        plt.show()

    def init_param_optim(self, epoch_num, env, environment):
        loss = init_opmtimistic(epoch_num, env, environment, self.policy_net, self.target_net, self.action_space, self.device)
        return loss        
    
    def update_agent(self, replay_memory):
        if len(replay_memory)<3:
            return None
        else:
            batch_size = min(len(replay_memory), self.batch)
            batch = self.batcher.make_batch(replay_memory, batch_size)
            loss, priorityloss = self.optimize_nn(batch, self.policy_net, self.target_net, self.policy_optimizer, self.gamma, self.device)
            if (loss is not None) and ('weight' in batch.keys()):
                replay_memory.updatePriority(batch['index'], priorityloss)
                
            return loss

    def get_greedy_action(self, next_state):    
        actions = self.policy_net(next_state).detach().cpu().numpy()
        actions = [i for i, a in enumerate(actions) if a==actions.max()]
        action = np.random.choice(actions)
        return action
        
    def get_action(self, next_state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * self.steps / self.eps_decay)
        #eps_threshold = self.epsilon
        if np.random.rand() > eps_threshold:
            action = self.get_greedy_action(next_state)
        else:
            action = np.random.randint(self.action_space)
        self.steps+=1

        if self.steps%self.target_update==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return action
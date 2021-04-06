import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.distributions import Categorical

import os, sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import random
from collections import namedtuple

from models import Model
from optimize_util import init_opmtimistic, optimize_nn_nSplit, optimize_nn_nSplit_sam
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Batcher.Batcher import Batcher
from optimizer.sam import SAM
from Noise.noise import AdaptiveParamNoiseSpec

class Agent:
    def __init__(self, action_space, nSplit=False, lr=5e-4, gamma=0.99, weight_decay=1e-4,
                 eps_start=0.99, eps_end=0.25, eps_decay=10000, batch=64, 
                 input_size=2, hidden_layers=[16,32], target_update=50, memory_size=50000, sam=True,
                 initial_stddev=0.2, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        
        self.device = 'cuda' if cuda.is_available() else 'cpu'
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
        #self.environment = environment

        if sam:
            self.optimize_nn = optimize_nn_nSplit_sam
        else:
            self.optimize_nn = optimize_nn_nSplit

        self.noise = AdaptiveParamNoiseSpec(initial_stddev=initial_stddev,
        desired_action_stddev=desired_action_stddev, adaptation_coefficient=adaptation_coefficient)

        self.model = Model(input_size=input_size, output_size=self.action_space, hidden_layers=hidden_layers)
        self.perturbed_model = Model(input_size=input_size, output_size=self.action_space, hidden_layers=hidden_layers)
        self.sam = sam
        if sam:
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=self.lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def ddpg_distance_metric(self, batch):
        transaction = batch['transaction']
        state = torch.FloatTensor([data.state for data in transaction]).to(self.device)
        self.model = self.model.to(self.device)
        self.perturbed_model = self.perturbed_model.to(self.device)

        out_1 = self.model(state)
        out_2 = self.perturbed_model(state)
        distance=0
        for key in out_1.keys():
            diff = out_1[key].detach().cpu().numpy()-out_2[key].detach().cpu().numpy()
            mean_diff = np.mean(diff**2, axis=0)
            distance += np.sqrt(np.mean(mean_diff))

        self.model = self.model.to("cpu")
        self.perturbed_model = self.perturbed_model.to("cpu")
        return distance/len(out_1.keys())

    def init_param_optim(self, epoch_num, env, environment):
        loss = init_opmtimistic(epoch_num, env, environment, self.model, self.action_space, self.device)
        self.perturbed_model.load_state_dict(self.model.state_dict())
        return loss        
    
    def update_agent(self, replay_memory):
        if len(replay_memory)<3:
            return None
        else:
            batch_size = min(len(replay_memory), self.batch)
            batch = self.batcher.make_batch(replay_memory, batch_size)
            loss, priorityloss = self.optimize_nn(batch, self.model, self.optimizer, self.gamma, self.device, self.steps)
            if (loss is not None) and ('weight' in batch.keys()):
                replay_memory.updatePriority(batch['index'], priorityloss)

            distance = self.ddpg_distance_metric(batch)
            self.noise.adapt(distance)
            return loss

    def get_action(self, next_state, perturbed=True):  
        if not perturbed:
            actions = self.model(next_state)["policy"]
            m = Categorical(logits=actions)
            action = m.sample()[0]
            return int(action)

        self.perturbed_model.load_state_dict(self.model.state_dict())
        params = self.perturbed_model.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            noise = torch.normal(mean=0, std=self.noise.current_stddev, size=param.shape)
            param += noise
        actions = self.perturbed_model(next_state)["policy"]
        m = Categorical(logits=actions)
        action = m.sample()[0]
        return int(action)
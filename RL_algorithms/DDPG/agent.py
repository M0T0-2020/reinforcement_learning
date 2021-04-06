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

from noise import NormalActionNoise, AdaptiveParamNoiseSpec, ddpg_distance_metric
from models import Critic, Actor
from optimize_util import optimize_critic, optimize_actor, init_opmtimistic
sys.path.append('/Users/kanoumotoharu/Desktop/machine_learning/強化学習/実験コード/RL_algorithms/batcher/')
from Batcher import Batcher
sys.path.append('/Users/kanoumotoharu/Desktop/machine_learning/強化学習/実験コード/RL_algorithms/Memory/')
from Memory import Replay_Memory

class Agent:
    def __init__(self, action_space, output_size, environment, critic_lr=1e-4, actor_lr=2e-4, gamma=0.99, tau=0.01,
                 action_noise=None, action_noise_std=0.2,
                 param_noise=None,
                 eps_start=0.99, eps_end=0.1, eps_decay=10000, batch=64, 
                 input_size=2, hidden_layers=[10,20], target_update=5, memory_size=50000):
        # lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        # discount rate
        self.gamma = gamma
        self.tau = tau
        self.action_space = int(action_space)
        self.output_size = output_size
        self.batch = batch
        self.steps = 0
        self.gamma = gamma 
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.action_noise_std = action_noise_std

        self.replay_memory = Replay_Memory(memory_size=memory_size)
        self.batcher = Batcher()
        self.target_update = target_update
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.environment = environment
        self.action_noise = action_noise(0,self.action_noise_std) if action_noise is not None else action_noise
        self.param_noise = param_noise

        self.target_critic = Critic(input_size=input_size, action_dim=self.output_size, hidden_layers=hidden_layers)
        self.target_actor = Actor(input_size=input_size, action_dim=self.output_size, hidden_layers=hidden_layers)

        self.policy_critic = Critic(input_size=input_size, action_dim=self.output_size, hidden_layers=hidden_layers)
        self.policy_actor = Actor(input_size=input_size, action_dim=self.output_size, hidden_layers=hidden_layers)

        self.perturb_actor = Actor(input_size=input_size, action_dim=self.output_size, hidden_layers=hidden_layers)
            
        self.critic_optimizer = optim.Adam(self.policy_critic.parameters(), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(self.policy_actor.parameters(), lr=self.actor_lr)

        self.sheduler = False
        self.critic_optimizer_sheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9999)
        self.actor_optimizer_sheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.9999)

    def perturb_actor_parameters(self):
        #https://github.com/ikostrikov/pytorch-ddpg-naf
        #Apply parameter noise to actor model, for exploration
        self.hard_update(self.perturb_actor, self.policy_actor)
        params = self.perturb_actor.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            noise = torch.normal(mean=0, std=self.param_noise.current_stddev, size=param.shape)
            param += noise

    def update_ParamNoize(self):
        batch_size = min(len(self.replay_memory), 256)
        if batch_size<3:
            return None

        batch = self.replay_memory.sample(batch_size)
        ddpg_dist = ddpg_distance_metric(batch, self.policy_actor, self.perturb_actor, self.device)
        self.param_noise.adapt(ddpg_dist)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def init_param_optim(self, epoch_num, env):
        loss = init_opmtimistic(epoch_num, env, self.environment,
                    self.policy_critic, self.target_critic, self.policy_actor, self.output_size, self.device)
        return loss
    
    def update_agent(self, env, obs, next_obs, reward, action):
        state, action, next_state, reward = self.environment.get_data(env, obs, next_obs, reward, action)
        self.replay_memory.push(state, action, next_state, reward)
        
        if self.steps<=3:
            return None, None
        else:
            batch_size = min(len(self.replay_memory), self.batch)
            batch = self.batcher.make_batch(self.replay_memory, batch_size)
            critic_loss = optimize_critic(batch, self.policy_critic, self.target_critic, 
                            self.policy_actor, self.target_actor, self.critic_optimizer, 
                            self.gamma, self.device)

            actor_loss = optimize_actor(batch, self.policy_critic,  self.policy_actor, 
                            self.actor_optimizer, self.device)
            
            if self.sheduler:
                self.critic_optimizer_sheduler.step()
                self.actor_optimizer_sheduler.step()

            return critic_loss, actor_loss

    def take_action(self, env, next_state):
        next_state = self.environment.get_state(env, next_state)
        next_state = torch.FloatTensor(next_state)
        
        if self.param_noise is not None:
            self.update_ParamNoize()
            self.perturb_actor_parameters()
            actions = self.perturb_actor(next_state).detach().cpu().numpy()
            action = actions[0]
        else:
            actions = self.policy_actor(next_state).detach().cpu().numpy()
            action = actions[0]
            

        if self.action_noise is not None:
            action+=self.action_noise()

        action = np.clip(action, 0, 1-1e-14)
        return action
        
    def get_action(self, env, next_state):
        action = self.take_action(env, next_state)
        self.steps+=1
        if self.steps%self.target_update==0:
        #if True:
            self.soft_update(self.target_critic, self.policy_critic)
            self.soft_update(self.target_actor, self.policy_actor)
        return int(self.action_space*action)
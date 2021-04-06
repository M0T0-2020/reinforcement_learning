import numpy as np
import torch
from math import sqrt

"""
episode_transitions = memory.memory[memory.position-t:memory.position]
states = torch.cat([transition[0] for transition in episode_transitions], 0)
unperturbed_actions = agent.select_action(states, None, None)
perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
param_noise.adapt(ddpg_dist)


def perturb_actor_parameters(self, param_noise):
    #Apply parameter noise to actor model, for exploration
    hard_update(self.actor_perturbed, self.actor)
    params = self.actor_perturbed.state_dict()
    for name in params:
        if 'ln' in name: 
            pass 
        param = params[name]
        noise = torch.normal(mean=0, std=param_noise.current_stddev, size=param.shape)
        param += noise

From OpenAI Baselines:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
"""
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.2, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise, 
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

def ddpg_distance_metric(batch, policy_actor, perturb_actor, device):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    policy_actor = policy_actor.to(device)
    state = torch.FloatTensor([data.state for data in batch]).to(device)
    action_1 = policy_actor(state).squeeze()
    action_2 = perturb_actor(state).squeeze()

    diff = action_1.detach().cpu().numpy()-action_2.detach().cpu().numpy()
    mean_diff = np.mean(diff**2, axis=0)
    distance = sqrt(np.mean(mean_diff))
    policy_actor = policy_actor.to('cpu')
    perturb_actor = perturb_actor.to('cpu')
    return distance

"""
def ddpg_distance_metric(batch, policy_actor, perturb_actor, device):
    #Compute "distance" between actions taken by two policies at the same states
    #Expects numpy arrays
    policy_actor = policy_actor.to(device)
    state = torch.FloatTensor([data.state for data in batch]).to(device)
    action_1 = policy_actor(state).squeeze()
    action_2 = torch.FloatTensor([data.action for data in batch]).to(device)

    diff = action_1.detach().numpy()-action_2.detach().numpy()
    mean_diff = np.mean(diff**2, axis=0)
    distance = sqrt(np.mean(mean_diff))
    policy_actor = policy_actor.to('cpu')
    return distance
"""



class ActionNoise(object):
    def reset(self):
        pass

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
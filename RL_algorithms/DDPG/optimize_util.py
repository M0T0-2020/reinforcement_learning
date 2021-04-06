import torch
from torch import nn
from torch import optim
from torch import cuda

def init_opmtimistic(epoch_num, env, environment, policy_critic, target_critic, policy_actor, output_size, device):
    b_size = 2*9
    data = [
            environment.get_state(env, env.reset()) for _  in range(b_size)
        ]
    policy_critic = policy_critic.to(device)
    policy_actor = policy_actor.to(device)
    tmp_opt = optim.Adam(policy_critic.parameters(), lr=1e-2)
    data = torch.FloatTensor(data).to(device)
    loss_list = []
    
    action = policy_actor(data).detach()
    for i in range(epoch_num):
        pseudo_action = torch.normal(0, 1, size=action.size(), device=device)
        loss = -(policy_critic(data, pseudo_action)).mean()
        tmp_opt.zero_grad()
        loss.backward()
        tmp_opt.step()
        loss_list.append(loss.item())
    policy_critic = policy_critic.to('cpu')
    policy_actor = policy_actor.to('cpu')
    target_critic.load_state_dict(policy_critic.state_dict())

    return loss_list


def optimize_critic(batch, policy_critic, target_critic, policy_actor, target_actor, critic_optimizer, gamma, device):
    state = torch.FloatTensor([data.state for data in batch]).to(device)
    action = torch.FloatTensor([data.action for data in batch]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in batch]).to(device)
    reward = torch.FloatTensor([data.reward for data in batch]).to(device)

    policy_critic = policy_critic.to(device)        
    policy_critic = policy_critic.to(device)        
    policy_actor = policy_actor.to(device)        
    target_actor = target_actor.to(device)
    
    target_action = target_actor(next_state).detach()
    #価値関数の更新にはノイズ混じりの方策を使う
    #target_action += torch.normal(0,0.1, size=target_action.size(), device=device)
    next_state_values = target_critic(next_state, target_action)
    next_state_values = next_state_values
    expected_state_action_values = (next_state_values * gamma) + reward.unsqueeze(1)
    expected_state_action_values = expected_state_action_values.detach()

    
    action = action.detach()
    #価値関数の更新にはノイズ混じりの方策を使う
    #action += torch.normal(0,0.1, size=action.size(), device=device)
    state_action_values = policy_critic(state, action)
    state_action_values = state_action_values

    loss_critic = nn.MSELoss()(state_action_values, expected_state_action_values)

    critic_optimizer.zero_grad()
    loss_critic.backward()

    for param in policy_critic.parameters():
        #勾配の値を直接変更する
        #https://pytorch.org/docs/master/torch.html#torch.clamp
        param.grad.data.clamp_(-1, 1)
    critic_optimizer.step()

    policy_critic = policy_critic.to('cpu')        
    policy_critic = policy_critic.to('cpu')        
    policy_actor = policy_actor.to('cpu')        
    target_actor = target_actor.to('cpu')
    return loss_critic.item()

def optimize_actor(batch, policy_critic, policy_actor, actor_optimizer, device):
    state = torch.FloatTensor([data.state for data in batch]).to(device)

    policy_critic = policy_critic.to(device)        
    policy_actor = policy_actor.to(device)
    
    actor_optimizer.zero_grad()
    actor_loss = -policy_critic(state, policy_actor(state))
    actor_loss = actor_loss.mean()
    actor_loss.backward()
    
    for param in policy_actor.parameters():
        #勾配の値を直接変更する
        #https://pytorch.org/docs/master/torch.html#torch.clamp
        param.grad.data.clamp_(-1, 1)
    
    actor_optimizer.step()
    
    policy_critic = policy_critic.to('cpu')
    policy_actor = policy_actor.to('cpu')
    return actor_loss.item()
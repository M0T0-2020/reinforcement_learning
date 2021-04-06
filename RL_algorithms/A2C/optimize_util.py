import torch
from torch import nn
from torch import optim
from torch import cuda
import torch.nn.functional as F
from torch.distributions import Categorical

def init_opmtimistic(epoch_num, env, environment, model, action_num, device):
    b_size = 128
    data = [
            environment.get_state(env, env.reset()) for _  in range(b_size)
        ]
    model = model.to(device)
    tmp_opt = optim.Adam(model.parameters(), lr=1e-2)
    data = torch.FloatTensor(data).to(device)
    size = (b_size, action_num)
    label = 10*torch.ones(size=size).to(device)
    criterion = nn.MSELoss()
    loss_list = []
    for i in range(epoch_num):
        _label = label + 0.3*torch.rand(size=size).to(device)
        _label = nn.Softmax()(_label)
        value = model(data)['policy']
        loss = criterion(value, _label)
        tmp_opt.zero_grad()
        loss.backward()
        tmp_opt.step()
        loss_list.append(loss.item())
    model = model.to('cpu')

    return loss_list


def calu_loss(out, next_out, action, reward, gamma):
    losses = {}
    policy = out['policy']
    value = out['value']
    policy = Categorical(logits=policy)

    next_value = next_out['value'].detach()
    
    log_probs = policy.logits.gather(1, action.unsqueeze(1))
    expected_state_action_values = (next_value * gamma) + reward.unsqueeze(1)
    advantages = expected_state_action_values - value

    #print(f"log_probs   {log_probs}")
    #print(f"advantages  {advantages}")
    
    actor_loss = -(log_probs * advantages.detach()).squeeze(1)
    critic_loss = (advantages**2).squeeze(1)
    entropy = policy.entropy()
    
    #print(f"actor_loss   {actor_loss}")
    #print(f"critic_loss  {critic_loss}")
    
    losses['actor_loss'] = actor_loss
    losses['critic_loss'] = critic_loss
    losses['entropy'] = entropy
    
    return losses


def optimize_nn(batch, model, optimizer, gamma, device):
    transaction = batch['transaction']
    state = torch.FloatTensor([data.state for data in transaction]).to(device)
    action = torch.LongTensor([data.action for data in transaction]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in transaction]).to(device)
    reward = torch.FloatTensor([data.reward for data in transaction]).to(device)

    model.train()

    model = model.to(device)        
    
    out = model(state)
    next_out = model(next_state)

    losses = calu_loss(out, next_out, action, reward, gamma)

    loss_priority = losses['critic_loss']
    if 'weight' in batch.keys():
        weight = torch.FloatTensor(batch['weight']).to(device)
    else:
        weight = 1

    loss = 0
    coef = {'actor_loss':1,'critic_loss':1, 'entropy':0.15}
    for key, loss_value in losses.items():
        loss += coef[key]*(weight*loss_value).mean()


    optimizer.zero_grad()
    loss.backward()
    # grad clipping
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    # パラメータの更新
    optimizer.step()
    
    loss_priority = loss_priority.detach().cpu().tolist()

    model = model.to('cpu')
    model.eval()
    return loss.item(), loss_priority


def calu_loss_nSplit(out, next_out, action, reward, gamma):
    losses = {}
    policy = out['policy']
    value = out['value']

    policy = Categorical(logits=policy)

    expected_state_action_values = next_out['value'].detach()
    for i in range(reward.size(1)):
        expected_state_action_values = expected_state_action_values * gamma + reward[:,i].unsqueeze(1)
    
    log_probs = policy.logits.gather(1, action.unsqueeze(1))
    advantages = expected_state_action_values - value

    #print(f"log_probs   {log_probs}")
    #print(f"advantages  {advantages}")
    
    actor_loss = -(log_probs * advantages.detach()).squeeze(1)
    critic_loss = (advantages**2).squeeze(1)
    entropy = policy.entropy()
    
    #print(f"actor_loss   {actor_loss}")
    #print(f"critic_loss  {critic_loss}")
    
    losses['actor_loss'] = actor_loss
    losses['critic_loss'] = critic_loss
    losses['entropy'] = entropy
    
    return losses

def optimize_nn_nSplit(batch, model, optimizer, gamma, device):
    transaction = batch['transaction']
    state = torch.FloatTensor([data.state for data in transaction]).to(device)
    action = torch.LongTensor([data.action for data in transaction]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in transaction]).to(device)
    reward = torch.FloatTensor([data.reward for data in transaction]).to(device)

    model.train()
    model = model.to(device)

    out = model(state)
    next_out = model(next_state)

    losses = calu_loss_nSplit(out, next_out, action, reward, gamma)

    loss_priority = losses['critic_loss']    
    if 'weight' in batch.keys():
        weight = torch.FloatTensor(batch['weight']).to(device)
    else:
        weight = 1
    
    loss = 0
    coef = {'actor_loss':1,'critic_loss':1, 'entropy':0.15}
    for key, loss_value in losses.items():
        loss += coef[key]*(weight*loss_value).mean()

    optimizer.zero_grad()
    loss.backward()
    # grad clipping
    #for param in model.parameters():
     #   param.grad.data.clamp_(-1, 1)
    # パラメータの更新
    optimizer.step()
    
    loss_priority = loss_priority.detach().cpu().tolist()

    model = model.to('cpu')
    model.eval()
    return loss.item(), loss_priority
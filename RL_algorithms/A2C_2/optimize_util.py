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
    label = 50*torch.ones(size=size).to(device)
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

def calu_loss_nSplit(out, next_out, action, values, rewards, gamma):
    losses = {}
    policies = out['policy']
    out_values = out['value']
    returns = out['return']

    policies = Categorical(logits=policies)

    expected_state_action_values = next_out['value'].detach()
    expected_state_action_returns = next_out['return'].detach()

    for i in range(rewards.size(1)):
        expected_state_action_values = expected_state_action_values * gamma + rewards[:,i].unsqueeze(1)
        expected_state_action_returns = expected_state_action_returns * gamma + values[:,i].unsqueeze(1)
    
    log_probs = policies.logits.gather(1, action.unsqueeze(1))
    value_advantages = expected_state_action_values - out_values
    return_advantages = expected_state_action_returns - returns
    
    advantages = 2*value_advantages + return_advantages
    #print(f"log_probs   {log_probs}")
    #print(f"advantages  {advantages}")
    
    actor_loss = -(log_probs * advantages.detach()).squeeze(1)
    critic_value_loss = F.mse_loss(out_values, expected_state_action_values, reduction='none').squeeze(1)
    critic_return_loss = F.smooth_l1_loss(returns, expected_state_action_returns, reduction='none').squeeze(1)
    entropy = -policies.entropy()
    
    #print(f"actor_loss   {actor_loss}")
    #print(f"critic_loss  {critic_loss}")
    
    losses['actor_loss'] = actor_loss
    losses['critic_value_loss'] = critic_value_loss
    losses['critic_return_loss'] = critic_return_loss
    losses['entropy'] = entropy
    
    return losses

def optimize_nn_nSplit(batch, model, optimizer, gamma, device, step=0):
    transaction = batch['transaction']
    state = torch.FloatTensor([data.state for data in transaction]).to(device)
    action = torch.LongTensor([data.action for data in transaction]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in transaction]).to(device)
    values = torch.FloatTensor([data.value for data in transaction]).to(device)
    rewards = torch.FloatTensor([data.reward for data in transaction]).to(device)

    model.train()
    model = model.to(device)

    out = model(state)
    next_out = model(next_state)
    
    losses = calu_loss_nSplit(out, next_out, action, values, rewards, gamma)
    loss_priority = 0.5*(losses['critic_value_loss']+losses["critic_return_loss"])
    if 'weight' in batch.keys():
        weight = torch.FloatTensor(batch['weight']).to(device)
    else:
        weight = 1
    
    loss = 0
    coef = {'actor_loss':1, 'critic_value_loss':1, 'critic_return_loss':1, 'entropy':0.8}
    for key, loss_value in losses.items():
        c = coef[key]
        if key=='entropy':
            c = c**step 
        loss += c*(weight*loss_value).mean()
        
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
    # grad clipping
    #for param in model.parameters():
     #   param.grad.data.clamp_(-1, 1)
    # パラメータの更新
    optimizer.step()
    
    loss_priority = loss_priority.detach().cpu().tolist()

    model = model.to('cpu')
    model.eval()
    return loss.item(), loss_priority


def calu_loss_nSplit_sam(next_out, values, rewards, gamma):
    target = {}
    expected_state_action_values = next_out['value'].detach()
    expected_state_action_returns = next_out['return'].detach()

    for i in range(rewards.size(1)):
        expected_state_action_values = expected_state_action_values * gamma + rewards[:,i].unsqueeze(1)
        expected_state_action_returns = expected_state_action_returns * gamma + values[:,i].unsqueeze(1)
    
    target['expected_state_action_values'] = expected_state_action_values
    target['expected_state_action_returns'] = expected_state_action_returns
    
    return target

def optimize_nn_nSplit_sam(batch, model, optimizer, gamma, device, step=0):
    transaction = batch['transaction']
    state = torch.FloatTensor([data.state for data in transaction]).to(device)
    action = torch.LongTensor([data.action for data in transaction]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in transaction]).to(device)
    values = torch.FloatTensor([data.value for data in transaction]).to(device)
    rewards = torch.FloatTensor([data.reward for data in transaction]).to(device)

    model.train()
    model = model.to(device)

    out = model(state)
    next_out = model(next_state)
    target = calu_loss_nSplit_sam(next_out, values, rewards, gamma)
    
    expected_state_action_values = target["expected_state_action_values"]
    expected_state_action_returns = target["expected_state_action_returns"]

    for i in range(2):
        out = model(state)
        policies = out['policy']
        out_values = out['value']
        returns = out['return']
        
        policies = Categorical(logits=policies)    
        log_probs = policies.logits.gather(1, action.unsqueeze(1))
        value_advantages = expected_state_action_values - out_values
        return_advantages = expected_state_action_returns - returns

        advantages = 2*value_advantages + return_advantages
        actor_loss = -(log_probs * advantages.detach()).squeeze(1)
        critic_value_loss = F.mse_loss(out_values, expected_state_action_values, reduction='none').squeeze(1)
        critic_return_loss = F.smooth_l1_loss(returns, expected_state_action_returns, reduction='none').squeeze(1)
        entropy = -policies.entropy()
        
        losses={}
        losses['actor_loss'] = actor_loss
        losses['critic_value_loss'] = critic_value_loss
        losses['critic_return_loss'] = critic_return_loss
        losses['entropy'] = entropy

        if 'weight' in batch.keys():
            weight = torch.FloatTensor(batch['weight']).to(device)
        else:
            weight = 1
        
        loss = 0
        coef = {'actor_loss':1, 'critic_value_loss':1, 'critic_return_loss':1, 'entropy':0.8}
        for key, loss_value in losses.items():
            c = coef[key]
            if key=='entropy':
                c = c**step 
            loss += c*(weight*loss_value).mean()
        
        if i==0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.first_step(zero_grad=True)
        if i==1:
            # second forward-backward pass
            optimizer.second_step(zero_grad=True)

    loss_priority = losses['actor_loss']+losses['critic_value_loss']+losses["critic_return_loss"]
    loss_priority = loss_priority.detach().cpu().tolist()

    model = model.to('cpu')
    model.eval()
    return loss.item(), loss_priority
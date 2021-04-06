import torch
from torch import nn
from torch import optim
from torch import cuda

def init_opmtimistic(epoch_num, env, environment, policy_net, target_net, output_size, device):
    b_size = 128
    data = [
            environment.get_state(env, env.reset()) for _  in range(b_size)
        ]
    policy_net = policy_net.to(device)
    tmp_opt = optim.Adam(policy_net.parameters(), lr=1e-2)
    data = torch.FloatTensor(data).to(device)
    size = (b_size, output_size)
    label = 500*torch.ones(size=size).to(device)
    criterion = nn.MSELoss()
    loss_list = []
    for i in range(epoch_num):
        _label = label + 0.3*torch.rand(size=size).to(device)
        _label = nn.Softmax()(_label)
        loss = criterion(policy_net(data), _label)
        tmp_opt.zero_grad()
        loss.backward()
        tmp_opt.step()
        loss_list.append(loss.item())
    policy_net = policy_net.to('cpu')
    target_net.load_state_dict(policy_net.state_dict())

    return loss_list

def optimize_nn(batch, policy_net, target_net, optimizer, gamma, device):
    transaction = batch['transaction']
    state = torch.FloatTensor([data.state for data in transaction]).to(device)
    action = torch.LongTensor([data.action for data in transaction]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in transaction]).to(device)
    reward = torch.FloatTensor([data.reward for data in transaction]).to(device)

    target_net.train()
    policy_net.train()

    target_net = target_net.to(device)        
    policy_net = policy_net.to(device)
    
    state_action_values = policy_net(state).gather(1, action.unsqueeze(1)).squeeze()
    next_state_values = target_net(next_state).max(1)[0].detach()
    
    expected_state_action_values = (next_state_values * gamma) + reward
    loss_priority = (state_action_values - expected_state_action_values)**2
    if 'weight' in batch.keys():
        weight = torch.FloatTensor(batch['weight']).to(device)
    else:
        weight = 1
    loss = (weight*loss_priority).mean()
    optimizer.zero_grad()
    loss.backward()
    # grad clipping
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # パラメータの更新
    optimizer.step()
    
    loss_priority = loss_priority.detach().cpu().tolist()

    policy_net = policy_net.to('cpu')
    target_net = target_net.to('cpu')
    target_net.eval()
    policy_net.eval()
    return loss.item(), loss_priority


def optimize_nn_nSplit(batch, policy_net, target_net, optimizer, gamma, device):
    transaction = batch['transaction']
    state = torch.FloatTensor([data.state for data in transaction]).to(device)
    action = torch.LongTensor([data.action for data in transaction]).to(device)
    next_state = torch.FloatTensor([data.next_state for data in transaction]).to(device)
    reward = torch.FloatTensor([data.reward for data in transaction]).to(device)

    target_net.train()
    policy_net.train()
    target_net = target_net.to(device)        
    policy_net = policy_net.to(device)
    
    state_action_values = policy_net(state).gather(1, action.unsqueeze(1)).squeeze()

    expected_state_action_values = target_net(next_state).max(1)[0].detach()
    for i in range(reward.size(1)):
        expected_state_action_values = expected_state_action_values * gamma + reward[:,i]

    loss_priority = (state_action_values - expected_state_action_values)**2
    if 'weight' in batch.keys():
        weight = torch.FloatTensor(batch['weight']).to(device)
    else:
        weight = 1
    loss = (weight*loss_priority).mean()
    optimizer.zero_grad()
    loss.backward()
    # grad clipping
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # パラメータの更新
    optimizer.step()
    
    loss_priority = loss_priority.detach().cpu().tolist()

    policy_net = policy_net.to('cpu')
    target_net = target_net.to('cpu')
    target_net.eval()
    policy_net.eval()
    return loss.item(), loss_priority
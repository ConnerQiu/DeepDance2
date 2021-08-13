import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

class Value(nn.Module):
    def __init__(self, input_dim, hidden_dims, net_out_dim=None, activation = 'tanh'):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, activation)
        if net_out_dim is None:
            net_out_dim = self.net.out_dim
        self.value_head = nn.Linear(net_out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        value = self.value_head(x)
        return value

class Policy(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        """This function should return a distribution to sample action from"""
        raise NotImplementedError

    def select_action(self, x, mean_action=False):
        action = self.forward(x)
        # action = dist.mean_sample() if mean_action else dist.sample()
        return action

    def get_kl(self, x):
        dist = self.forward(x)
        return dist.kl()

    def get_log_prob(self, x, action):
        dist = self.forward(x)
        return dist.log_prob(action)

class PolicyDetermin(Policy):
    def __init__(self, input_dim, hidden_dims, action_dim, activation = 'tanh'): 
        super().__init__()
        self.type = 'Determin'
        self.net = MLP(input_dim, hidden_dims, activation)
        self.action_head = nn.Linear(self.net.out_dim, action_dim)

    def forward(self, x):
        x = self.net(x)
        action = self.action_head(x)
        return action

class PolicyGaussian(Policy):
    def __init__(self, net, action_dim, net_out_dim=None, log_std=0, fix_std=False):
        super().__init__()
        self.type = 'gaussian'
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)

    def forward(self, x):
        x = self.net(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return DiagGaussian(action_mean, action_std)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {'std_id': std_id, 'std_index': std_index}
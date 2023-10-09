import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

num_agent = 5

EPS = 0.003


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim + 5 * (num_agent - 1)
        self.action_dim = action_dim.shape[0] + 1
        # um_inputs + 5 * (num_agent - 1) + num_actions + num_agent, hidden_dim
        self.fcs1 = nn.Linear(self.state_dim + self.action_dim , 64)
        self.fcs2 = nn.Linear(64, 64)
        self.fca1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fcs2(x))
        x = F.relu(self.fca1(x))
        x = self.fc2(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = torch.tensor([0.2])
        self.action_bias = torch.tensor([0.])
        self.fc1 = nn.Linear(self.state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.mean_linear = nn.Linear(64, action_dim)
        self.log_std_linear = nn.Linear(64, action_dim)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.relu(self.fc4(x))
        mean = self.mean_linear(action)
        log_std = self.log_std_linear(action)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)  # construct a normal distribution
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def policy_weights_init_(m):
    if isinstance(m, nn.Linear):
        m = torch.load("..\\model\\pretrain.pb")


# ValueNetwork inherits from nn.Module
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        # ValueNetwork calls to the nn.Module,initialize the partial model
        super(ValueNetwork, self).__init__()
        # 3 layer linear function
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)
        self.linear_weight = torch.tensor([1.])

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.linear(self.linear4(x), self.linear_weight)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear_weight = torch.tensor([1.])

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        # self.apply(policy_weights_init_)

    # make forward propagation to 2 critic networks
    def forward(self, state, action):
        # xu is a one dimensional vector including state and action
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)

        x2 = F.relu(self.linear5(xu))
        x2 = F.relu(self.linear6(x2))
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)
        # x1/x2 is the output
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        # construct the connection between hidden_dim(Penultimate layer) to the output layer
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        # self.apply(policy_weights_init_)
        self.apply(weights_init_)
        self.action_scale = torch.tensor([0.2])
        self.action_bias = torch.tensor([0.])

        # if action_space is None:
        #     self.action_scale = torch.tensor(1.)
        #     self.action_bias = torch.tensor(0.)
        # else:
        #     self.action_scale = torch.FloatTensor(
        #         (self.action_space_high - self.action_space_low) / 2.)
        #     self.action_bias = torch.FloatTensor(
        #         (self.action_space_high + self.action_space_low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamps all elements in input into the range [ min, max ]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)  # construct a normal distribution
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)
        # self.apply(policy_weights_init_)
        self.apply(weights_init_)

        # action rescaling
        # if action_space is None:
        #     self.action_scale = 1.
        #     self.action_bias = 0.
        # else:
        #     self.action_scale = torch.FloatTensor(
        #         (action_space.high - action_space.low) / 2.)
        #     self.action_bias = torch.FloatTensor(
        #         (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        # self.action_scale = self.action_scale.to(device)
        # self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)



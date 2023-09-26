import argparse
import sys

import numpy as np
import torch
import pandas as pd
from sac import SAC
# from ac import AC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import random
import matplotlib.pyplot as plt
import math
import os
from utils import *
from env import Env
import datetime
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork
from torch.optim import Adam, SGD


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--clr', type=float, default=0.0003, metavar='G',
                    help='critic learning rate (default: 0.0003)')
parser.add_argument('--plr', type=float, default=0.0001, metavar='G',
                    help='policy learning rate (default: 0.0003)')
parser.add_argument('--alphalr', type=float, default=0.0002, metavar='G',
                    help='alpha learning rate (default: 0.2)')
parser.add_argument('--num_agent', type=int, default=2, metavar='N',
                    help='number of agents (default: 2)')
parser.add_argument('--algorithm', type=str, default="SAC", metavar='N',
                    help='type of algorithm (default: sac)')
args = parser.parse_args()


class Leader:  # 需要follow的列表，tensorboard写入器，
    def __init__(self, follower_list, writer, station_num=1):
        self.followers = follower_list
        self.writer = writer
        self.station_num = station_num
        self.state = torch.randn(1, len(self.followers) * self.followers[0].action.shape[0])   # leader的状态是所有follower的action维度之和，这里默认
        # 所有的follower的动作维度一样，并且假设follower list里面至少有一个follower
        self.action = torch.randn(self.station_num)  # leader的action就是电价值，我们这里假设他是一个向量，不同的维度表示不同充电站的电价，也就是说不同的充电站可以有不同的电价
        self.agent = SAC(self.state.shape[1], self.action, args)
        self.memory = ReplayMemory(args.replay_size, args.seed)
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic = QNetwork(self.state.shape[1], self.action.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.clr)
        self.critic_target = QNetwork(self.state.shape[1], self.action.shape[0], args.hidden_size).to(self.device)
        # self.critic_target = Q_target.to(device=self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(torch.tensor([]).shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                # log_alpha is the updated parameter in SAC
                self.alpha_optim = Adam([self.log_alpha], lr=args.alphalr)

                # self.alpha_optim = SGD([self.log_alpha], lr=args.alphalr, momentum=0.9)

            self.policy = GaussianPolicy(self.state.shape[1], self.action.shape[0], args.hidden_size).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.plr)

    def select_action(self, state, evaluate=False):
        # FloatTensor:32-bit floating point
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, id):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, other_action_batch, other_action_next_batch\
            = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        other_action_batch = torch.FloatTensor(other_action_batch).to(self.device)
        other_action_next_batch = torch.FloatTensor(other_action_next_batch).to(self.device)
        print(other_action_batch.shape)
        # disabled gradient calculation, calculate the loss of q network
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(torch.cat((next_state_batch[:, 0:48], state_batch[:, 48 + id * 5: 48 + (id + 1) * 5]), 1))
            # qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, torch.mean(other_action_next_batch, dim=1, keepdim=True))
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, other_action_next_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)  # - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # qf1, qf2 = self.critic(state_batch, action_batch, torch.mean(other_action_batch, dim=1, keepdim=True))  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch, other_action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # g_loss = torch.mean(self.g_net(other_action_batch))
        # qf1_loss += g_loss
        # qf2_loss += g_loss
        qf_loss = qf1_loss + qf2_loss

        # update q network parameters
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        # self.critic_scheduler.step()

        pi, log_pi, _ = self.policy.sample(torch.cat((state_batch[:, 0:48], state_batch[:, 48 + id * 5: 48 + (id + 1) * 5]), 1))

        qf1_pi, qf2_pi = self.critic(state_batch, pi, other_action_batch)
        # qf1_pi, qf2_pi = self.critic(state_batch, pi, torch.mean(other_action_batch, dim=1, keepdim=True))
        # qf1_pi += g_loss
        # qf2_pi += g_loss
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        #  policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))] SAC
        policy_loss = -1.0 * (log_pi * min_qf_pi).mean()  # AC
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        # self.policy_scheduler.step()
        # update par alpha automatically through calculating alpha loss
        # if self.automatic_entropy_tuning:
        #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        #
        #     self.alpha_optim.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optim.step()
        #
        #     self.alpha = self.log_alpha.exp()
        #     alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        # else:
        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        # item()-tuple (key,value)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
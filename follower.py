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
from model import GaussianPolicy, QNetwork

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


class Follower:
    def __init__(self, ev_id, global_env, writer, bus=1, d1_distribution=None, anxious_duration=None, EV_case=None, anx="low"):
        if anxious_duration is None:
            anxious_duration = [1, 4]
        if d1_distribution is None:
            d1_distribution = [0.85, 0.95]
        if EV_case is None:
            mu_arr = [18.8, 9.8, 11.6]
            sigma_arr = [3.6, 2.9, 3.6]
            mu_dep = [9.1, 18.4, 15.6]
            sigma_dep = [2.3, 3.1, 3.7]
            EV_case = [mu_arr, sigma_arr, mu_dep, sigma_dep]
        self.ev_id = ev_id  # follower的id，不会发生变化
        self.env = Env(EV_case, anx)
        self.grid = global_env
        self.writer = writer
        self.bus = bus  # follower当前所在的节点编号
        self.d1 = d1_distribution
        self.anxious_duration = anxious_duration
        self.EV_case = EV_case
        self.state = torch.randn(1, 6)  # follower的状态包括6个维度，分别是price，t_x，t_d，soc，soc_x，soc_d
        self.action = torch.tensor([0.])
        self.agent = SAC(self.state.shape[1], self.action, args)  # 使用SAC算法
        self.state = self.env.state
        self.memory = ReplayMemory(args.replay_size, args.seed)  # replay buffer
        self.episode_r = []
        self.epoch_price = []
        self.epoch_anx = []
        self.epoch_power = []
        self.cr1_lst = []
        self.cr2_lst = []
        self.policy_lst = []
        self.alpha_lst = []
        self.episode_reward = np.array([0.0], dtype='f8')
        self.anx_reward = np.array([0.0], dtype='f8')
        self.price_reward = np.array([0.0], dtype='f8')
        self.episode_step = 0
        self.updates = 0

    def update_paramaters(self):
        if len(self.memory) > args.batch_size:
            if args.algorithm == "SAC":
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
                    = self.agent.update_parameters(self.memory, args.batch_size, self.updates)
                self.cr1_lst.append(critic_1_loss)
                self.cr2_lst.append(critic_2_loss)
                self.alpha_lst.append(alpha)
                self.writer.add_scalar(self.ev_id + '/loss/critic_1', critic_1_loss, self.updates)
                self.writer.add_scalar(self.ev_id + '/loss/critic_2', critic_1_loss, self.updates)
                self.writer.add_scalar(self.ev_id + '/loss/policy', policy_loss, self.updates)
                self.writer.add_scalar(self.ev_id + '/loss/entropy_loss', ent_loss, self.updates)
                self.writer.add_scalar(self.ev_id + '/entropy_temperature/alpha', alpha, self.updates)
                self.updates += 1
            else:
                critic_loss, policy_loss = self.agent.update_parameters(self.memory, args.batch_size, self.updates)
                self.cr1_lst.append(critic_loss)
                self.writer.add_scalar(self.ev_id + '/loss/critic_1', critic_loss, self.updates)
                self.writer.add_scalar(self.ev_id + '/loss/policy', policy_loss, self.updates)
                self.updates += 1

    def interact(self, global_t, price=1.0):  # 在global_t时刻以电价price和全局环境进行交互，返回当前时间和电价下的action
        done = False
        if global_t > self.env.t_d:  #  如果现在的时间还没到自己的离开时间，那么自己的环境就不需要重置时间重新算一个t_d
            self.state = self.env.reset(global_t, price)
        self.episode_reward = np.array([0.0], dtype='f8')  # 在这24个小时内的reward，下一个24小时要重置
        self.anx_reward = np.array([0.0], dtype='f8')
        self.price_reward = np.array([0.0], dtype='f8')

        action = self.agent.select_action(self.state)
        next_state, reward_tuple, action, done = self.env.step(action)
        reward = reward_tuple[0]
        anx = reward_tuple[1]
        price = reward_tuple[2]

        self.anx_reward += anx
        self.price_reward += price
        self.memory.push(self.state, action/0.2, reward, next_state, float(done))
        self.state = next_state
        return action, self.price_reward, self.anx_reward
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
import torch.nn.functional as F
from utils import *
from env import Env
import datetime



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


class Leader:  # 需要follow的列表，tensorboard写入器，网络中节点的数量
    def __init__(self, writer, node_num=15):
        self.writer = writer
        self.node_num = node_num  # 电网中节点的数量
        self.state = torch.randn(1, self.node_num)  # leader的状态是所有节点的有功功率，这里默认整个电网中至少有一个节点
        self.action = torch.randn(self.node_num)  # leader的action就是电价值，我们这里假设他是一个向量，不同的维度表示不同充电站的电价，也就是说不同的充电站可以有不同的电价，我们假设每一个节点都有电站
        self.agent = SAC(self.state.shape[1], self.action, args)  # leader使用的学习算法为sac
        self.memory = ReplayMemory(args.replay_size, args.seed)  # replay buffer
        self.r_reward = np.array([0.0], dtype='f8')  # 通过卖电获得的奖励收益
        self.p_reward = np.array([0.0], dtype='f8')  # OPF的奖励收益
        self.cr1_lst = []
        self.cr2_lst = []
        self.policy_lst = []
        self.alpha_lst = []
        self.total_numsteps = 0
        self.episode_step = 0
        self.updates = 0


    def get_state(self, nodes_power_list):
        self.state = torch.Tensor(nodes_power_list)
        return self.state

    def select_action(self, state):
        action = self.agent.select_action(state)
        return action




    def update_paramaters(self):
        if len(self.memory) > args.batch_size:
            if args.algorithm == "SAC":
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
                    = self.agent.update_parameters(self.memory, args.batch_size, self.updates, id)
                self.cr1_lst.append(critic_1_loss)
                self.cr2_lst.append(critic_2_loss)
                self.policy_lst.append(policy_loss.item())
                self.alpha_lst.append(alpha)
                self.writer.add_scalar("leader" + '/loss/critic_1', critic_1_loss, self.updates)
                self.writer.add_scalar("leader" + '/loss/critic_2', critic_1_loss, self.updates)
                self.writer.add_scalar("leader" + '/loss/policy', policy_loss, self.updates)
                self.writer.add_scalar("leader" + '/loss/entropy_loss', ent_loss, self.updates)
                self.writer.add_scalar("leader" + '/entropy_temperature/alpha', alpha, self.updates)
                self.updates += 1
            else:
                critic_loss, policy_loss = self.agent.update_parameters(self.memory, args.batch_size, self.updates, id)
                self.cr1_lst.append(critic_loss)
                self.writer.add_scalar(self.ev_id + '/loss/critic_1', critic_loss, self.updates)
                self.writer.add_scalar(self.ev_id + '/loss/policy', policy_loss, self.updates)
                self.updates += 1


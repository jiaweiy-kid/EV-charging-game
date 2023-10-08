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


class Agent:
    def __init__(self, ev_id, global_env, writer, bus=1, d1_distribution=None,
                 anxious_duration=None, EV_case=None, anx="low"):
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
        self.ev_id = ev_id
        self.anx = anx
        # self.env = Env('..\\price\\trainPrice.xlsx', d1_distribution, anxious_duration, EV_case)
        self.env = Env('../price/trainPrice.xlsx', d1_distribution, anxious_duration, EV_case, anx)
        self.global_env = global_env
        self.writer = writer
        self.bus = bus
        self.d1 = d1_distribution
        self.anxious_duration = anxious_duration
        self.EV_case = EV_case
        self.state = torch.randn(1, 53)
        self.action = torch.tensor([0.])
        if args.algorithm == "SAC":
            self.agent = SAC(self.state.shape[1], self.action, args)
        # else:
        #     self.agent = AC(self.state.shape[1], self.action, args)
        self.memory = ReplayMemory(args.replay_size, args.seed)
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
        self.power_reward = np.array([0.0], dtype='f8')
        self.total_numsteps = 0
        self.episode_step = 0
        self.updates = 0

    # def power_train(self, i_episode, episode_states, episode_powers, clients_episode_actions, clients_episode_states,
    #                 id):  # i_episode表示本地的轮数，episode_states表示一个episode的所有状态四元组
    #     # episode_powers表示一个episode的OPF的reward
    #     for i in range(len(episode_states)):
    #         other_action = [None] * len(clients_episode_actions)
    #         other_action_next = [None] * len(clients_episode_actions)
    #         other_state = [None] * len(clients_episode_actions)
    #         other_state_next = [None] * len(clients_episode_actions)
    #         episode_state = episode_states[i]
    #         episode_power = episode_powers[i]
    #         for j in range(len(clients_episode_actions)):  # 遍历所有agent的episode action
    #             if i < len(clients_episode_actions[j]):  # 如果agent在这一步有action，则放入，否则放入0.0
    #                 other_action[j] = clients_episode_actions[j][i]
    #                 other_state[j] = clients_episode_states[j][i][0][-5:].tolist()
    #                 if i + 1 < len(clients_episode_actions[j]):
    #                     other_action_next[j] = clients_episode_actions[j][i + 1]
    #                     other_state_next[j] = clients_episode_states[j][i][3][-5:].tolist()
    #                 else:
    #                     other_action_next[j] = 0.0
    #                     other_state_next[j] = clients_episode_states[j][-1][3][-5:].tolist()
    #
    #             else:
    #                 other_action[j] = 0.0
    #                 other_action_next[j] = 0.0
    #                 other_state[j] = clients_episode_states[j][-1][0][-5:].tolist()
    #                 other_state_next[j] = clients_episode_states[j][-1][3][-5:].tolist()
    #         global_state = np.append(episode_state[0][0:48], np.array(sum(other_state, [])))
    #         global_state_next = np.append(episode_state[3][0:48], np.array(sum(other_state_next, [])))
    #         self.memory.push(global_state, episode_state[1] / 0.2, episode_state[2],
    #                          global_state_next, episode_state[4], other_action, other_action_next)
    #         self.episode_reward += episode_state[2]
    #         self.power_reward += episode_power
    #
    #     print(self.ev_id, " episode_reward:", self.episode_reward, "price reward:", self.price_reward, "anx_reward:",
    #           self.anx_reward, "power_reward:", self.power_reward)
    #
    #     self.writer.add_scalar(self.ev_id + '/reward/episode_reward', self.episode_reward, i_episode)
    #     self.writer.add_scalar(self.ev_id + '/reward/price_reward', self.price_reward, i_episode)
    #     self.writer.add_scalar(self.ev_id + '/reward/anx_reward', self.anx_reward, i_episode)
    #     self.writer.add_scalar(self.ev_id + '/reward/opf_reward', self.power_reward, i_episode)
    #
    #     self.episode_r.append(self.episode_reward.copy())
    #     self.epoch_price.append(self.price_reward.copy())
    #     self.epoch_anx.append(self.anx_reward.copy())
    #     self.epoch_power.append(self.power_reward.copy())
    #
    #     self.episode_reward = np.array([0.0], dtype='f8')
    #     self.anx_reward = np.array([0.0], dtype='f8')
    #     self.price_reward = np.array([0.0], dtype='f8')
    #     self.power_reward = np.array([0.0], dtype='f8')
    #     if len(self.memory) > args.batch_size:
    #         # for i in range(10):  # each training tep
    #         if args.algorithm == "SAC":
    #             critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
    #                 = self.agent.update_parameters(self.memory, args.batch_size, self.updates, id)
    #             self.cr1_lst.append(critic_1_loss)
    #             self.cr2_lst.append(critic_2_loss)
    #             # self.policy_lst.append(policy_loss.item())
    #             self.alpha_lst.append(alpha)
    #             # global_q.writer.add_scalar("global_critic" + '/loss/critic_1', critic_1_loss, global_q.updates)
    #             # global_q.writer.add_scalar("global_critic" + '/loss/critic_2', critic_2_loss, global_q.updates)
    #             self.writer.add_scalar(self.ev_id + '/loss/critic_1', critic_1_loss, self.updates)
    #             self.writer.add_scalar(self.ev_id + '/loss/critic_2', critic_1_loss, self.updates)
    #             self.writer.add_scalar(self.ev_id + '/loss/policy', policy_loss, self.updates)
    #             self.writer.add_scalar(self.ev_id + '/loss/entropy_loss', ent_loss, self.updates)
    #             self.writer.add_scalar(self.ev_id + '/entropy_temperature/alpha', alpha, self.updates)
    #             self.updates += 1
    #         else:
    #             critic_loss, policy_loss = self.agent.update_parameters(self.memory, args.batch_size, self.updates, id)
    #             self.cr1_lst.append(critic_loss)
    #             self.writer.add_scalar(self.ev_id + '/loss/critic_1', critic_loss, self.updates)
    #             self.writer.add_scalar(self.ev_id + '/loss/policy', policy_loss, self.updates)
    #             self.updates += 1


from torch.utils.tensorboard import SummaryWriter
from follower import Follower
import torch
from env import Env
from utils import soft_update, hard_update
import torch.nn.functional as F
import numpy as np
import random
from globalenv import globalenv
from simulation_case import get_simulation_case, get_EV_case
from leader import Leader
from replay_memory import ReplayMemory


client_num = 5  # followers的数量
client_d1_distribution = [[0.85, 0.95], [0.85, 0.9], [0.9, 0.95]]
for i in range(client_num-3):
    client_d1_distribution.append([0.85, 0.95])
client_anxious_duration = [[1, 4], [1, 2], [2, 4]]
for i in range(client_num-3):
    client_anxious_duration.append([1, 4])

lower_rounds = 1000  # 下层的训练轮数
upper_rounds = 500  # 上层的训练轮数
case_list = get_EV_case()
client_location = [random.randint(1, 14) for i in range(client_num)]
writer = SummaryWriter('../run/game')  # tensorboard的记录器
leader = Leader(writer)  #  供电商
globalEnv = globalenv(leader)  # 电网环境
anx_list = ["low", "high", "low", "high", "low"]
followers = [Follower(str(i), globalEnv, writer, client_location[i], client_d1_distribution[i],\
                 client_anxious_duration[i], case_list[i], anx_list[i]) for i in range(client_num)]  # 初始化所有的agent

# followers训练学习，leader不动
# 这里需要注意，在每一次leader更新之后，理论上所有的follower需要重置所有的网络参数，这样才能避免陷入局部解，但是我们这里为了能够更快的学习，直接把上一轮的结果作为初始值
for global_round in range(20):
    for episode in range(lower_rounds):
        globalEnv.reset()
        episode_total_reward = [0.0] * len(followers)
        episode_price_reward = [0.0] * len(followers)
        episode_anxious_reward = [0.0] * len(followers)
        for global_t in range(1, 25):
            globalEnv.reset()
            grid_state = globalEnv.state  # 电网每一个节点的有功功率，也就是leader当前的状态
            price = globalEnv.get_price()  # 获得当前的电价，也就是leader当前的action，是一个list
            followers_action_list = []
            for follower_id in range(len(followers)):
                action, price_reward, anxious_reward = followers[follower_id].interact(global_t, price[client_location[follower_id]])
                followers_action_list.append(action)
                episode_anxious_reward[follower_id] += anxious_reward
                episode_price_reward[follower_id] += price_reward
                episode_total_reward[follower_id] += (anxious_reward + price_reward)
                # 获取当前步所有的follower的action
            next_grid_state, _, _, _ = globalEnv.step(followers_action_list, client_location)  # 电网的环境迭代一步，leader获取下一步的状态以及奖励值，但是这里因为follower还没有达到平衡，所以不需要reward
        print("EV total rewards is")
        print(episode_total_reward)
        for i in range(len(followers)):  # 更新网络的参数
            followers[i].update_paramaters()
            followers[i].writer.add_scalar(followers[i].ev_id + '/reward/episode_reward', episode_total_reward[i], followers[i].updates)
            followers[i].writer.add_scalar(followers[i].ev_id + '/reward/price_reward', episode_price_reward[i], followers[i].updates)
            followers[i].writer.add_scalar(followers[i].ev_id + '/reward/anx_reward', episode_anxious_reward[i], followers[i].updates)


    # followers不动，leader进行更新
    for episode in range(lower_rounds):
        globalEnv.reset()
        episode_sale_reward = 0.0
        episode_opf_reward = 0.0
        leader_total_rewards = 0.0
        for global_t in range(1, 25):
            globalEnv.reset()
            grid_state = globalEnv.state  # 电网每一个节点的有功功率，也就是leader当前的状态
            price = globalEnv.get_price()  # 获得当前的电价，也就是leader当前的action，是一个list
            followers_action_list = []
            for follower_id in range(len(followers)):
                action, _, _ = followers[follower_id].interact(global_t, price[client_location[follower_id]])
                followers_action_list.append(action)
                # 获取当前步所有的follower的action
            next_grid_state, leader_sale_reward, leader_opf_reward, leader_total_reward = globalEnv.step(followers_action_list, client_location)  # 电网的环境迭代一步，leader获取下一步的状态以及奖励值，但是这里因为follower还没有达到平衡，所以不需要reward
            episode_sale_reward += leader_sale_reward
            episode_opf_reward += leader_opf_reward
            leader_total_rewards += leader_total_reward

            leader.memory.push(grid_state, price, leader_total_reward, next_grid_state, 0.0)
        print("leader total reward is {}".format(leader_total_rewards))
        # 更新网络的参数
        leader.update_paramaters()
        leader.writer.add_scalar('leader' + '/reward/total_reward', leader_total_rewards, leader.updates)
        leader.writer.add_scalar('leader' + '/reward/sale_reward', episode_sale_reward, leader.updates)
        leader.writer.add_scalar('leader' + '/reward/opf_reward', episode_opf_reward, leader.updates)


















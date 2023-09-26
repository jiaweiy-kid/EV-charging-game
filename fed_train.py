from torch.utils.tensorboard import SummaryWriter

from agent import Agent
import torch
from env import Env
from utils import soft_update, hard_update
import torch.nn.functional as F
import numpy as np
import random
from globalenv import globalenv
from simulation_case import get_simulation_case, get_EV_case
from replay_memory import ReplayMemory
from torch.optim import Adam, SGD
#  设定的一些参数
client_num = 5  # agent的数量
client_d1_distribution = [[0.85, 0.95], [0.85, 0.9], [0.9, 0.95]]
for i in range(client_num-3):
    client_d1_distribution.append([0.85, 0.95])
client_anxious_duration = [[1, 4], [1, 2], [2, 4]]
for i in range(client_num-3):
    client_anxious_duration.append([1, 4])
episode_times = 5000  # 本地训练的轮数，在agent中默认是1000，这里设置为2000
simulation_list = get_simulation_case()
case_list = get_EV_case()
# test_path = '..\\price\\testPrice.xlsx'
test_path = '../price/testPrice.xlsx'
client_location = [random.randint(1, 14) for i in range(client_num)]  # agent连接的bus位置
globalEnv = globalenv(client_num)  # 初始化全局环境（电网的case）
writer = SummaryWriter('../run/ac_mix')
anx_list = ["low", "high", "low", "high", "low"]
clients = [Agent(str(i), globalEnv, writer, client_location[i], episode_times, client_d1_distribution[i],\
                 client_anxious_duration[i], case_list[i], anx_list[i]) for i in range(client_num)]  # 初始化所有的agent
num_round = 1  # 联邦训练的轮数
memory = ReplayMemory(100000, 123456)

for i_episode in range(1, episode_times + 1):
    print('episode_{}:'.format(i_episode))
    clients_episode_states = []
    clients_episode_times = []  # client一轮中采取动作的时间点
    clients_episode_actions = []  # client一轮的所有动作
    bus_list = []
    # t_tongb = random.randint(clients[0].env.start_point, len(clients[0].env.data) - 100)
    t_tongb = 2000
    for client in clients:
        episode_states, episode_times = client.online_interact(t_tongb)
        # print(episode_times)
        clients_episode_states.append(episode_states)
        clients_episode_times.append(episode_times)
        bus_list.append(client.bus)
        client_episode_actions = []
        for i in range(len(episode_states)):
            # 电桩最高是30kw
            client_episode_actions.append(episode_states[i][1].tolist()[0] * 0.15)
        clients_episode_actions.append(client_episode_actions)
    # 计算24小时每个时刻的global reward
    time_power_reward = globalEnv.calculateReward(bus_list, clients_episode_actions, clients_episode_times)
    clients_episode_powers = []

    for j in range(client_num):
        if len(clients_episode_powers) <= j:
            clients_episode_powers.append([])
        for i, time in enumerate(clients_episode_times[j]):
            index = time % 24
            mu = 0.5
            clients_episode_powers[j].append(time_power_reward[index])
            if j <= 1:  # balance
                clients_episode_states[j][i][2] = clients_episode_states[j][i][2] + time_power_reward[index]  # 将OPF的reward加到里面去
            elif j == 4: # 无私
                clients_episode_states[j][i][2] = 0 * clients_episode_states[j][i][2] + time_power_reward[index]
            else:  # 自私
                clients_episode_states[j][i][2] = clients_episode_states[j][i][2] + 0 * time_power_reward[index]
            # clients_episode_states[j][i][2] += time_power_reward[index]
    clients_global = []
    for i, client in enumerate(clients):
        clients[i].power_train(i_episode, clients_episode_states[i], clients_episode_powers[i], clients_episode_actions, clients_episode_states, i)
    if i_episode % 400 == 0:
        for i, client in enumerate(clients):
            clients[i].agent.save_checkpoint(str(i_episode), str(i), ckpt_path=None)


# env = Env('..\\price\\trainPrice.xlsx', [1, 4], [0.85, 0.95], case_list[0])
# env.simulation(clients[0].agent, "1", random.randint(clients[0].env.start_point, len(clients[0].env.data) - 100))
# env.simulation(clients[1].agent, "2", random.randint(clients[0].env.start_point, len(clients[0].env.data) - 100))
# env.simulation(clients[2].agent, "3", random.randint(clients[0].env.start_point, len(clients[0].env.data) - 100))










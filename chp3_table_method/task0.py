#!/usr/bin/env python
# coding=utf-8
import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gymnasium as gym
import torch
import datetime

from env.gridworld_env import CliffWalkingWapper
from qlearning import QLearning
from common.utils import plot_rewards, create_moving_average_calculator
from common.utils import save_results, make_dir

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'Q-learning'  # 算法名称
        self.env_name = 'CliffWalking-v0'  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")  # 检测GPUgjgjlkhfsf风刀霜的撒发十
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 400  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        ################################################################################

        ################################## 算法超参数 ###################################
        self.gamma = 0.90  # 强化学习中的折扣因子
        self.epsilon_start = 0.95  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 300  # e-greedy策略中epsilon的衰减率
        self.lr = 0.1  # 学习率
        ################################################################################

        ################################# 保存结果相关参数 ################################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片
        ################################################################################


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        state = env.reset(seed=cfg.seed)  # 重置环境,即开始新的回合
        while True:
            action = agent.choose_action(state)  # 根据算法选择一个动作
            next_state, reward, done, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, done)  # Q学习算法更新
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print("回合数：{}/{}，奖励{:.1f}".format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    return rewards


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    for item in agent.Q_table.items():
        print(item)
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state = env.reset(seed=cfg.seed)  # 重置环境,即开始新的回合
        while True:
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state  # 更新状态
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep+1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards


def env_agent_config(cfg, rmode=None):
    env = gym.make(cfg.env_name, render_mode=rmode)
    env = CliffWalkingWapper(env)
    state_dim = env.observation_space.n  # 状态维度
    action_dim = env.action_space.n  # 动作维度
    agent = QLearning(state_dim, action_dim, cfg)
    return env, agent


if __name__ == "__main__":
    cfg = Config()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=cfg.model_path)  # 保存模型
    save_results(rewards, tag='train', path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, cfg, tag="train")  # 画出结果
    # 测试
    env, agent = env_agent_config(cfg, rmode='human')
    agent.load(path=cfg.model_path)  # 导入模型
    rewards = test(cfg, env, agent)
    save_results(rewards, tag='test', path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, cfg, tag="test")  # 画出结果

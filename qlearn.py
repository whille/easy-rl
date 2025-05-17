#!/usr/bin/env python
import numpy as np
from collections import defaultdict


def e_greedy_action(env, Q, s, epsilon):
    if np.random.rand() < epsilon:
        a = np.random.choice(env.get_valid_actions(s))  # 探索
    else:
        valid_actions = env.get_valid_actions(s)
        a = valid_actions[np.argmax([Q[s][a] for a in valid_actions])]  # 利用
    return a


def q_learning(env, num_episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Q学习是一种基于时间差分（TD）的强化学习方法，它通过更新Q值来学习最优策略。
    在这个Q学习版本中，我们使用了一个ε-贪婪策略来选择动作
    :param env: 环境对象（需实现reset(), step()等方法）
    :param num_episodes: 训练回合数
    :param alpha: 学习率
    :param gamma: 折扣因子
    :param epsilon: 探索率
    :return: 最优策略 pi, 动作价值函数 Q
    """
    # 初始化
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))  # 动作价值函数 Q(s,a)
    for iter_i in range(num_episodes):
        epsilon = 0.001 + 0.099 * (1 - iter_i / num_episodes)
        print(f"Episode: {iter_i + 1}/{num_episodes}, epsilon: {epsilon}", end="\r")
        s = env.reset()
        done = False
        while not done:
            a = e_greedy_action(env, Q, s, epsilon)
            next_s, r, done, _ = env.step(a)
            TD_target = r + gamma * np.max(Q[next_s])
            TD_error = TD_target - Q[s][a]
            Q[s][a] += alpha * TD_error
            s = next_s
    pi = calc_pi(env, Q)
    return pi, Q


# 从 Q 中提取最优策略
def calc_pi(env, Q):
    nA = env.action_space.n
    pi = defaultdict(lambda: np.random.choice(nA))
    for s in Q:
        valid_actions = env.get_valid_actions(s)
        pi[s] = valid_actions[np.argmax([Q[s][a] for a in valid_actions])]
    return pi


def sarsa(env, num_episodes=100, alpha=0.1, gamma=0.9, epsilon=0.01):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))  # 动作价值函数 Q(s,a)
    for iter_i in range(num_episodes):
        print(f"Episode: {iter_i + 1}/{num_episodes}, epsilon: {epsilon}", end="\r")
        s = env.reset()
        a = e_greedy_action(env, Q, s, epsilon)
        done = False
        while not done:
            next_s, r, done, _ = env.step(a)
            next_a = e_greedy_action(env, Q, s, epsilon)
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[next_s][next_a] - Q[s][a])
            s = next_s
            a = next_a
    pi = calc_pi(env, Q)
    return pi, Q

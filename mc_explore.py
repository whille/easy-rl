#!/usr/bin/env python
import numpy as np
from collections import defaultdict


def monte_carlo_es(env, num_episodes=100, gamma=0.9):
    """
    ref: ./mc_explore.md
    基于探索性开始的蒙特卡洛方法
    :param env: 环境对象（需实现reset(), step()等方法）
    :param num_episodes: 训练回合数
    :param gamma: 折扣因子
    :return: 最优策略 pi, 动作价值函数 Q
    """
    # 初始化
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))  # 动作价值函数 Q(s,a)
    R = defaultdict(list)                  # 存储每个(s,a)的回报序列
    pi = defaultdict(lambda: np.random.choice(nA))  # 随机初始化策略 π(s)

    for iter_i in range(num_episodes):
        print(f"Episode: {iter_i + 1}/{num_episodes}", end="\r")
        # 1. 随机选择初始状态和动作（探索性开始）
        s0 = env.reset()
        a0 = np.random.choice(env.get_valid_actions(s0))

        # 2. 生成回合轨迹: s0, a0, r1, s1, a1, ..., sT
        trajectory = []
        s, a = s0, a0
        done = False
        while not done:
            next_s, r, done, _ = env.step(a)
            trajectory.append((s, a, r))
            # print(f"step: {_}, s: {s}, a:{a}, trajectory: {len(trajectory)}")
            s = next_s
            a = np.random.choice(env.get_valid_actions(s)) if not done else None  # 修复点

        # 3. 反向计算累积奖励并更新
        G = 0
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t_plus_1 = trajectory[t]
            G = gamma * G + r_t_plus_1

            # 首次访问MC：仅当(s_t,a_t)未在之前出现时更新
            if (s_t, a_t) in [(x[0], x[1]) for x in trajectory[:t]]:
                R[(s_t, a_t)].append(G)
                Q[s_t][a_t] = np.mean(R[(s_t, a_t)])  # 更新Q值为平均回报
                valid_actions = env.get_valid_actions(s_t)  # 新增方法，获取合法动作
                if valid_actions:
                    pi[s_t] = valid_actions[np.argmax([Q[s_t][a] for a in valid_actions])]
        #print(f"step: {_}, s0: {s0}, a0:{a0}, pi: {pi}, Q: {Q}")  # 打印当前策略和Q值

    return pi, Q

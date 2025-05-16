#!/usr/bin/env python
import enum
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


# 示例：简单网格世界环境
class Action(enum.Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class GridWorld:
    def __init__(self):
        self.action_space = type('', (), {'n': 4})()  # 动作：右(0)、左(1)、上(2)、下(3)
        self.observation_space = type('', (), {'n': 16})()  # 4x4网格
        self.reset()

    def reset(self):
        self.state = 0
        return self.state

    def get_valid_actions(self, s):
        """新增方法：返回当前状态下不会导致越界的动作"""
        x, y = s // 4, s % 4
        valid_actions = []
        if y < 3: valid_actions.append(0)  # 右
        if y > 0: valid_actions.append(1)  # 左
        if x > 0: valid_actions.append(2)  # 上
        if x < 3: valid_actions.append(3)  # 下
        return valid_actions

    def step(self, action):
        x, y = self.state // 4, self.state % 4
        if action == 0: y = min(y + 1, 3)  # 右
        elif action == 1: y = max(y - 1, 0)  # 左
        elif action == 2: x = max(x - 1, 0)  # 上
        elif action == 3: x = min(x + 1, 3)  # 下

        self.state = x * 4 + y
        reward = 10 if self.state == 15 else -1
        done = (self.state == 15)
        return self.state, reward, done, {}


if __name__ == "__main__":
    env = GridWorld()
    pi, Q = monte_carlo_es(env, num_episodes=1000)

    print("最优策略（网格坐标→动作）：")
    for s in range(16):
        valid_actions = env.get_valid_actions(s)
        action_symbol = '→←↑↓'[pi[s]] if pi[s] in valid_actions else '×'
        print(f"({s//4}, {s%4}): {action_symbol}", end="\t")
        if s % 4 == 3:
            print()

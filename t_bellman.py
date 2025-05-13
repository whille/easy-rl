#!/usr/bin/env python

import numpy as np

# 定义环境参数
gamma = 0.9  # 折扣因子

# 定义奖励函数和状态转移概率, R(s,a)
reward_matrix = np.array([[0, 1], [-1, 0], [0, -1], [1, 0], [0, 0]])
num_states, num_actions = reward_matrix.shape

# p(s′∣s,a), 从当前状态s采取动作a转移到未来状态s'的概率
transition_matrix = np.array([
    [[0.8, 0.1, 0.05, 0.05, 0.0], [0.3, 0.3, 0.2, 0.1, 0.1]],
    [[0.7, 0.1, 0.1, 0.05, 0.05], [0.2, 0.4, 0.2, 0.1, 0.1]],
    [[0.6, 0.2, 0.1, 0.05, 0.05], [0.4, 0.3, 0.1, 0.1, 0.1]],
    [[0.4, 0.3, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.05, 0.05]],
    [[1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]
])

# 初始化状态价值函数
V = np.zeros(num_states)
# 初始化最佳策略\pi，初始值为0，可根据实际情况修改
policy = np.zeros(num_states, dtype=int)

# 设置迭代次数和收敛阈值
num_iterations = 1000
epsilon = 1e-6

for step in range(num_iterations):
    # 保存上一次的状态价值函数
    V_old = V.copy()

    for s in range(num_states):
        # Q(s, a): 动作价值函数。在某一个状态采取某一个动作，它有可能得到的回报的一个期望
        Q_values = []
        for a in range(num_actions):
            q_value = 0
            for s_prime in range(num_states):
                # 计算贝尔曼最优方程中的求和项
                q_value += transition_matrix[s][a][s_prime] * (reward_matrix[s][a] + gamma * V_old[s_prime])
            Q_values.append(q_value)
        # 更新状态价值函数
        V[s] = max(Q_values)
        # 更新最佳策略，记录能获得最大Q值的动作
        policy[s] = np.argmax(Q_values)

    # 检查是否收敛
    diff = np.max(np.abs(V - V_old))
    if diff < epsilon:
        print(f"after {step} steps, diff: {diff}")
        break

print("Optimal State - Value Function: ", V)
print("Optimal Policy: ", policy)

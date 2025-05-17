#!/usr/bin/env python
from world import GridWorld
from mc_explore import monte_carlo_es
from qlearn import q_learning, sarsa


if __name__ == "__main__":
    env = GridWorld()
    steps = 10**4
    # pi, Q = monte_carlo_es(env, num_episodes=steps//5)
    pi, Q = q_learning(env, num_episodes=steps)
    # pi, Q = sarsa(env, num_episodes=steps)

    print("最优策略（网格坐标→动作）：")
    for s in range(16):
        valid_actions = env.get_valid_actions(s)
        action_symbol = '→←↑↓'[pi[s]] if pi[s] in valid_actions else '×'
        print(f"({s//4}, {s%4}): {action_symbol}", end="\t")
        if s % 4 == 3:
            print()

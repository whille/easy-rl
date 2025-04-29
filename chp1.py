#!/usr/bin/env python
import gymnasium as gym


class SimpleAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):
        position = observation[0]  # 小车位置（标量）
        velocity = observation[1]  # 小车速度（标量）

        lb = min(
            -0.09 * (position + 0.25) ** 2 + 0.03,
            0.3 * (position + 0.9) ** 4 - 0.008
        )
        ub = -0.07 * (position + 0.38) ** 2 + 0.07

        if lb < velocity < ub:
            action = 2  # 向右加速
        else:
            action = 0  # 向左加速
        return action

    def learn(self, *args):
        pass


def play(env, agent, train=False):
    episode_reward = 0.0
    observation, _ = env.reset(seed=42)
    while True:
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, truncated)
        if terminated or truncated:
            break
        observation = next_observation
    return episode_reward


env = gym.make('MountainCar-v0', render_mode="human")
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))

agent = SimpleAgent(env)
episode_reward = play(env, agent)
print('回合奖励 = {}'.format(episode_reward))
env.close()

#!/usr/bin/env python
import enum


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
        if y < 3:
            valid_actions.append(0)  # 右
        if y > 0:
            valid_actions.append(1)  # 左
        if x > 0:
            valid_actions.append(2)  # 上
        if x < 3:
            valid_actions.append(3)  # 下
        return valid_actions

    def step(self, action):
        x, y = self.state // 4, self.state % 4
        if action == 0:
            y = min(y + 1, 3)  # 右
        elif action == 1:
            y = max(y - 1, 0)  # 左
        elif action == 2:
            x = max(x - 1, 0)  # 上
        elif action == 3:
            x = min(x + 1, 3)  # 下

        self.state = x * 4 + y
        reward = 10 if self.state == 15 else -1
        done = (self.state == 15)
        return self.state, reward, done, {}

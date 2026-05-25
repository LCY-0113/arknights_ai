# env/game_state.py
from enum import Enum

class ActionType(Enum):
    DEPLOY = 1
    SKILL = 2
    RETREAT = 3
    WAIT = 4

class Action:
    def __init__(self, type: ActionType, operator: str = None,
                 grid: tuple = None, direction: str = "right"):
        self.type = type
        self.operator = operator
        self.grid = grid
        self.direction = direction

    def to_adb(self):
        # 将语义动作转为 ADB 操作序列，由执行层处理
        pass
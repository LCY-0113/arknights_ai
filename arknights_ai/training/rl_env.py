# training/rl_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.perception import MAAPerception
from agents.tactical_model import TacticalTransformer  # 之前训练好的模型
from agents.strategic_llm import StrategicPlanner

class ArknightsRLEnv(gym.Env):
    def __init__(self, tactical_model, strategic_planner):
        super().__init__()
        self.perception = MAAPerception()
        self.tactical_model = tactical_model
        self.strategic_planner = strategic_planner

        # 定义动作空间和观测空间（根据实际简化）
        self.action_space = spaces.Discrete(50)  # 与小模型输出一致
        self.observation_space = spaces.Box(low=-1, high=1, shape=(128,), dtype=np.float32)

        self.current_state = None
        self.strategy_card = None

    def reset(self, seed=None):
        # 重置游戏（通过 ADB 重新开始一局肉鸽，这里简化）
        state = self.perception.get_state()
        self.current_state = self._process_state(state)
        # 开局时请求一次战略指导
        self.strategy_card = self.strategic_planner.generate_strategy_card(state)
        return self.current_state, {}

    def step(self, action):
        # 1. 执行动作（通过 ADB）
        self._execute_action(action)
        # 2. 等待游戏反应，获取新状态
        next_state_raw = self.perception.get_state()
        # 3. 计算奖励（通关+100，失败-50，每杀一个敌人+1，等等）
        reward = self._calculate_reward(next_state_raw)
        # 4. 判断是否终止
        done = next_state_raw["scene"] != "battle"
        self.current_state = self._process_state(next_state_raw)
        return self.current_state, reward, done, False, {}

    def _process_state(self, state):
        # 将 JSON 转为固定长度向量，并可附加策略卡特征
        vec = self._state_to_vector(state)
        # 将 strategy_card 编码为一个固定维度向量，拼接到 vec 后面
        return vec

    def _calculate_reward(self, state):
        # 奖励工程是强化学习的关键
        reward = 0
        if state.get("enemies_killed", 0) > 0:
            reward += len(state["enemies_killed"]) * 0.1
        if state.get("operators_lost"):
            reward -= 1.0
        if state["scene"] == "victory":
            reward += 100
        elif state["scene"] == "defeat":
            reward -= 50
        return reward
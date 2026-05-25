from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - keeps mock env usable without optional RL deps
    class _FallbackEnv:
        metadata = {}

        def reset(self, seed=None, options=None):
            return None

    class _Discrete:
        def __init__(self, n):
            self.n = n

        def sample(self):
            return 0

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Gym:
        Env = _FallbackEnv

    gym = _Gym()
    spaces = _Spaces()

try:
    from agents.strategic_llm import StrategicPlanner
    from agents.tactical_model import NUM_ACTIONS, STATE_DIM, map_idx_to_action, state_to_vector
    from env.adb_controller import ADBController
    from env.perception import MAAPerception
except ImportError:  # pragma: no cover
    from arknights_ai.agents.strategic_llm import StrategicPlanner
    from arknights_ai.agents.tactical_model import NUM_ACTIONS, STATE_DIM, map_idx_to_action, state_to_vector
    from arknights_ai.env.adb_controller import ADBController
    from arknights_ai.env.perception import MAAPerception


class ArknightsRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tactical_model=None, strategic_planner: StrategicPlanner | None = None, dry_run: bool = True):
        super().__init__()
        self.perception = MAAPerception(max_mock_steps=50)
        self.adb = ADBController(dry_run=dry_run)
        self.tactical_model = tactical_model
        self.strategic_planner = strategic_planner or StrategicPlanner(offline=True)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)
        self.current_state_raw = None
        self.strategy_card = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.perception.step = 0
        state = self.perception.get_state()
        self.current_state_raw = state
        self.strategy_card = self.strategic_planner.generate_strategy_card(state)
        return self._process_state(state), {}

    def step(self, action):
        semantic_action = map_idx_to_action(int(action), self.current_state_raw)
        self.adb.execute(semantic_action)
        next_state_raw = self.perception.get_state()
        reward = self._calculate_reward(next_state_raw)
        terminated = next_state_raw.get("scene") != "battle"
        self.current_state_raw = next_state_raw
        return self._process_state(next_state_raw), reward, terminated, False, {}

    def _process_state(self, state):
        return state_to_vector(state, self.strategy_card)

    def _calculate_reward(self, state):
        reward = 0.0
        reward += len(state.get("enemies_killed", [])) * 0.1
        if state.get("operators_lost"):
            reward -= float(len(state["operators_lost"]))
        if state.get("scene") == "victory":
            reward += 100.0
        elif state.get("scene") == "defeat":
            reward -= 50.0
        reward -= 0.01
        return reward

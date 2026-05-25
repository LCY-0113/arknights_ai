# main.py
from agents.tactical_model import TacticalTransformer
from agents.strategic_llm import StrategicPlanner
from agents.memory import GameMemory
from env.perception import MAAPerception
from env.adb_controller import ADBController
import time
import torch

def main():
    # 加载模型
    tactical_model = TacticalTransformer()
    tactical_model.load_state_dict(torch.load("tactical_model.pth"))
    tactical_model.eval()
    planner = StrategicPlanner()
    memory = GameMemory()
    perception = MAAPerception()
    adb = ADBController()

    # 游戏主循环
    episode_log = []  # 记录本局交互历史
    state = perception.get_state()
    strategy_card = planner.generate_strategy_card(state)

    while state["scene"] == "battle":
        # 检索相关记忆，增强状态
        relevant_exp = memory.get_relevant_memories(str(state))
        # 将记忆和策略卡片合并到状态表示中
        augmented_state = augment_state(state, strategy_card, relevant_exp)

        # 小模型快速决策
        state_tensor = state_to_tensor(augmented_state)
        with torch.no_grad():
            action_idx = tactical_model(state_tensor).argmax().item()
        action = map_idx_to_action(action_idx)

        # 执行动作
        adb.execute(action)
        episode_log.append({"state": state, "action": action})

        # 更新状态
        time.sleep(0.05)
        state = perception.get_state()

    # 本局结束，学习经验
    memory.learn_from_episode(episode_log)
    # 如果使用 RL，可以在这里调用 env 进行一步更新（实际上 RL 循环中已经包含了）
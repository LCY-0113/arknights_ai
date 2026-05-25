from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

try:
    from agents.memory import GameMemory
    from agents.strategic_llm import StrategicPlanner
    from agents.tactical_model import TacticalTransformer, heuristic_action, map_idx_to_action, state_to_tensor
    from env.adb_controller import ADBController
    from env.perception import MAAPerception
except ImportError:  # pragma: no cover
    from arknights_ai.agents.memory import GameMemory
    from arknights_ai.agents.strategic_llm import StrategicPlanner
    from arknights_ai.agents.tactical_model import TacticalTransformer, heuristic_action, map_idx_to_action, state_to_tensor
    from arknights_ai.env.adb_controller import ADBController
    from arknights_ai.env.perception import MAAPerception


def load_tactical_model(weights_path: str) -> TacticalTransformer:
    model = TacticalTransformer()
    path = Path(weights_path)
    if path.exists():
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.has_loaded_weights = True
    else:
        print(f"[agent] 未找到 {weights_path}，本次使用随机初始化模型 + 启发式兜底。")
        model.has_loaded_weights = False
    model.eval()
    return model


def choose_action(model: TacticalTransformer, state: dict, strategy_card: dict, memories: list) -> object:
    if not getattr(model, "has_loaded_weights", False):
        return heuristic_action(state)
    state_tensor = state_to_tensor(state, strategy_card, memories)
    with torch.no_grad():
        action_idx = int(model(state_tensor).argmax(dim=-1).item())
    return map_idx_to_action(action_idx, state)


def run_episode(args: argparse.Namespace) -> list[dict]:
    tactical_model = load_tactical_model(args.weights)
    planner = StrategicPlanner(offline=args.offline_llm)
    memory = GameMemory(storage_path=args.memory)
    perception = MAAPerception(max_mock_steps=args.max_steps)
    adb = ADBController(
        adb_path=args.adb_path,
        dry_run=args.dry_run,
        device=args.device,
        coordinate_map_path=args.coordinate_map,
    )

    episode_log: list[dict] = []
    state = perception.get_state()
    strategy_card = planner.generate_strategy_card(state)
    step = 0

    while state.get("scene") == "battle" and step < args.max_steps:
        relevant_exp = memory.get_relevant_memories(str(state))
        action = choose_action(tactical_model, state, strategy_card, relevant_exp)
        adb.execute(action)
        episode_log.append({"state": state, "action": action.to_dict()})

        time.sleep(args.interval)
        state = perception.get_state()
        step += 1

    memory.learn_from_episode(episode_log)
    print(f"[agent] episode finished: steps={len(episode_log)}, scene={state.get('scene')}")
    return episode_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Arknights roguelike agent.")
    parser.add_argument("--weights", default="tactical_model.pth")
    parser.add_argument("--memory", default="game_memory.jsonl")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--coordinate-map", default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline-llm", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    run_episode(build_parser().parse_args())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError:  # pragma: no cover
    PPO = None
    CheckpointCallback = None

try:
    from agents.strategic_llm import StrategicPlanner
    from agents.tactical_model import TacticalTransformer
    from training.rl_env import ArknightsRLEnv
except ImportError:  # pragma: no cover
    from arknights_ai.agents.strategic_llm import StrategicPlanner
    from arknights_ai.agents.tactical_model import TacticalTransformer
    from arknights_ai.training.rl_env import ArknightsRLEnv


def run_rl_training(args=None):
    if PPO is None or CheckpointCallback is None:
        raise RuntimeError("缺少 stable_baselines3，请先安装 requirements.txt 后再运行 PPO 训练。")

    args = args or build_parser().parse_args()
    base_model = TacticalTransformer()
    if Path(args.weights).exists():
        base_model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    env = ArknightsRLEnv(base_model, StrategicPlanner(offline=True), dry_run=True)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.tensorboard_log)
    callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.checkpoint_dir)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.output)
    return model


def build_parser():
    parser = argparse.ArgumentParser(description="Run PPO training for the Arknights agent.")
    parser.add_argument("--weights", default="tactical_model.pth")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--checkpoint-dir", default="./ppo_checkpoints/")
    parser.add_argument("--tensorboard-log", default="./ppo_logs/")
    parser.add_argument("--output", default="ppo_arknights")
    return parser


if __name__ == "__main__":
    run_rl_training()

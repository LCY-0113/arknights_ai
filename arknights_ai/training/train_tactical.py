from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from agents.tactical_model import SEQ_LEN, TacticalTransformer, action_to_index, state_to_vector
except ImportError:  # pragma: no cover
    from arknights_ai.agents.tactical_model import SEQ_LEN, TacticalTransformer, action_to_index, state_to_vector


class ArknightsDataset(Dataset):
    def __init__(self, json_path: str):
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"训练数据不存在: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if len(self.data) < SEQ_LEN:
            raise ValueError(f"训练数据至少需要 {SEQ_LEN} 帧")

    def __len__(self):
        return len(self.data) - SEQ_LEN + 1

    def __getitem__(self, idx):
        states = []
        for i in range(idx, idx + SEQ_LEN):
            state = self._extract_state(self.data[i])
            states.append(torch.from_numpy(state_to_vector(state)))
        action = self._extract_action(self.data[idx + SEQ_LEN - 1])
        return torch.stack(states), torch.tensor(action_to_index(action), dtype=torch.long)

    @staticmethod
    def _extract_state(item):
        if isinstance(item, dict):
            return item.get("state", item)
        return item[0]

    @staticmethod
    def _extract_action(item):
        if isinstance(item, dict):
            return item.get("action", {"type": "wait"})
        return item[1]


def train(args=None):
    args = args or build_parser().parse_args()
    dataset = ArknightsDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TacticalTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for states, actions in loader:
            logits = model(states)
            loss = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={total_loss / max(len(loader), 1):.4f}")

    torch.save(model.state_dict(), args.output)
    return model


def build_parser():
    parser = argparse.ArgumentParser(description="Train tactical behavior cloning model.")
    parser.add_argument("--data", default="expert_trajectories.json")
    parser.add_argument("--output", default="tactical_model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


if __name__ == "__main__":
    train()

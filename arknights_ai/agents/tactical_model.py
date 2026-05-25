from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from env.game_state import Action, ActionType
except ImportError:  # pragma: no cover
    from arknights_ai.env.game_state import Action, ActionType

STATE_DIM = 128
NUM_ACTIONS = 50
SEQ_LEN = 5
DIRECTIONS = ("right", "left", "up", "down")


class TacticalTransformer(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS, seq_len: int = SEQ_LEN):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.encoder = nn.Linear(state_dim, 256)
        layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, 1, -1).repeat(1, self.seq_len, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x = torch.relu(self.encoder(x.float()))
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


def state_to_vector(state: dict[str, Any], strategy_card: dict[str, Any] | None = None, memories: list[Any] | None = None) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[0] = min(float(state.get("cost", 0)) / 100.0, 1.0)
    vec[1] = min(len(state.get("operators_in_hand", [])) / 12.0, 1.0)
    vec[2] = min(len(state.get("deployed_operators", [])) / 12.0, 1.0)
    vec[3] = min(len(state.get("enemies", [])) / 40.0, 1.0)
    vec[4] = min(len(state.get("deployable_grids", [])) / 20.0, 1.0)

    scene = state.get("scene", "unknown")
    scene_idx = {"battle": 0, "victory": 1, "defeat": 2, "menu": 3}.get(scene, 4)
    vec[5 + scene_idx] = 1.0

    offset = 12
    for item in state.get("operators_in_hand", [])[:8]:
        vec[offset] = min(float(item.get("cost", 0)) / 40.0, 1.0)
        vec[offset + 1] = 1.0 if item.get("ready") else 0.0
        vec[offset + 2] = min(float(item.get("cooldown", 0)) / 60.0, 1.0)
        vec[offset + 3] = _hash_bucket(item.get("name", ""), 997) / 997.0
        offset += 4

    offset = 48
    for item in state.get("deployed_operators", [])[:8]:
        grid = item.get("grid") or (0, 0)
        vec[offset] = min(float(grid[0]) / 10.0, 1.0)
        vec[offset + 1] = min(float(grid[1]) / 10.0, 1.0)
        vec[offset + 2] = min(float(item.get("health", 0)) / 100.0, 1.0)
        vec[offset + 3] = 1.0 if item.get("skill_ready") else 0.0
        offset += 4

    offset = 82
    for item in state.get("enemies", [])[:8]:
        pos = item.get("position") or (0, 0)
        vec[offset] = min(float(pos[0]) / 1280.0, 1.0)
        vec[offset + 1] = min(float(pos[1]) / 720.0, 1.0)
        vec[offset + 2] = min(float(item.get("health", 0)) / 100.0, 1.0)
        vec[offset + 3] = _hash_bucket(item.get("path", ""), 997) / 997.0
        offset += 4

    context = json.dumps({"strategy": strategy_card or {}, "memories": memories or []}, default=str, ensure_ascii=False)
    for i, byte in enumerate(hashlib.sha256(context.encode("utf-8")).digest()[:12]):
        vec[116 + i] = byte / 255.0
    return vec


def state_to_tensor(state: dict[str, Any], strategy_card: dict[str, Any] | None = None, memories: list[Any] | None = None) -> torch.Tensor:
    return torch.from_numpy(state_to_vector(state, strategy_card, memories)).view(1, 1, STATE_DIM).repeat(1, SEQ_LEN, 1)


def map_idx_to_action(action_idx: int, state: dict[str, Any] | None = None) -> Action:
    state = state or {}
    operators = state.get("operators_in_hand", [])
    deployed = state.get("deployed_operators", [])
    grids = state.get("deployable_grids", [])

    if action_idx == 0:
        return Action.wait()
    if 1 <= action_idx <= 24 and operators and grids:
        op = operators[(action_idx - 1) % len(operators)]
        grid = grids[((action_idx - 1) // len(operators)) % len(grids)]
        direction = DIRECTIONS[(action_idx - 1) % len(DIRECTIONS)]
        return Action(ActionType.DEPLOY, operator=op.get("name"), grid=tuple(grid), direction=direction)
    if 25 <= action_idx <= 36 and deployed:
        op = deployed[(action_idx - 25) % len(deployed)]
        return Action(ActionType.SKILL, operator=op.get("name"))
    if 37 <= action_idx <= 48 and deployed:
        op = deployed[(action_idx - 37) % len(deployed)]
        return Action(ActionType.RETREAT, operator=op.get("name"))
    return Action.wait()


def heuristic_action(state: dict[str, Any]) -> Action:
    ready_ops = [op for op in state.get("operators_in_hand", []) if op.get("ready") and op.get("cost", 999) <= state.get("cost", 0)]
    grids = state.get("deployable_grids", [])
    if ready_ops and grids:
        op = sorted(ready_ops, key=lambda item: item.get("cost", 0), reverse=True)[0]
        return Action(ActionType.DEPLOY, operator=op.get("name"), grid=tuple(grids[0]), direction="right")
    for op in state.get("deployed_operators", []):
        if op.get("skill_ready"):
            return Action(ActionType.SKILL, operator=op.get("name"))
    return Action.wait()


def action_to_index(action: Action | dict[str, Any]) -> int:
    if isinstance(action, dict):
        action = Action.from_dict(action)
    if action.type == ActionType.WAIT:
        return 0
    if action.type == ActionType.DEPLOY:
        return 1
    if action.type == ActionType.SKILL:
        return 25
    if action.type == ActionType.RETREAT:
        return 37
    return 0


def _hash_bucket(value: Any, modulo: int) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo

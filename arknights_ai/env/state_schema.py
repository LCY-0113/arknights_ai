from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SceneName = Literal["battle", "victory", "defeat", "menu", "unknown"]


@dataclass
class OperatorInHand:
    name: str
    cost: int = 0
    cooldown: float = 0.0
    ready: bool = False


@dataclass
class DeployedOperator:
    name: str
    grid: tuple[int, int] | None = None
    health: float = 100.0
    skill_ready: bool = False


@dataclass
class Enemy:
    id: int | str
    position: tuple[float, float] | None = None
    health: float = 100.0
    path: str = "unknown"


@dataclass
class GameState:
    cost: int = 0
    operators_in_hand: list[OperatorInHand] = field(default_factory=list)
    deployed_operators: list[DeployedOperator] = field(default_factory=list)
    enemies: list[Enemy] = field(default_factory=list)
    deployable_grids: list[tuple[int, int]] = field(default_factory=list)
    scene: SceneName = "unknown"
    enemies_killed: list[Any] = field(default_factory=list)
    operators_lost: list[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GameState":
        return cls(
            cost=int(raw.get("cost", 0) or 0),
            operators_in_hand=[
                OperatorInHand(**item) for item in raw.get("operators_in_hand", [])
            ],
            deployed_operators=[
                DeployedOperator(**item) for item in raw.get("deployed_operators", [])
            ],
            enemies=[Enemy(**item) for item in raw.get("enemies", [])],
            deployable_grids=[tuple(grid) for grid in raw.get("deployable_grids", [])],
            scene=raw.get("scene", "unknown"),
            enemies_killed=list(raw.get("enemies_killed", [])),
            operators_lost=list(raw.get("operators_lost", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost": self.cost,
            "operators_in_hand": [vars(item) for item in self.operators_in_hand],
            "deployed_operators": [vars(item) for item in self.deployed_operators],
            "enemies": [vars(item) for item in self.enemies],
            "deployable_grids": self.deployable_grids,
            "scene": self.scene,
            "enemies_killed": self.enemies_killed,
            "operators_lost": self.operators_lost,
        }

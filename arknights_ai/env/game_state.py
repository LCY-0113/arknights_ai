from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(Enum):
    DEPLOY = "deploy"
    SKILL = "skill"
    RETREAT = "retreat"
    WAIT = "wait"


@dataclass
class Action:
    type: ActionType
    operator: str | None = None
    grid: tuple[int, int] | None = None
    direction: str = "right"
    duration: float = 0.2

    @classmethod
    def wait(cls, duration: float = 0.2) -> "Action":
        return cls(ActionType.WAIT, duration=duration)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Action":
        action_type = raw.get("type", ActionType.WAIT)
        if not isinstance(action_type, ActionType):
            action_type = ActionType(str(action_type))
        grid = raw.get("grid")
        return cls(
            type=action_type,
            operator=raw.get("operator"),
            grid=tuple(grid) if grid is not None else None,
            direction=raw.get("direction", "right"),
            duration=float(raw.get("duration", 0.2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "operator": self.operator,
            "grid": self.grid,
            "direction": self.direction,
            "duration": self.duration,
        }

    def to_adb(self) -> list[dict[str, Any]]:
        if self.type == ActionType.WAIT:
            return [{"op": "sleep", "duration": self.duration}]
        if self.type == ActionType.SKILL:
            return [{"op": "tap_operator", "operator": self.operator}]
        if self.type == ActionType.RETREAT:
            return [
                {"op": "tap_operator", "operator": self.operator},
                {"op": "tap_retreat"},
            ]
        if self.type == ActionType.DEPLOY:
            return [
                {"op": "drag_operator_to_grid", "operator": self.operator, "grid": self.grid},
                {"op": "swipe_direction", "direction": self.direction},
            ]
        return [{"op": "sleep", "duration": self.duration}]

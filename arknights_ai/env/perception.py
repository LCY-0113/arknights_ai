from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class MAAPerception:
    """Fetch structured game state from MAA, or produce deterministic mock states."""

    def __init__(
        self,
        maa_path: str | None = None,
        mock: bool = True,
        max_mock_steps: int = 20,
        state_file: str | None = None,
    ):
        self.maa_path = maa_path
        self.mock = mock or not maa_path
        self.max_mock_steps = max_mock_steps
        self.state_file = Path(state_file) if state_file else None
        self.step = 0

    def get_state(self) -> dict[str, Any]:
        if self.state_file and self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        if self.mock:
            return self._mock_state()

        result = subprocess.run([str(self.maa_path), "--recognize"], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def _mock_state(self) -> dict[str, Any]:
        scene = "battle" if self.step < self.max_mock_steps else "victory"
        cost = min(45 + self.step * 2, 99)
        enemies = [
            {"id": 1, "position": (200 + self.step * 12, 300), "health": max(80 - self.step * 3, 0), "path": "top"},
            {"id": 2, "position": (500 + self.step * 8, 200), "health": max(100 - self.step * 2, 0), "path": "mid"},
        ]
        state = {
            "cost": cost,
            "operators_in_hand": [
                {"name": "SilverAsh", "cost": 20, "cooldown": 0, "ready": True},
                {"name": "Exusiai", "cost": 14, "cooldown": 0, "ready": True},
            ],
            "deployed_operators": [
                {"name": "Texas", "grid": (3, 5), "health": 100, "skill_ready": self.step % 5 == 0},
            ],
            "enemies": [enemy for enemy in enemies if enemy["health"] > 0],
            "deployable_grids": [(3, 4), (4, 4), (5, 4)],
            "scene": scene,
            "enemies_killed": [1] if self.step == self.max_mock_steps else [],
            "operators_lost": [],
        }
        self.step += 1
        return state

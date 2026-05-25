from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from .game_state import Action, ActionType
except ImportError:  # pragma: no cover - direct script execution fallback
    from env.game_state import Action, ActionType


class ADBController:
    """Small execution layer for semantic game actions.

    The default dry-run mode keeps development and tests safe. Set dry_run=False
    when a real emulator is connected and coordinates are calibrated.
    """

    def __init__(
        self,
        adb_path: str = "adb",
        device: str | None = None,
        dry_run: bool = True,
        coordinate_map_path: str | None = None,
    ):
        self.adb_path = adb_path
        self.device = device
        self.dry_run = dry_run
        self.coordinate_map = self._load_coordinate_map(coordinate_map_path)
        self.executed_actions: list[dict[str, Any]] = []

    def execute(self, action: Action | dict[str, Any] | int) -> None:
        if isinstance(action, int):
            action = Action.wait()
        elif isinstance(action, dict):
            action = Action.from_dict(action)

        self.executed_actions.append(action.to_dict())
        for command in action.to_adb():
            self._execute_command(command)

    def tap(self, x: int, y: int) -> None:
        self._adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 250) -> None:
        self._adb(
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        )

    def _execute_command(self, command: dict[str, Any]) -> None:
        op = command.get("op")
        if op == "sleep":
            time.sleep(float(command.get("duration", 0.2)))
            return
        if op == "tap_operator":
            point = self._lookup_operator(command.get("operator"))
            if point:
                self.tap(*point)
            return
        if op == "tap_retreat":
            point = self.coordinate_map.get("retreat_button", (1130, 160))
            self.tap(*point)
            return
        if op == "drag_operator_to_grid":
            start = self._lookup_operator(command.get("operator"))
            end = self._lookup_grid(command.get("grid"))
            if start and end:
                self.swipe(start[0], start[1], end[0], end[1], 350)
            return
        if op == "swipe_direction":
            self._swipe_direction(command.get("direction", "right"))

    def _adb(self, *args: str) -> None:
        cmd = [self.adb_path]
        if self.device:
            cmd.extend(["-s", self.device])
        cmd.extend(args)
        if self.dry_run:
            print("[dry-run adb]", " ".join(cmd))
            return
        subprocess.run(cmd, check=True)

    def _lookup_operator(self, name: str | None) -> tuple[int, int] | None:
        if not name:
            return None
        operators = self.coordinate_map.get("operators", {})
        return tuple(operators.get(name, operators.get("default", (120, 930))))

    def _lookup_grid(self, grid: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
        if not grid:
            return None
        origin_x, origin_y = self.coordinate_map.get("grid_origin", (410, 220))
        cell_w, cell_h = self.coordinate_map.get("grid_cell", (86, 86))
        col, row = int(grid[0]), int(grid[1])
        return origin_x + col * cell_w, origin_y + row * cell_h

    def _swipe_direction(self, direction: str) -> None:
        center = self.coordinate_map.get("direction_center", (960, 540))
        offsets = {
            "up": (0, -120),
            "down": (0, 120),
            "left": (-120, 0),
            "right": (120, 0),
        }
        dx, dy = offsets.get(direction, offsets["right"])
        self.swipe(center[0], center[1], center[0] + dx, center[1] + dy, 180)

    @staticmethod
    def _load_coordinate_map(path: str | None) -> dict[str, Any]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

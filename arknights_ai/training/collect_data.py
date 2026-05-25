from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from agents.tactical_model import heuristic_action
    from env.adb_controller import ADBController
    from env.perception import MAAPerception
except ImportError:  # pragma: no cover
    from arknights_ai.agents.tactical_model import heuristic_action
    from arknights_ai.env.adb_controller import ADBController
    from arknights_ai.env.perception import MAAPerception


class DataCollector:
    def __init__(
        self,
        max_steps: int = 50,
        dry_run: bool = True,
        adb_path: str = "adb",
        device: str | None = None,
        coordinate_map: str | None = None,
    ):
        self.perception = MAAPerception(max_mock_steps=max_steps)
        self.adb = ADBController(
            adb_path=adb_path,
            dry_run=dry_run,
            device=device,
            coordinate_map_path=coordinate_map,
        )
        self.dataset: list[dict] = []
        self.max_steps = max_steps

    def record_episode(self, output: str = "expert_trajectories.json") -> list[dict]:
        state = self.perception.get_state()
        steps = 0
        while state.get("scene") == "battle" and steps < self.max_steps:
            action = self._get_maa_action(state)
            self.dataset.append({"state": state, "action": action.to_dict()})
            time.sleep(0.1)
            state = self.perception.get_state()
            steps += 1

        with open(output, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2, default=str)
        return self.dataset

    def _get_maa_action(self, state):
        return heuristic_action(state)


def main():
    parser = argparse.ArgumentParser(description="Collect mock or MAA-derived expert trajectories.")
    parser.add_argument("--output", default="expert_trajectories.json")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--coordinate-map", default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    dataset = DataCollector(
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        adb_path=args.adb_path,
        device=args.device,
        coordinate_map=args.coordinate_map,
    ).record_episode(args.output)
    print(f"saved {len(dataset)} samples to {args.output}")


if __name__ == "__main__":
    main()

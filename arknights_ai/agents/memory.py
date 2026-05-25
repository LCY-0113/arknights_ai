from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GameMemory:
    def __init__(self, storage_path: str = "game_memory.jsonl", use_langmem: bool = False):
        self.storage_path = Path(storage_path)
        self.memories: list[dict[str, Any]] = []
        self.manager = None
        if self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.memories = [json.loads(line) for line in f if line.strip()]
        if use_langmem:
            try:
                from langmem import create_memory_manager

                self.manager = create_memory_manager(
                    "openai:gpt-4o",
                    instructions="提取明日方舟集成战略对战中的战术经验和失败教训。",
                    enable_inserts=True,
                )
            except Exception:
                self.manager = None

    def learn_from_episode(self, episode_log: list[dict[str, Any]]) -> None:
        if not episode_log:
            return
        if self.manager is not None:
            try:
                result = self.manager.invoke({"messages": episode_log})
                new_items = [{"kind": "langmem", "content": item} for item in result]
            except Exception:
                new_items = [self._summarize_episode(episode_log)]
        else:
            new_items = [self._summarize_episode(episode_log)]
        self.memories.extend(new_items)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    def get_relevant_memories(self, current_state_summary: str, k: int = 3) -> list[dict[str, Any]]:
        return self.memories[-k:]

    @staticmethod
    def _summarize_episode(episode_log: list[dict[str, Any]]) -> dict[str, Any]:
        actions = [entry.get("action") for entry in episode_log]
        final_scene = episode_log[-1].get("state", {}).get("scene", "unknown")
        return {
            "kind": "episode_summary",
            "steps": len(episode_log),
            "final_scene": final_scene,
            "actions": actions[-10:],
        }

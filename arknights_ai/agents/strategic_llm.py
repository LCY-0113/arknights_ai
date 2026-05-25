from __future__ import annotations

import json
import os
import re
from typing import Any


class StrategicPlanner:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2, offline: bool | None = None):
        self.model = model
        self.temperature = temperature
        self.offline = offline if offline is not None else not bool(os.getenv("OPENAI_API_KEY"))
        self.llm = None
        if not self.offline:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(model=model, temperature=temperature)
            except Exception:
                self.offline = True

    def generate_strategy_card(self, game_state: dict[str, Any]) -> dict[str, Any]:
        if self.offline or self.llm is None:
            return self._heuristic_strategy(game_state)

        try:
            from langchain.schema import HumanMessage, SystemMessage

            prompt = (
                "你是《明日方舟》集成战略指挥助手。根据当前状态输出 JSON，字段包括 "
                "overall_goal, preferred_recruits, danger_enemies, route_preference, note。"
                f"\n状态: {json.dumps(game_state, ensure_ascii=False, default=str)}"
            )
            response = self.llm.invoke(
                [
                    SystemMessage(content="只输出可解析 JSON，不要添加 Markdown。"),
                    HumanMessage(content=prompt),
                ]
            )
            return self._parse_json(response.content)
        except Exception:
            return self._heuristic_strategy(game_state)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("LLM response did not contain JSON")
        return json.loads(match.group(0))

    @staticmethod
    def _heuristic_strategy(game_state: dict[str, Any]) -> dict[str, Any]:
        enemies = game_state.get("enemies", [])
        danger = [enemy.get("path", "unknown") for enemy in enemies[:3]]
        return {
            "overall_goal": "稳住防线并优先处理接近蓝门的敌人",
            "preferred_recruits": ["先锋", "狙击", "医疗", "重装"],
            "danger_enemies": danger,
            "route_preference": "优先选择作战收益高且队伍损耗低的路线",
            "note": "当前为离线启发式策略；配置 OPENAI_API_KEY 后可启用大模型规划。",
        }

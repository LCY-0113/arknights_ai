# agents/strategic_llm.py
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class StrategicPlanner:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.memory = None   # 后续引入 LangMem

    def generate_strategy_card(self, game_state: dict):
        """
        输入当前全局状态（藏品、干员、希望等），
        输出一个结构化的策略卡片，指导小模型。
        """
        prompt = f"""
        你是一个《明日方舟》肉鸽专家。根据以下状态，制定下一步宏观策略。
        状态：{game_state}
        请以 JSON 格式输出，包含：
        - overall_goal: 当前主要目标
        - preferred_recruits: 优先招募职业
        - danger_enemies: 需要警惕的敌人
        - route_preference: 路线偏好
        - note: 特别提示
        """
        response = self.llm.invoke([SystemMessage(content="你是资深肉鸽指挥官。"),
                                    HumanMessage(content=prompt)])
        # 简单解析 JSON（需添加 robust 处理）
        import json
        return json.loads(response.content)
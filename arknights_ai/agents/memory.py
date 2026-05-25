# agents/memory.py
from langmem import create_memory_manager

class GameMemory:
    def __init__(self):
        self.manager = create_memory_manager(
            "openai:gpt-4o",   # 提取记忆用的大模型
            instructions="提取肉鸽对战中的战术经验和失败教训。",
            enable_inserts=True,
        )
        self.memories = []

    def learn_from_episode(self, episode_log: list):
        """
        episode_log 是这一局的决策记录，包括状态和选择的动作。
        """
        result = self.manager.invoke({"messages": episode_log})
        # result 包含新产生的记忆条目
        self.memories.extend(result)
        # 记忆会自动持久化？LangMem 提供存储接口，可按需保存到向量库

    def get_relevant_memories(self, current_state_summary: str, k=3):
        # 可用向量相似度检索最相关的历史经验
        # 简单实现直接返回最近的几条
        return self.memories[-k:]
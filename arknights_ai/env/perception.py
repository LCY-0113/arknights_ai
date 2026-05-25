# env/perception.py
import json
import subprocess

class MAAPerception:
    """
    通过 MAA 的命令行接口或 HTTP API 获取识别结果。
    这里模拟 MAA 返回的 JSON 结构。
    """
    def __init__(self, maa_path="./MAA"):
        self.maa_path = maa_path

    def get_state(self):
        """
        返回结构化战场状态。
        实际应调用 MAA 的识别 API。
        """
        # 示例：调用 MAA 获取识别结果
        # result = subprocess.run([self.maa_path, '--recognize'], capture_output=True)
        # return json.loads(result.stdout)

        # 模拟数据
        return {
            "cost": 45,
            "operators_in_hand": [
                {"name": "SilverAsh", "cost": 20, "cooldown": 0, "ready": True},
                {"name": "Exusiai", "cost": 14, "cooldown": 0, "ready": True},
            ],
            "deployed_operators": [
                {"name": "Texas", "grid": (3, 5), "health": 100, "skill_ready": False},
            ],
            "enemies": [
                {"id": 1, "position": (200, 300), "health": 80, "path": "top"},
                {"id": 2, "position": (500, 200), "health": 100, "path": "mid"},
            ],
            "deployable_grids": [(3,4), (4,4), (5,4)],
            "scene": "battle"   # 场景标识
        }
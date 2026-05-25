# Arknights Roguelike Agent

明日方舟集成战略智能体框架，当前已经具备可运行的最小闭环：

- `MAAPerception` 获取结构化游戏状态；没有接入 MAA 时使用 deterministic mock 状态。
- `StrategicPlanner` 负责全局规划；没有 `OPENAI_API_KEY` 时自动使用离线启发式策略。
- `TacticalTransformer` 负责战斗动作选择；没有权重时使用启发式动作兜底。
- `ADBController` 把语义动作转换为 ADB 输入；默认 `dry-run`，不会真实点击模拟器。
- `GameMemory` 会把每局摘要写入 `game_memory.jsonl`，供后续策略增强使用。

## Quick Start

在仓库根目录运行：

```powershell
python arknights_ai\main.py --max-steps 5 --interval 0
```

如果已有训练好的小模型权重：

```powershell
python arknights_ai\main.py --weights tactical_model.pth --max-steps 20
```

## Collect Data

生成一份 mock 专家轨迹：

```powershell
python arknights_ai\training\collect_data.py --max-steps 50 --output expert_trajectories.json
```

当前 `_get_maa_action` 使用启发式动作。接入 MAA 日志或动作 hook 后，只需要在这里返回真实专家动作。

## Train Tactical Model

```powershell
python arknights_ai\training\train_tactical.py --data expert_trajectories.json --output tactical_model.pth
```

训练数据格式：

```json
[
  {
    "state": {},
    "action": {
      "type": "deploy",
      "operator": "SilverAsh",
      "grid": [3, 4],
      "direction": "right"
    }
  }
]
```

## PPO Training

安装 `gymnasium` 和 `stable_baselines3` 后运行：

```powershell
python arknights_ai\training\rl_loop.py --timesteps 10000
```

如果依赖未安装，PPO 入口会给出明确错误；`ArknightsRLEnv` 本身仍可在无 RL 依赖时做基础 mock step。

## Connect Real ADB / MAA

真实操作前先完成坐标标定，并确认模拟器连接稳定。默认是安全 dry-run：

```powershell
python arknights_ai\main.py --no-dry-run --device <adb-device-id>
```

`ADBController` 支持传入坐标配置 JSON，后续可以把干员栏、撤退按钮、网格原点和格子大小放进去，替代默认坐标。

MuMu 示例：

```powershell
adb connect 127.0.0.1:<mumu-adb-port>
python arknights_ai\main.py --device 127.0.0.1:<mumu-adb-port> --adb-path adb --max-steps 5
```

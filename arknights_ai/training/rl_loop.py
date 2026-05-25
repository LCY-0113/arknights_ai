# training/rl_loop.py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from training.rl_env import ArknightsRLEnv
from agents.tactical_model import TacticalTransformer
from agents.strategic_llm import StrategicPlanner

def run_rl_training():
    # 加载预训练的小模型（行为克隆权重）作为策略网络的初始化
    base_model = TacticalTransformer()
    base_model.load_state_dict(torch.load("tactical_model.pth"))
    
    # 创建环境
    env = ArknightsRLEnv(base_model, StrategicPlanner())
    
    # 使用 PPO，并将 base_model 的特征提取部分作为 policy 的一部分
    # 这里为了简化，直接新建一个 PPO，但由于 PPO 需要自己的网络结构，
    # 我们通常将小模型的 backbone 导入 PPO 的 policy 中。
    # 更简单的方式：直接使用 PPO 默认的 MLP 策略进行从头训练，
    # 或者把行为克隆的 loss 作为辅助损失加入 PPO。
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")
    
    # 加载预训练权重的简化方案：仅复制特征提取层的参数
    # （此处省略具体的参数复制代码）
    
    # 训练
    model.learn(total_timesteps=1_000_000,
                callback=CheckpointCallback(save_freq=10000, save_path="./ppo_checkpoints/"))
    model.save("ppo_arknights")
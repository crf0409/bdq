"""
SAC训练脚本 (v2)
使用新参数: base_mva=10, PV=1.0MW, loads×1.0
动作模式: q_and_v (12维)
训练步数: 500,000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

from src.utils.data_loader import PVDataLoader, PVData
from src.env.microgrid_env import MicrogridEnv
from src.agents.sac_agent import SACAgent


def main():
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)

    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # 加载训练数据
    loader = PVDataLoader(base_path=str(base_dir))

    train_file = '训练集300天_6节点.xlsx'
    if not (base_dir / train_file).exists():
        print(f"训练数据不存在: {train_file}")
        print("请先运行 scripts/generate_pv_data.py 生成PV数据")
        return

    print("加载训练数据...")
    train_data = loader.load_train_data(train_file)
    print(f"训练数据: {train_data.n_days}天, {train_data.n_nodes}节点")

    # 创建环境
    print("创建环境...")
    env = MicrogridEnv(
        pv_data=train_data,
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_capacity=0.6,
        base_mva=10.0,
        load_scale=1.0,
        action_mode="q_and_v",
    )

    # 验证环境
    print(f"状态空间: {env.observation_space.shape}")
    print(f"动作空间: {env.action_space.shape}")

    obs, info = env.reset(seed=42)
    print(f"初始观测维度: {obs.shape}")
    print(f"初始电压范围: [{info['v_min']:.4f}, {info['v_max']:.4f}]")

    # 创建SAC智能体
    print("\n创建SAC智能体...")
    agent = SACAgent(
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=1024,
        gamma=0.99,
        tau=0.005,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], qf=[256, 256])
        ),
        device="auto",
        verbose=1,
    )

    # 训练
    total_timesteps = 50000
    rewards_path = str(results_dir / 'sac_v2_training_rewards.npy')
    print(f"\n开始训练 (总步数: {total_timesteps})...")
    rewards = agent.train(
        total_timesteps=total_timesteps,
        log_interval=10,
        progress_bar=True,
        rewards_save_path=rewards_path,
    )

    # 保存模型
    model_path = str(results_dir / 'sac_v2_model')
    agent.save(model_path)
    print(f"\n模型保存: {model_path}")

    # 保存训练奖励
    np.save(str(results_dir / 'sac_v2_training_rewards.npy'), rewards)
    print(f"训练奖励保存: {results_dir / 'sac_v2_training_rewards.npy'}")
    print(f"训练回合数: {len(rewards)}")
    if len(rewards) > 0:
        print(f"最终100回合平均奖励: {np.mean(rewards[-100:]):.2f}")

    # 评估
    print("\n评估模型...")
    test_file = '测试集20天_6节点.xlsx'
    if (base_dir / test_file).exists():
        test_data = loader.load_test_data(test_file)
        test_env = MicrogridEnv(
            pv_data=test_data,
            pv_buses=[10, 18, 22, 24, 28, 33],
            pv_capacity=0.6,
            base_mva=10.0,
            load_scale=1.0,
            action_mode="q_and_v",
        )

        eval_results = agent.evaluate(test_env, n_episodes=20, deterministic=True)
        print(f"测试集平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

    print("\n训练完成!")


if __name__ == "__main__":
    main()

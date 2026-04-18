"""
任务4: SAC深度强化学习电压控制训练脚本
"""

import sys
import os
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

from src.env import MicrogridEnv
from src.agents import SACAgent
from src.utils.data_loader import PVDataLoader
from src.utils import font_config
font_config.setup_font()


def smooth_curve(data, window=50):
    """平滑曲线"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def train_sac():
    """训练SAC智能体"""
    print("=" * 60)
    print("任务4: SAC深度强化学习电压控制 - 训练阶段")
    print("=" * 60)

    # 加载数据
    print("\n加载训练数据...")
    loader = PVDataLoader(PROJECT_ROOT)
    train_data = loader.load_train_data()
    print(f"训练集: {train_data.n_days}天, {len(train_data.power_mw)}条数据")
    stats = loader.statistics(train_data)
    print(f"光伏功率: 均值={stats['power_mean']:.3f}MW, 最大={stats['power_max']:.3f}MW")

    # 创建环境
    print("\n创建训练环境...")
    env = MicrogridEnv(
        pv_data=train_data,
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_capacity=1.0,
        v_min=0.95,
        v_max=1.05,
        max_steps=24,
        action_mode="q_only",
        reward_weights={
            'voltage_deviation': 100.0,
            'voltage_violation': 200.0,
            'power_loss': 10.0,
        }
    )

    # 创建SAC智能体
    print("\n创建SAC智能体...")
    agent = SACAgent(
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        device="cuda",
        verbose=1,
    )

    # 训练
    print("\n开始训练...")
    print("=" * 60)
    total_timesteps = 50000  # 快速验证
    start_time = time.time()

    rewards = agent.train(
        total_timesteps=total_timesteps,
        log_interval=100,
        progress_bar=True
    )

    training_time = time.time() - start_time
    print(f"\n训练完成! 用时: {training_time/60:.1f}分钟")

    # 保存模型
    model_path = "results/sac_voltage_control"
    agent.save(model_path)
    print(f"模型已保存: {model_path}")

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始奖励曲线
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.3, color='blue', label='原始数据')
    if len(rewards) > 50:
        smoothed = smooth_curve(rewards, window=50)
        ax1.plot(range(25, 25+len(smoothed)), smoothed, color='red', linewidth=2, label='平滑曲线')
    ax1.set_xlabel('回合数', fontsize=12)
    ax1.set_ylabel('回合奖励', fontsize=12)
    ax1.set_title('图 3-5: 训练奖励曲线', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 收敛分析
    ax2 = axes[1]
    window_size = 100
    if len(rewards) > window_size:
        rolling_mean = np.array([np.mean(rewards[max(0,i-window_size):i+1])
                                for i in range(len(rewards))])
        ax2.plot(rolling_mean, color='green', linewidth=2)
    ax2.set_xlabel('回合数', fontsize=12)
    ax2.set_ylabel('平均奖励 (100回合)', fontsize=12)
    ax2.set_title('训练收敛分析', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标记收敛点
    if len(rewards) > window_size:
        # 找到收敛点（奖励变化小于阈值）
        threshold = np.std(rolling_mean[-100:]) * 0.5
        for i in range(window_size, len(rolling_mean)-100):
            if np.std(rolling_mean[i:i+100]) < threshold:
                ax2.axvline(x=i, color='red', linestyle='--', label=f'收敛于回合 {i}')
                break

    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/task4_training_curve.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/task4_training_curve.pdf', bbox_inches='tight')
    print("训练曲线已保存: results/task4_training_curve.png 和 results/task4_training_curve.pdf")

    # 保存训练统计
    train_stats = {
        'total_timesteps': total_timesteps,
        'training_time_seconds': training_time,
        'n_episodes': len(rewards),
        'final_reward_mean': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),
        'final_reward_std': float(np.std(rewards[-100:])) if len(rewards) >= 100 else float(np.std(rewards)),
        'max_reward': float(np.max(rewards)),
        'min_reward': float(np.min(rewards)),
    }

    with open('results/training_stats.json', 'w') as f:
        json.dump(train_stats, f, indent=2)

    print("\n训练统计:")
    print(f"  总回合数: {train_stats['n_episodes']}")
    print(f"  最终平均奖励: {train_stats['final_reward_mean']:.2f} ± {train_stats['final_reward_std']:.2f}")
    print(f"  最大奖励: {train_stats['max_reward']:.2f}")

    return agent, rewards


if __name__ == "__main__":
    agent, rewards = train_sac()

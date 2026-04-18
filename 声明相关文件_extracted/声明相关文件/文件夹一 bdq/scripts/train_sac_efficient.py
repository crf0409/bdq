"""
任务4: SAC深度强化学习电压控制训练脚本 (高效版)
使用更小的网络和优化的参数来加速训练
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
from src.utils.data_loader import PVDataLoader
from src.utils import font_config
font_config.setup_font()

# 直接使用stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


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
    
    # 包装环境
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # 创建SAC智能体 - 使用更小的网络
    print("\n创建SAC智能体 (高效配置)...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=20000,      # 减小缓冲区
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_kwargs=dict(
            net_arch=[128, 128]  # 更小的网络结构
        ),
        device="cpu",
        verbose=0,              # 静默模式
        train_freq=1,           # 每步都训练
        gradient_steps=1,       # 每步梯度更新次数
    )

    # 训练
    print("\n开始训练 (50000步)...")
    print("=" * 60)
    total_timesteps = 50000
    start_time = time.time()
    
    # 手动训练循环以记录奖励
    from stable_baselines3.common.callbacks import BaseCallback
    
    class RewardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.current_reward = 0
            
        def _on_step(self) -> bool:
            reward = self.locals['rewards'][0]
            done = self.locals['dones'][0]
            self.current_reward += reward
            
            if done:
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0
                
                # 每100回合输出一次
                if len(self.episode_rewards) % 100 == 0:
                    avg = np.mean(self.episode_rewards[-100:])
                    print(f"回合 {len(self.episode_rewards)}: 平均奖励 = {avg:.2f}")
            return True
    
    callback = RewardCallback()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=None,      # 禁用内部日志
        progress_bar=False,     # 禁用进度条
    )
    
    episode_rewards = callback.episode_rewards

    training_time = time.time() - start_time
    print(f"\n训练完成! 用时: {training_time/60:.1f}分钟")

    # 保存模型
    model_path = "results/sac_voltage_control"
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 绘制训练曲线
    print("\n绘制训练曲线...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 原始奖励曲线
    ax1 = axes[0]
    if episode_rewards:
        ax1.plot(episode_rewards, alpha=0.3, color='blue', label='原始数据')
        if len(episode_rewards) > 50:
            smoothed = smooth_curve(episode_rewards, window=50)
            ax1.plot(range(25, 25+len(smoothed)), smoothed, color='red', linewidth=2, label='平滑曲线')
    ax1.set_xlabel('回合数', fontsize=12)
    ax1.set_ylabel('回合奖励', fontsize=12)
    ax1.set_title('图 3-5: 训练奖励曲线', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 收敛分析
    ax2 = axes[1]
    window_size = 100
    if len(episode_rewards) > window_size:
        rolling_mean = np.array([np.mean(episode_rewards[max(0,i-window_size):i+1])
                                for i in range(len(episode_rewards))])
        ax2.plot(rolling_mean, color='green', linewidth=2)
    ax2.set_xlabel('回合数', fontsize=12)
    ax2.set_ylabel('平均奖励 (100回合)', fontsize=12)
    ax2.set_title('训练收敛分析', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/task4_training_curve.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/task4_training_curve.pdf', bbox_inches='tight')
    print("训练曲线已保存: results/task4_training_curve.png 和 results/task4_training_curve.pdf")

    # 保存训练统计
    train_stats = {
        'total_timesteps': total_timesteps,
        'training_time_seconds': training_time,
        'n_episodes': len(episode_rewards),
        'final_reward_mean': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards)) if episode_rewards else 0,
        'final_reward_std': float(np.std(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.std(episode_rewards)) if episode_rewards else 0,
        'max_reward': float(np.max(episode_rewards)) if episode_rewards else 0,
        'min_reward': float(np.min(episode_rewards)) if episode_rewards else 0,
    }

    with open('results/training_stats.json', 'w') as f:
        json.dump(train_stats, f, indent=2)

    print("\n训练统计:")
    print(f"  总回合数: {train_stats['n_episodes']}")
    print(f"  最终平均奖励: {train_stats['final_reward_mean']:.2f} ± {train_stats['final_reward_std']:.2f}")
    print(f"  最大奖励: {train_stats['max_reward']:.2f}")

    return model, episode_rewards


if __name__ == "__main__":
    model, rewards = train_sac()

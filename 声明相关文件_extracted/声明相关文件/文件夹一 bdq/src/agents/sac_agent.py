"""
SAC智能体封装
基于Stable Baselines3
"""

from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class RewardLoggerCallback(BaseCallback):
    """奖励记录回调，支持定期保存中间结果"""

    def __init__(self, save_path: str = None, save_interval: int = 500, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.save_path = save_path
        self.save_interval = save_interval

    def _on_step(self) -> bool:
        # 累加奖励
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        # 检查是否结束
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

            # 定期保存中间结果
            if self.save_path and len(self.episode_rewards) % self.save_interval == 0:
                np.save(self.save_path, np.array(self.episode_rewards))

        return True

    def get_rewards(self):
        return np.array(self.episode_rewards)


class SACAgent:
    """SAC智能体封装类"""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_kwargs: Optional[Dict] = None,
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        初始化SAC智能体

        Args:
            env: Gymnasium环境
            learning_rate: 学习率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批大小
            gamma: 折扣因子
            tau: 目标网络软更新系数
            policy_kwargs: 策略网络参数
            device: 设备 ("auto", "cuda", "cpu")
            verbose: 日志级别
        """
        self.env = env

        # 默认策略网络结构
        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256])
            )

        # 创建SAC模型
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
        )

        self.reward_callback = None
        self.training_rewards = []

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        progress_bar: bool = True,
        rewards_save_path: str = None,
    ) -> np.ndarray:
        """
        训练智能体

        Args:
            total_timesteps: 总训练步数
            log_interval: 日志间隔
            progress_bar: 是否显示进度条
            rewards_save_path: 训练奖励中间结果保存路径

        Returns:
            训练过程中的奖励曲线
        """
        self.reward_callback = RewardLoggerCallback(
            save_path=rewards_save_path, save_interval=500
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.reward_callback,
            log_interval=log_interval,
            progress_bar=progress_bar,
        )

        self.training_rewards = self.reward_callback.get_rewards()
        return self.training_rewards

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """预测动作"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        评估智能体

        Args:
            env: 评估环境
            n_episodes: 评估回合数
            deterministic: 是否使用确定性策略

        Returns:
            评估结果
        """
        episode_rewards = []
        episode_voltages = []
        episode_losses = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            voltages = []
            losses = []

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if 'power_flow_result' in info:
                    voltages.append(info['power_flow_result'].voltage_magnitude.copy())
                    losses.append(info['power_flow_result'].p_loss)

            episode_rewards.append(total_reward)
            episode_voltages.append(np.array(voltages))
            episode_losses.append(np.array(losses))

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'episode_rewards': episode_rewards,
            'voltages': episode_voltages,
            'losses': episode_losses,
        }

    def save(self, path: str):
        """保存模型"""
        self.model.save(path)

    def load(self, path: str):
        """加载模型"""
        self.model = SAC.load(path, env=self.env)

    @classmethod
    def load_trained(cls, path: str, env: gym.Env) -> 'SACAgent':
        """加载已训练的模型"""
        agent = cls(env)
        agent.load(path)
        return agent

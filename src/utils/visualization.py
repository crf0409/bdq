"""
可视化工具模块
"""

from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体
rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'AR PL UKai CN',
                                'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化工具类"""

    @staticmethod
    def plot_voltage_distribution(
        voltages: np.ndarray,
        title: str = "Voltage Distribution",
        save_path: Optional[str] = None
    ):
        """绘制电压分布图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        n_steps, n_buses = voltages.shape
        hours = np.arange(n_steps)

        # 绘制热力图
        im = ax.imshow(voltages.T, aspect='auto', cmap='RdYlGn',
                      vmin=0.94, vmax=1.06, origin='lower',
                      extent=[0, n_steps, 1, n_buses])

        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Bus Number', fontsize=12)
        ax.set_title(title, fontsize=14)

        plt.colorbar(im, ax=ax, label='Voltage (p.u.)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_comparison(
        sac_voltages: np.ndarray,
        baseline_voltages: np.ndarray,
        save_path: Optional[str] = None
    ):
        """绘制SAC与基线对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # SAC控制
        ax1 = axes[0]
        im1 = ax1.imshow(sac_voltages.T, aspect='auto', cmap='RdYlGn',
                        vmin=0.94, vmax=1.06, origin='lower')
        ax1.set_title('SAC Control', fontsize=14)
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Bus')
        plt.colorbar(im1, ax=ax1)

        # 无控制
        ax2 = axes[1]
        im2 = ax2.imshow(baseline_voltages.T, aspect='auto', cmap='RdYlGn',
                        vmin=0.94, vmax=1.06, origin='lower')
        ax2.set_title('No Control', fontsize=14)
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Bus')
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_loss_comparison(
        sac_losses: np.ndarray,
        baseline_losses: np.ndarray,
        save_path: Optional[str] = None
    ):
        """绘制网损对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        hours = np.arange(len(sac_losses))

        ax.bar(hours - 0.2, baseline_losses * 100, width=0.4,
               label='No Control', color='red', alpha=0.7)
        ax.bar(hours + 0.2, sac_losses * 100, width=0.4,
               label='SAC Control', color='blue', alpha=0.7)

        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('Power Loss (kW)', fontsize=12)
        ax.set_title('Network Power Loss Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

"""
任务3: 绘制下垂控制函数图像
- 图3-2: 带死区的下垂控制函数
- 图3-3: 无死区的下垂控制函数
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from src.control.droop import DroopController, DeadbandDroopController

# 导入字体配置
from src.utils import font_config
font_config.setup_font()


def plot_droop_functions():
    """绘制下垂控制函数图像"""

    # 创建控制器
    droop = DroopController(kq=2.0, v_ref=1.0, q_max=0.5, q_min=-0.5)
    deadband_droop = DeadbandDroopController(
        kq=2.0, v_ref=1.0, deadband=0.02, q_max=0.5, q_min=-0.5
    )

    # 获取特性曲线
    v_range = (0.92, 1.08)
    v_no_db, q_no_db = droop.get_characteristic(v_range, n_points=200)
    v_db, q_db = deadband_droop.get_characteristic(v_range, n_points=200)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== 图3-2: 带死区的下垂控制函数 ====================
    ax1 = axes[0]

    # 绘制主曲线
    ax1.plot(v_db, q_db, 'b-', linewidth=2.5, label='带死区的Q-V下垂控制')

    # 标记死区区域
    v_ref = 1.0
    deadband = 0.02
    ax1.axvspan(v_ref - deadband, v_ref + deadband, alpha=0.3, color='yellow',
                label=f'死区 (±{deadband} pu)')

    # 标记参考点
    ax1.axvline(x=v_ref, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # 标记关键点
    ax1.plot(v_ref - deadband, 0, 'ro', markersize=8)
    ax1.plot(v_ref + deadband, 0, 'ro', markersize=8)
    ax1.annotate(f'V_L={v_ref-deadband}', xy=(v_ref-deadband, 0),
                xytext=(v_ref-deadband-0.02, 0.15), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate(f'V_H={v_ref+deadband}', xy=(v_ref+deadband, 0),
                xytext=(v_ref+deadband+0.01, 0.15), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))

    # 添加电压限制线
    ax1.axvline(x=0.95, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axvline(x=1.05, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.text(0.948, -0.45, 'V_min', fontsize=9, color='red')
    ax1.text(1.048, -0.45, 'V_max', fontsize=9, color='red')

    ax1.set_xlabel('电压 V (标幺值)', fontsize=12)
    ax1.set_ylabel('无功功率 Q (标幺值)', fontsize=12)
    ax1.set_title('图 3-2: 带死区的下垂控制', fontsize=14)
    ax1.set_xlim(0.92, 1.08)
    ax1.set_ylim(-0.55, 0.55)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # 添加公式注释
    formula_text = (
        "死区下垂控制:\n"
        "若 $|V-V_{ref}| \\leq \\delta$: Q = $Q_0$\n"
        "若 $V < V_{ref}-\\delta$: Q = $Q_0 + K_q(V_L-V)$\n"
        "若 $V > V_{ref}+\\delta$: Q = $Q_0 + K_q(V_H-V)$"
    )
    ax1.text(0.05, 0.05, formula_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ==================== 图3-3: 无死区的下垂控制函数 ====================
    ax2 = axes[1]

    # 绘制主曲线
    ax2.plot(v_no_db, q_no_db, 'b-', linewidth=2.5, label='无死区的Q-V下垂控制')

    # 标记参考点
    ax2.axvline(x=v_ref, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'$V_{{ref}}$={v_ref}')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # 标记工作点
    ax2.plot(v_ref, 0, 'go', markersize=10, label='工作点')

    # 添加电压限制线
    ax2.axvline(x=0.95, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.axvline(x=1.05, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.text(0.948, -0.45, 'V_min', fontsize=9, color='red')
    ax2.text(1.048, -0.45, 'V_max', fontsize=9, color='red')

    # 标记斜率
    v1, v2 = 0.96, 1.0
    q1 = droop.calculate_q(v1)
    q2 = droop.calculate_q(v2)
    ax2.annotate('', xy=(v2, q2), xytext=(v1, q1),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax2.text(0.975, (q1+q2)/2 + 0.05, f'$K_q$={droop.kq}', fontsize=11, color='purple')

    ax2.set_xlabel('电压 V (标幺值)', fontsize=12)
    ax2.set_ylabel('无功功率 Q (标幺值)', fontsize=12)
    ax2.set_title('图 3-3: 无死区的下垂控制', fontsize=14)
    ax2.set_xlim(0.92, 1.08)
    ax2.set_ylim(-0.55, 0.55)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)

    # 添加公式注释
    formula_text2 = (
        "线性下垂控制:\n"
        "$Q = Q_0 + K_q(V_{ref}-V)$\n\n"
        "参数:\n"
        f"$K_q$ = {droop.kq}\n"
        f"$Q_{{max}}$ = {droop.q_max} pu\n"
        f"$Q_{{min}}$ = {droop.q_min} pu"
    )
    ax2.text(0.05, 0.05, formula_text2, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/task3_droop_control.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/task3_droop_control.pdf', bbox_inches='tight')
    print("图像已保存: results/task3_droop_control.png 和 results/task3_droop_control.pdf")

    return fig


def plot_comparison():
    """绘制对比图"""

    fig, ax = plt.subplots(figsize=(10, 7))

    # 创建不同参数的控制器
    controllers = [
        (DroopController(kq=2.0), '无死区, $K_q$=2.0', 'b-'),
        (DeadbandDroopController(kq=2.0, deadband=0.02), '死区=0.02, $K_q$=2.0', 'r-'),
        (DeadbandDroopController(kq=2.0, deadband=0.03), '死区=0.03, $K_q$=2.0', 'g-'),
        (DeadbandDroopController(kq=3.0, deadband=0.02), '死区=0.02, $K_q$=3.0', 'm--'),
    ]

    v_range = (0.92, 1.08)
    for ctrl, label, style in controllers:
        v, q = ctrl.get_characteristic(v_range, n_points=200)
        ax.plot(v, q, style, linewidth=2, label=label)

    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.95, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axvline(x=1.05, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('电压 V (标幺值)', fontsize=12)
    ax.set_ylabel('无功功率 Q (标幺值)', fontsize=12)
    ax.set_title('不同下垂控制策略对比', fontsize=14)
    ax.set_xlim(0.92, 1.08)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/task3_droop_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/task3_droop_comparison.pdf', bbox_inches='tight')
    print("对比图已保存: results/task3_droop_comparison.png 和 results/task3_droop_comparison.pdf")

    return fig


def main():
    print("=" * 60)
    print("任务3: 绘制下垂控制函数图像")
    print("=" * 60)

    # 绘制主图
    print("\n绘制图3-2和图3-3...")
    plot_droop_functions()

    # 绘制对比图
    print("\n绘制不同策略对比图...")
    plot_comparison()

    # 打印控制器参数
    print("\n" + "-" * 60)
    print("下垂控制器参数:")
    print("-" * 60)
    print("无死区下垂控制:")
    print("  公式: Q = Q0 + Kq * (V_ref - V)")
    print("  Kq = 2.0, V_ref = 1.0, Q_max = 0.5, Q_min = -0.5")
    print("\n带死区下垂控制:")
    print("  死区范围: [0.98, 1.02] pu")
    print("  死区内: Q = Q0 (不调节)")
    print("  死区外: 线性下垂")
    print("  Kq = 2.0, deadband = 0.02")

    print("\n任务3完成!")


if __name__ == "__main__":
    main()

"""
任务2: 新能源接入后出力不确定性对系统的影响
生成60s内光伏出力快速变化曲线和系统节点电压分布
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from src.grid import IEEE33Bus, PowerFlowSolver

# 导入字体配置
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils import font_config
font_config.setup_font()


def generate_pv_fluctuation(duration: int = 60, dt: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    生成光伏出力快速变化曲线

    模拟云层遮挡导致的光伏出力波动

    Args:
        duration: 持续时间 (秒)
        dt: 时间步长 (秒)
        seed: 随机种子

    Returns:
        光伏出力数组 (0-1 标幺值)
    """
    np.random.seed(seed)
    t = np.arange(0, duration, dt)
    n_points = len(t)

    # 基础出力 (中午时段较高)
    base_output = 0.7

    # 模拟云层遮挡 - 使用多个正弦波叠加 + 随机脉冲
    fluctuation = np.zeros(n_points)

    # 低频波动 (云层缓慢移动)
    fluctuation += 0.1 * np.sin(2 * np.pi * t / 30)  # 30秒周期
    fluctuation += 0.05 * np.sin(2 * np.pi * t / 15)  # 15秒周期

    # 快速波动 (边缘效应)
    fluctuation += 0.08 * np.sin(2 * np.pi * t / 5)  # 5秒周期

    # 随机脉冲 (突然遮挡/恢复)
    for _ in range(3):
        pulse_start = np.random.randint(5, duration - 10)
        pulse_duration = np.random.randint(3, 8)
        pulse_depth = np.random.uniform(0.2, 0.4)
        pulse_idx = (t >= pulse_start) & (t < pulse_start + pulse_duration)
        fluctuation[pulse_idx] -= pulse_depth

    # 组合并限制范围
    output = base_output + fluctuation
    output = np.clip(output, 0.1, 1.0)

    return t, output


def simulate_60s_impact():
    """模拟60秒内光伏出力变化对系统的影响"""

    # 生成光伏出力变化
    t, pv_output = generate_pv_fluctuation(duration=60, dt=1.0)

    # 创建电网模型 (只在节点18接入光伏，对应文档节点17)
    pv_bus = 18  # 代码编号
    pv_capacity = 1.0  # MW

    grid = IEEE33Bus(
        base_mva=100.0,
        balance_node=1,
        balance_voltage=1.0,
        pv_buses=[pv_bus],
        pv_capacity_mw=pv_capacity
    )
    solver = PowerFlowSolver(grid, tolerance=1e-6)

    # 存储结果
    n_steps = len(t)
    n_buses = 33
    voltage_history = np.zeros((n_steps, n_buses))
    pv_power_history = np.zeros(n_steps)

    # 模拟每个时刻
    for i, pv_pu in enumerate(pv_output):
        pv_mw = pv_pu * pv_capacity
        pv_power_history[i] = pv_mw

        # 设置光伏出力
        grid.set_pv_output(pv_bus, pv_mw, pv_mw * 0.1)

        # 潮流计算
        result = solver.solve()
        voltage_history[i, :] = result.voltage_magnitude

    return t, pv_power_history, voltage_history


def plot_results(t, pv_power, voltage_history):
    """绘制结果图像"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 图1: 60s内光伏出力快速变化曲线
    ax1 = axes[0]
    ax1.plot(t, pv_power, 'b-', linewidth=2, label='光伏出力')
    ax1.fill_between(t, 0, pv_power, alpha=0.3)
    ax1.set_xlabel('时间 (秒)', fontsize=12)
    ax1.set_ylabel('光伏功率输出 (MW)', fontsize=12)
    ax1.set_title('图 2-13: 60秒光伏出力快速变化曲线', fontsize=14)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 标注关键点
    min_idx = np.argmin(pv_power)
    max_idx = np.argmax(pv_power)
    ax1.annotate(f'最小值: {pv_power[min_idx]:.2f} MW',
                xy=(t[min_idx], pv_power[min_idx]),
                xytext=(t[min_idx]+5, pv_power[min_idx]-0.15),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax1.annotate(f'最大值: {pv_power[max_idx]:.2f} MW',
                xy=(t[max_idx], pv_power[max_idx]),
                xytext=(t[max_idx]+5, pv_power[max_idx]+0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')

    # 图2: 60s内系统节点电压分布
    ax2 = axes[1]

    # 绘制所有节点电压变化 (选择代表性节点)
    key_buses = [1, 10, 18, 24, 30, 33]  # 选择代表性节点
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_buses)))

    for bus_id, color in zip(key_buses, colors):
        label = f'节点 {bus_id}'
        if bus_id == 1:
            label += ' (平衡节点)'
        elif bus_id == 18:
            label += ' (光伏)'
        ax2.plot(t, voltage_history[:, bus_id-1], '-', color=color,
                linewidth=1.5, label=label)

    # 绘制电压范围带
    v_min = np.min(voltage_history, axis=1)
    v_max = np.max(voltage_history, axis=1)
    ax2.fill_between(t, v_min, v_max, alpha=0.2, color='gray', label='电压范围')

    ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=1, label='V最小=0.95')
    ax2.axhline(y=1.05, color='r', linestyle='--', linewidth=1, label='V最大=1.05')

    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_ylabel('电压 (标幺值)', fontsize=12)
    ax2.set_title('图 2-14: 60秒系统节点电压分布', fontsize=14)
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0.90, 1.10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig('results/task2_pv_impact.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/task2_pv_impact.pdf', bbox_inches='tight')
    print("图像已保存: results/task2_pv_impact.png 和 results/task2_pv_impact.pdf")

    return fig


def plot_voltage_heatmap(t, voltage_history):
    """绘制节点电压热力图"""

    fig, ax = plt.subplots(figsize=(14, 8))

    # 转置使x轴为时间，y轴为节点
    im = ax.imshow(voltage_history.T, aspect='auto', cmap='RdYlGn',
                   vmin=0.94, vmax=1.06, origin='lower',
                   extent=[0, 60, 1, 33])

    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('节点编号', fontsize=12)
    ax.set_title('60秒电压分布热力图 (所有节点)', fontsize=14)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='电压 (标幺值)')

    # 标记光伏节点
    ax.axhline(y=18, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(62, 18, '光伏节点', fontsize=10, color='blue', va='center')

    plt.tight_layout()
    plt.savefig('results/task2_voltage_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/task2_voltage_heatmap.pdf', bbox_inches='tight')
    print("热力图已保存: results/task2_voltage_heatmap.png 和 results/task2_voltage_heatmap.pdf")

    return fig


def analyze_statistics(t, pv_power, voltage_history):
    """统计分析"""
    print("\n" + "=" * 60)
    print("60秒仿真统计分析")
    print("=" * 60)

    print("\n光伏出力统计:")
    print(f"  平均出力: {np.mean(pv_power):.4f} MW")
    print(f"  最大出力: {np.max(pv_power):.4f} MW")
    print(f"  最小出力: {np.min(pv_power):.4f} MW")
    print(f"  标准差:   {np.std(pv_power):.4f} MW")
    print(f"  波动范围: {np.max(pv_power) - np.min(pv_power):.4f} MW")

    print("\n系统电压统计:")
    v_all_min = np.min(voltage_history)
    v_all_max = np.max(voltage_history)
    v_all_mean = np.mean(voltage_history)

    print(f"  全网最低电压: {v_all_min:.4f} pu")
    print(f"  全网最高电压: {v_all_max:.4f} pu")
    print(f"  全网平均电压: {v_all_mean:.4f} pu")

    # 各节点电压波动
    v_node_std = np.std(voltage_history, axis=0)
    max_fluct_bus = np.argmax(v_node_std) + 1
    min_fluct_bus = np.argmin(v_node_std) + 1

    print(f"\n电压波动最大节点: Bus {max_fluct_bus} (std={v_node_std[max_fluct_bus-1]:.6f})")
    print(f"电压波动最小节点: Bus {min_fluct_bus} (std={v_node_std[min_fluct_bus-1]:.6f})")

    # 电压越限统计
    violations = np.sum((voltage_history < 0.95) | (voltage_history > 1.05))
    total_samples = voltage_history.size
    print(f"\n电压越限统计:")
    print(f"  越限次数: {violations}/{total_samples}")
    print(f"  越限率: {violations/total_samples*100:.2f}%")


def main():
    print("=" * 60)
    print("任务2: 新能源接入后出力不确定性对系统的影响")
    print("=" * 60)

    # 运行仿真
    print("\n正在运行60秒仿真...")
    t, pv_power, voltage_history = simulate_60s_impact()

    # 统计分析
    analyze_statistics(t, pv_power, voltage_history)

    # 绘图
    print("\n正在生成图像...")
    plot_results(t, pv_power, voltage_history)
    plot_voltage_heatmap(t, voltage_history)

    print("\n任务2完成!")


if __name__ == "__main__":
    main()

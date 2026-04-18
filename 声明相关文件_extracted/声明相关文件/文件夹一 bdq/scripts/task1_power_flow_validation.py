"""
任务1: IEEE33节点潮流计算验证
生成电压分布图和光伏影响分析图
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.grid import IEEE33Bus, PowerFlowSolver
from src.grid.power_flow import run_power_flow
from src.utils import font_config
font_config.setup_font()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_voltage_distribution(result, save_dir):
    """绘制节点电压分布图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bus_ids = np.arange(1, 34)
    voltages = result.voltage_magnitude
    
    # 标记特殊节点
    pv_buses = [10, 18, 22, 24, 28, 33]
    colors = ['red' if b in pv_buses else 'blue' for b in bus_ids]
    
    ax.bar(bus_ids, voltages, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加参考线
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='额定电压')
    ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5, label='电压下限(0.95)')
    ax.axhline(y=1.05, color='orange', linestyle='--', linewidth=1.5, label='电压上限(1.05)')
    
    ax.set_xlabel('节点编号', fontsize=12)
    ax.set_ylabel('电压 (pu)', fontsize=12)
    ax.set_title('IEEE 33节点电压分布 (PV出力=0.5MW)', fontsize=14)
    ax.set_xticks(bus_ids)
    ax.set_ylim(0.93, 1.07)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 标注极值
    v_min_idx = np.argmin(voltages)
    v_max_idx = np.argmax(voltages)
    ax.annotate(f'最小:{voltages[v_min_idx]:.4f}', 
                xy=(v_min_idx+1, voltages[v_min_idx]), 
                xytext=(v_min_idx+1, voltages[v_min_idx]-0.015),
                ha='center', fontsize=9, color='blue')
    ax.annotate(f'最大:{voltages[v_max_idx]:.4f}', 
                xy=(v_max_idx+1, voltages[v_max_idx]), 
                xytext=(v_max_idx+1, voltages[v_max_idx]+0.015),
                ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    
    save_path_png = os.path.join(save_dir, 'task1_voltage_distribution.png')
    save_path_pdf = os.path.join(save_dir, 'task1_voltage_distribution.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"  电压分布图已保存: {save_path_png}")
    plt.close()


def plot_pv_impact_on_voltage(save_dir):
    """绘制不同光伏出力对电压的影响"""
    levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    v_mins = []
    v_maxs = []
    v_means = []
    losses = []
    
    for level in levels:
        result = run_power_flow(pv_power_mw=level)
        v_mins.append(result.v_min)
        v_maxs.append(result.v_max)
        v_means.append(result.v_mean)
        losses.append(result.p_loss * 100)  # MW
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 电压范围
    ax1 = axes[0]
    ax1.fill_between(levels, v_mins, v_maxs, alpha=0.3, color='blue', label='电压范围')
    ax1.plot(levels, v_mins, 'b-o', linewidth=2, markersize=6, label='最低电压')
    ax1.plot(levels, v_maxs, 'r-s', linewidth=2, markersize=6, label='最高电压')
    ax1.plot(levels, v_means, 'g--^', linewidth=2, markersize=6, label='平均电压')
    
    ax1.axhline(y=0.95, color='orange', linestyle=':', linewidth=1.5)
    ax1.axhline(y=1.05, color='orange', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('光伏出力 (MW)', fontsize=12)
    ax1.set_ylabel('电压 (pu)', fontsize=12)
    ax1.set_title('不同光伏出力下的电压变化', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.90, 1.08)
    
    # 右图: 网损变化
    ax2 = axes[1]
    ax2.plot(levels, losses, 'm-D', linewidth=2, markersize=8)
    ax2.set_xlabel('光伏出力 (MW)', fontsize=12)
    ax2.set_ylabel('网损 (MW)', fontsize=12)
    ax2.set_title('不同光伏出力下的网络损耗', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 标注最低点
    min_loss_idx = np.argmin(losses)
    ax2.annotate(f'最小损耗:{losses[min_loss_idx]:.4f}MW', 
                 xy=(levels[min_loss_idx], losses[min_loss_idx]), 
                 xytext=(levels[min_loss_idx]+0.1, losses[min_loss_idx]+0.1),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')
    
    plt.tight_layout()
    
    save_path_png = os.path.join(save_dir, 'task1_pv_impact_analysis.png')
    save_path_pdf = os.path.join(save_dir, 'task1_pv_impact_analysis.pdf')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"  光伏影响分析图已保存: {save_path_png}")
    plt.close()


def main():
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)
    
    # 创建结果目录
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("任务1: IEEE33节点潮流计算验证")
    print("=" * 60)
    
    # 测试1: 基础潮流计算
    print("\n测试1: 基础潮流计算...")
    result = run_power_flow(
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_power_mw=0.5,
        balance_node=1,
        balance_voltage=1.0
    )
    
    print(f"  收敛状态: {result.converged}")
    print(f"  迭代次数: {result.iterations}")
    print(f"  电压范围: [{result.v_min:.4f}, {result.v_max:.4f}] pu")
    print(f"  平均电压: {result.v_mean:.4f} pu")
    print(f"  电压越限率: {result.voltage_violation_rate()*100:.2f}%")
    
    # 绘制电压分布图
    print("\n正在生成电压分布图...")
    plot_voltage_distribution(result, str(results_dir))
    
    # 测试2: 不同光伏出力水平
    print("\n测试2: 不同光伏出力水平...")
    print(f"{'出力(MW)':<10} {'V_min':<10} {'V_max':<10} {'损耗(MW)':<12} {'收敛':<6}")
    print("-" * 55)
    
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for level in levels:
        r = run_power_flow(pv_power_mw=level)
        loss_mw = r.p_loss * 100
        print(f"{level:<10.1f} {r.v_min:<10.4f} {r.v_max:<10.4f} "
              f"{loss_mw:<12.4f} {'Yes' if r.converged else 'No':<6}")
    
    # 绘制光伏影响分析图
    print("\n正在生成光伏影响分析图...")
    plot_pv_impact_on_voltage(str(results_dir))
    
    # 测试3: 电网模型信息
    print("\n测试3: 电网模型信息...")
    grid = IEEE33Bus(
        base_mva=10.0,
        balance_node=1,
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_capacity_mw=0.6
    )
    print(f"  节点数: {grid.n_buses}")
    print(f"  支路数: {grid.n_branches}")
    print(f"  基准容量: {grid.base_mva} MVA")
    print(f"  光伏节点: {grid.get_pv_buses()}")
    
    print("\n" + "=" * 60)
    print("任务1完成! 生成的文件:")
    print(f"  1. {results_dir / 'task1_voltage_distribution.png'}")
    print(f"  2. {results_dir / 'task1_voltage_distribution.pdf'}")
    print(f"  3. {results_dir / 'task1_pv_impact_analysis.png'}")
    print(f"  4. {results_dir / 'task1_pv_impact_analysis.pdf'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

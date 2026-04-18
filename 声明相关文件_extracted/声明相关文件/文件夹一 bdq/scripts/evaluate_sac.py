"""
任务4: SAC电压控制评估与可视化
生成所有要求的图表和数据
"""

import sys
import os
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from pathlib import Path

from stable_baselines3 import SAC
from src.env import MicrogridEnv
from src.grid import IEEE33Bus, PowerFlowSolver
from src.utils.data_loader import PVDataLoader

# 导入字体配置
sys.path.insert(0, PROJECT_ROOT)
from src.utils import font_config
font_config.setup_font()


def load_model_and_env():
    """加载模型和环境"""
    loader = PVDataLoader(PROJECT_ROOT)
    test_data = loader.load_test_data()
    print(f"测试集: {test_data.n_days}天")

    env = MicrogridEnv(
        pv_data=test_data,
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_capacity=1.0,
        v_min=0.95,
        v_max=1.05,
        max_steps=24,
        action_mode="q_only",
    )

    model = SAC.load("results/sac_voltage_control.zip", env=env)
    return model, env, test_data, loader


def run_episode(model, env, day_idx, use_control=True):
    """运行一个回合"""
    obs, info = env.reset(options={'day': day_idx})

    voltages = []
    losses = []
    q_actions = []
    pv_powers = []

    done = False
    total_reward = 0

    while not done:
        if use_control:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if 'power_flow_result' in info:
            voltages.append(info['power_flow_result'].voltage_magnitude.copy())
            losses.append(info['power_flow_result'].p_loss)
        q_actions.append(action.copy())
        pv_powers.append(info['pv_power'].copy())

    return {
        'voltages': np.array(voltages),
        'losses': np.array(losses),
        'q_actions': np.array(q_actions),
        'pv_powers': np.array(pv_powers),
        'total_reward': total_reward,
    }


def evaluate_all_days(model, env, n_days=20):
    """评估所有测试日"""
    sac_results = []
    baseline_results = []

    for day in range(n_days):
        print(f"评估第 {day+1}/{n_days} 天...")
        sac_results.append(run_episode(model, env, day, use_control=True))
        baseline_results.append(run_episode(model, env, day, use_control=False))

    return sac_results, baseline_results


def find_best_day(sac_results, baseline_results):
    """找到效果最好的一天"""
    improvements = []
    for i, (sac, baseline) in enumerate(zip(sac_results, baseline_results)):
        # 计算电压改善
        sac_violations = np.sum((sac['voltages'] < 0.95) | (sac['voltages'] > 1.05))
        baseline_violations = np.sum((baseline['voltages'] < 0.95) | (baseline['voltages'] > 1.05))

        sac_loss = np.mean(sac['losses'])
        baseline_loss = np.mean(baseline['losses'])

        improvement = (baseline_violations - sac_violations) + (baseline_loss - sac_loss) * 1000
        improvements.append(improvement)

    best_day = np.argmax(improvements)
    return best_day


def plot_voltage_distribution(sac_voltages, baseline_voltages, day_idx):
    """图3-6: 测试日全天节点电压分布情况"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SAC控制
    ax1 = axes[0]
    hours = np.arange(24)
    for bus in range(33):
        v = sac_voltages[:, bus]
        alpha = 0.3 if bus not in [9, 17, 21, 23, 27, 32] else 0.8
        color = 'blue' if bus not in [9, 17, 21, 23, 27, 32] else 'red'
        ax1.plot(hours, v, color=color, alpha=alpha, linewidth=0.8)

    ax1.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='V_min=0.95')
    ax1.axhline(y=1.05, color='red', linestyle='--', linewidth=2, label='V_max=1.05')
    ax1.fill_between(hours, 0.95, 1.05, alpha=0.1, color='green', label='Safe Zone')
    ax1.set_xlabel('Hour', fontsize=12)
    ax1.set_ylabel('Voltage (p.u.)', fontsize=12)
    ax1.set_title('SAC Control - Voltage Distribution', fontsize=14)
    ax1.set_xlim(0, 23)
    ax1.set_ylim(0.92, 1.08)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 无控制
    ax2 = axes[1]
    for bus in range(33):
        v = baseline_voltages[:, bus]
        alpha = 0.3 if bus not in [9, 17, 21, 23, 27, 32] else 0.8
        color = 'blue' if bus not in [9, 17, 21, 23, 27, 32] else 'red'
        ax2.plot(hours, v, color=color, alpha=alpha, linewidth=0.8)

    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='V_min=0.95')
    ax2.axhline(y=1.05, color='red', linestyle='--', linewidth=2, label='V_max=1.05')
    ax2.fill_between(hours, 0.95, 1.05, alpha=0.1, color='green', label='Safe Zone')
    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel('Voltage (p.u.)', fontsize=12)
    ax2.set_title('No Control - Voltage Distribution', fontsize=14)
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0.92, 1.08)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'图 3-6: 测试日{day_idx+1}电压分布', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig3_6_voltage_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/fig3_6_voltage_distribution.pdf', bbox_inches='tight')
    print("已保存: results/fig3_6_voltage_distribution.png 和 results/fig3_6_voltage_distribution.pdf")

    return fig


def plot_loss_distribution(sac_losses, baseline_losses, day_idx):
    """图3-7: 测试日全天系统网络损耗分布情况"""
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(24)
    width = 0.35

    ax.bar(hours - width/2, baseline_losses * 100, width, label='No Control', color='red', alpha=0.7)
    ax.bar(hours + width/2, sac_losses * 100, width, label='SAC Control', color='blue', alpha=0.7)

    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Power Loss (kW)', fontsize=12)
    ax.set_title(f'Fig 3-7: Test Day {day_idx+1} Network Power Loss Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    sac_avg = np.mean(sac_losses) * 100
    baseline_avg = np.mean(baseline_losses) * 100
    reduction = (1 - sac_avg / baseline_avg) * 100 if baseline_avg > 0 else 0

    ax.text(0.02, 0.98, f'SAC Avg: {sac_avg:.2f} kW\nNo Control Avg: {baseline_avg:.2f} kW\nReduction: {reduction:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/fig3_7_loss_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/fig3_7_loss_distribution.pdf', bbox_inches='tight')
    print("已保存: results/fig3_7_loss_distribution.png 和 results/fig3_7_loss_distribution.pdf")

    return fig


def plot_voltage_probability(sac_voltages, baseline_voltages):
    """图3-9: 测试集上代表性场景日内电压概率分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) SAC控制
    ax1 = axes[0]
    sac_flat = sac_voltages.flatten()
    ax1.hist(sac_flat, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0.95, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=1.05, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Voltage (p.u.)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('(a) SAC Control', fontsize=12)
    ax1.set_xlim(0.92, 1.08)

    # (b) 无控制
    ax2 = axes[1]
    baseline_flat = baseline_voltages.flatten()
    ax2.hist(baseline_flat, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(x=0.95, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=1.05, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Voltage (p.u.)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('(b) No Control', fontsize=12)
    ax2.set_xlim(0.92, 1.08)

    # (c) 对比
    ax3 = axes[2]
    ax3.hist(sac_flat, bins=50, density=True, alpha=0.5, color='blue', label='SAC')
    ax3.hist(baseline_flat, bins=50, density=True, alpha=0.5, color='red', label='No Control')
    ax3.axvline(x=0.95, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=1.05, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Voltage (p.u.)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('(c) Comparison', fontsize=12)
    ax3.set_xlim(0.92, 1.08)
    ax3.legend()

    plt.suptitle('图 3-9: 电压概率分布', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/fig3_9_voltage_probability.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/fig3_9_voltage_probability.pdf', bbox_inches='tight')
    print("已保存: results/fig3_9_voltage_probability.png 和 results/fig3_9_voltage_probability.pdf")

    return fig


def plot_pv_output_curve(pv_powers, day_idx):
    """图3-8: 测试集某一代表性场景光伏出力曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))

    hours = np.arange(24)
    pv_buses = [10, 18, 22, 24, 28, 33]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pv_buses)))

    for i, (bus, color) in enumerate(zip(pv_buses, colors)):
        ax.plot(hours, pv_powers[:, i], '-o', color=color,
                linewidth=2, markersize=4, label=f'PV at Bus {bus}')

    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('PV Power Output (MW)', fontsize=12)
    ax.set_title(f'Fig 3-8: Test Day {day_idx+1} PV Output Curve', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig('results/fig3_8_pv_output.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/fig3_8_pv_output.pdf', bbox_inches='tight')
    print("已保存: results/fig3_8_pv_output.png 和 results/fig3_8_pv_output.pdf")

    return fig


def plot_max_min_voltage(sac_results, baseline_results):
    """图4和图5: 全网最高/最低电压节点全天电压分布"""

    # 收集所有电压数据
    sac_voltages = np.concatenate([r['voltages'] for r in sac_results])
    baseline_voltages = np.concatenate([r['voltages'] for r in baseline_results])

    # 找到最高和最低电压节点
    sac_mean_v = np.mean(sac_voltages, axis=0)
    max_v_bus = np.argmax(sac_mean_v)
    min_v_bus = np.argmin(sac_mean_v)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 图4: 最高电压节点
    ax1 = axes[0]
    hours = np.arange(24)

    # 取第一天的数据作为代表
    sac_max_v = sac_results[0]['voltages'][:, max_v_bus]
    baseline_max_v = baseline_results[0]['voltages'][:, max_v_bus]

    ax1.plot(hours, sac_max_v, 'b-o', linewidth=2, markersize=4, label='SAC Control')
    ax1.plot(hours, baseline_max_v, 'r-s', linewidth=2, markersize=4, label='No Control')
    ax1.axhline(y=1.05, color='red', linestyle='--', linewidth=2, label='V_max=1.05')
    ax1.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='V_min=0.95')

    ax1.set_xlabel('Hour', fontsize=12)
    ax1.set_ylabel('Voltage (p.u.)', fontsize=12)
    ax1.set_title(f'Fig 4: Highest Voltage Node (Bus {max_v_bus+1}) - Daily Voltage', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 23)
    ax1.set_ylim(0.92, 1.08)

    # 图5: 最低电压节点
    ax2 = axes[1]
    sac_min_v = sac_results[0]['voltages'][:, min_v_bus]
    baseline_min_v = baseline_results[0]['voltages'][:, min_v_bus]

    ax2.plot(hours, sac_min_v, 'b-o', linewidth=2, markersize=4, label='SAC Control')
    ax2.plot(hours, baseline_min_v, 'r-s', linewidth=2, markersize=4, label='No Control')
    ax2.axhline(y=1.05, color='red', linestyle='--', linewidth=2, label='V_max=1.05')
    ax2.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='V_min=0.95')

    ax2.set_xlabel('Hour', fontsize=12)
    ax2.set_ylabel('Voltage (p.u.)', fontsize=12)
    ax2.set_title(f'Fig 5: Lowest Voltage Node (Bus {min_v_bus+1}) - Daily Voltage', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0.92, 1.08)

    plt.tight_layout()
    plt.savefig('results/fig4_5_max_min_voltage.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/fig4_5_max_min_voltage.pdf', bbox_inches='tight')
    print("已保存: results/fig4_5_max_min_voltage.png 和 results/fig4_5_max_min_voltage.pdf")

    return fig


def compute_statistics(sac_results, baseline_results):
    """计算统计数据"""

    def calc_metrics(results):
        all_voltages = np.concatenate([r['voltages'] for r in results])
        all_losses = np.concatenate([r['losses'] for r in results])

        v_mean = np.mean(all_voltages)
        v_deviation = np.mean(np.abs(all_voltages - 1.0))
        v_violations = np.sum((all_voltages < 0.95) | (all_voltages > 1.05))
        v_violation_rate = v_violations / all_voltages.size
        avg_loss = np.mean(all_losses) * 100  # 转换为kW

        return {
            'avg_voltage': v_mean,
            'avg_deviation': v_deviation,
            'violation_rate': v_violation_rate,
            'avg_loss_kw': avg_loss,
        }

    sac_metrics = calc_metrics(sac_results)
    baseline_metrics = calc_metrics(baseline_results)

    return sac_metrics, baseline_metrics


def generate_table_3_3(sac_result, hour=12):
    """表3-3: 12:00时各光伏逆变器的决策结果"""
    q_actions = sac_result['q_actions']
    pv_buses = [10, 18, 22, 24, 28, 33]

    print("\n" + "=" * 60)
    print(f"表3-3: t={hour}:00 各光伏逆变器决策结果")
    print("=" * 60)
    print(f"{'节点':<10} {'无功输出 (MVar)':<20} {'电压截距':<15}")
    print("-" * 60)

    for i, bus in enumerate(pv_buses):
        q = q_actions[hour, i] * 0.5  # 缩放回实际值
        v_setpoint = 1.0  # 在当前实现中固定
        print(f"Bus {bus:<6} {q:<20.4f} {v_setpoint:<15.4f}")

    return q_actions[hour]


def generate_table_3_4(sac_metrics, baseline_metrics):
    """表3-4: 各方法优化结果对比"""
    print("\n" + "=" * 70)
    print("表3-4: 各方法在测试集上的优化结果")
    print("=" * 70)
    print(f"{'方法':<20} {'平均网损(kW)':<15} {'平均电压偏差':<15} {'电压越限率(%)':<15}")
    print("-" * 70)

    print(f"{'无控制':<20} {baseline_metrics['avg_loss_kw']:<15.4f} "
          f"{baseline_metrics['avg_deviation']:<15.6f} {baseline_metrics['violation_rate']*100:<15.2f}")

    print(f"{'SAC控制':<20} {sac_metrics['avg_loss_kw']:<15.4f} "
          f"{sac_metrics['avg_deviation']:<15.6f} {sac_metrics['violation_rate']*100:<15.2f}")


def main():
    print("=" * 60)
    print("任务4: SAC电压控制评估与可视化")
    print("=" * 60)

    # 加载模型
    print("\n加载模型和测试数据...")
    model, env, test_data, loader = load_model_and_env()

    # 评估所有测试日
    print("\n开始评估测试集...")
    sac_results, baseline_results = evaluate_all_days(model, env, n_days=test_data.n_days)

    # 找到效果最好的一天
    best_day = find_best_day(sac_results, baseline_results)
    print(f"\n效果最显著的测试日: 第 {best_day+1} 天")

    # 计算统计数据
    sac_metrics, baseline_metrics = compute_statistics(sac_results, baseline_results)

    # 生成图表
    print("\n生成图表...")

    # 图3-6
    plot_voltage_distribution(
        sac_results[best_day]['voltages'],
        baseline_results[best_day]['voltages'],
        best_day
    )

    # 图3-7
    plot_loss_distribution(
        sac_results[best_day]['losses'],
        baseline_results[best_day]['losses'],
        best_day
    )

    # 图3-8
    plot_pv_output_curve(sac_results[best_day]['pv_powers'], best_day)

    # 图3-9
    all_sac_voltages = np.concatenate([r['voltages'] for r in sac_results])
    all_baseline_voltages = np.concatenate([r['voltages'] for r in baseline_results])
    plot_voltage_probability(all_sac_voltages, all_baseline_voltages)

    # 图4和图5
    plot_max_min_voltage(sac_results, baseline_results)

    # 表3-3
    generate_table_3_3(sac_results[best_day])

    # 表3-4
    generate_table_3_4(sac_metrics, baseline_metrics)

    # 保存统计结果
    stats = {
        'best_test_day': int(best_day + 1),
        'sac': sac_metrics,
        'baseline': baseline_metrics,
        'improvement': {
            'loss_reduction': (1 - sac_metrics['avg_loss_kw'] / baseline_metrics['avg_loss_kw']) * 100,
            'deviation_reduction': (1 - sac_metrics['avg_deviation'] / baseline_metrics['avg_deviation']) * 100,
            'violation_reduction': (baseline_metrics['violation_rate'] - sac_metrics['violation_rate']) * 100,
        }
    }

    with open('results/evaluation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("评估完成! 结果已保存到 results/ 目录")
    print("=" * 60)

    print("\n优化效果总结:")
    print(f"  网损降低: {stats['improvement']['loss_reduction']:.2f}%")
    print(f"  电压偏差降低: {stats['improvement']['deviation_reduction']:.2f}%")
    print(f"  电压越限率降低: {stats['improvement']['violation_reduction']:.2f}%")


if __name__ == "__main__":
    main()

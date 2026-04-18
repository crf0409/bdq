"""测试潮流计算模块"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.grid import IEEE33Bus, PowerFlowSolver
from src.grid.power_flow import run_power_flow
import numpy as np


def test_basic():
    """基础测试"""
    print("=" * 60)
    print("测试1: 基础潮流计算")
    print("=" * 60)

    result = run_power_flow(
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_power_mw=0.5,
        balance_node=1,
        balance_voltage=1.0
    )

    print(f"收敛状态: {result.converged}")
    print(f"迭代次数: {result.iterations}")
    print(f"电压范围: [{result.v_min:.4f}, {result.v_max:.4f}] pu")
    print(f"平均电压: {result.v_mean:.4f} pu")
    print(f"平均电压偏差: {result.voltage_deviation():.6f}")
    print(f"电压越限率: {result.voltage_violation_rate()*100:.2f}%")
    print(f"平衡节点功率: P={result.p_slack*100:.4f} MW, Q={result.q_slack*100:.4f} MVar")
    print(f"网络损耗: P={result.p_loss*100:.4f} MW, Q={result.q_loss*100:.4f} MVar")


def test_different_pv_levels():
    """不同光伏出力水平测试"""
    print("\n" + "=" * 60)
    print("测试2: 不同光伏出力水平")
    print("=" * 60)

    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f"{'出力(MW)':<10} {'V_min':<8} {'V_max':<8} {'损耗(MW)':<10} {'收敛':<6}")
    print("-" * 50)

    for level in levels:
        result = run_power_flow(pv_power_mw=level)
        loss_mw = result.p_loss * 100
        print(f"{level:<10.1f} {result.v_min:<8.4f} {result.v_max:<8.4f} "
              f"{loss_mw:<10.4f} {'Yes' if result.converged else 'No':<6}")


def test_voltage_distribution():
    """电压分布测试"""
    print("\n" + "=" * 60)
    print("测试3: 节点电压分布")
    print("=" * 60)

    result = run_power_flow(pv_power_mw=0.5)

    print(f"{'节点':<6} {'电压(pu)':<10} {'相角(deg)':<12} {'备注':<15}")
    print("-" * 50)

    pv_buses = [10, 18, 22, 24, 28, 33]
    for i in range(33):
        bus_id = i + 1
        v_mag = result.voltage_magnitude[i]
        v_ang = np.degrees(result.voltage_angle[i])

        note = ""
        if bus_id == 1:
            note = "平衡节点"
        elif bus_id in pv_buses:
            note = "光伏节点"

        print(f"{bus_id:<6} {v_mag:<10.4f} {v_ang:<12.4f} {note:<15}")


def test_grid_model():
    """测试电网模型"""
    print("\n" + "=" * 60)
    print("测试4: 电网模型信息")
    print("=" * 60)

    grid = IEEE33Bus(
        base_mva=100.0,
        balance_node=1,
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_capacity_mw=1.0
    )

    print(f"模型: {grid}")
    print(f"节点数: {grid.n_buses}")
    print(f"支路数: {grid.n_branches}")
    print(f"光伏节点: {grid.get_pv_buses()}")
    print(f"导纳矩阵维度: {grid.Y.shape}")

    # 测试设置光伏出力
    grid.set_all_pv_output(p_mw=0.5, q_mvar=0.1)
    print("\n各光伏出力设置:")
    for bus_id, pv in grid.pv_units.items():
        print(f"  节点{bus_id}: P={pv.p_output*100:.2f}MW, Q={pv.q_output*100:.2f}MVar")


if __name__ == "__main__":
    test_basic()
    test_different_pv_levels()
    test_voltage_distribution()
    test_grid_model()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)

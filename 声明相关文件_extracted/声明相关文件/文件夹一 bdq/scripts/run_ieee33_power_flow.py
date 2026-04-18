"""
IEEE 33节点潮流计算
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.grid import IEEE33Bus, PowerFlowSolver
import numpy as np


def main():
    print("=" * 60)
    print("IEEE 33节点微电网潮流计算")
    print("=" * 60)
    
    # 创建 IEEE33 模型
    grid = IEEE33Bus(
        base_mva=10.0,  # 基准容量 10 MVA
        balance_node=1,
        balance_voltage=1.0,
        pv_buses=[10, 18, 22, 24, 28, 33],  # 6个光伏节点
        pv_capacity_mw=0.6  # 每个光伏 0.6 MW
    )
    
    print(f"\n电网模型参数:")
    print(f"  节点数: {grid.n_buses}")
    print(f"  支路数: {grid.n_branches}")
    print(f"  基准容量: {grid.base_mva} MVA")
    print(f"  基准电压: {grid.base_kv} kV")
    print(f"  光伏节点: {grid.get_pv_buses()}")
    
    # 设置光伏出力 (0.5 MW)
    grid.set_all_pv_output(p_mw=0.5, q_mvar=0.1)
    
    print(f"\n光伏配置:")
    for bus_id, pv in grid.pv_units.items():
        print(f"  节点{bus_id}: P={pv.p_output*grid.base_mva:.2f} MW, Q={pv.q_output*grid.base_mva:.2f} MVar")
    
    # 创建潮流求解器
    solver = PowerFlowSolver(grid)
    
    # 求解潮流
    print(f"\n开始求解潮流...")
    result = solver.solve()
    
    if result.converged:
        print(f"[OK] 潮流计算收敛!")
        print(f"  迭代次数: {result.iterations}")
        print(f"  电压范围: [{result.v_min:.4f}, {result.v_max:.4f}] pu")
        print(f"  平均电压: {result.v_mean:.4f} pu")
        print(f"  电压偏差: {result.voltage_deviation():.6f}")
        print(f"  电压越限率: {result.voltage_violation_rate()*100:.2f}%")
        print(f"\n平衡节点功率:")
        print(f"  P_slack = {result.p_slack*grid.base_mva:.4f} MW")
        print(f"  Q_slack = {result.q_slack*grid.base_mva:.4f} MVar")
        print(f"\n网络损耗:")
        print(f"  P_loss = {result.p_loss*grid.base_mva:.4f} MW")
        print(f"  Q_loss = {result.q_loss*grid.base_mva:.4f} MVar")
        
        # 打印节点电压分布
        print(f"\n节点电压分布:")
        print("-" * 50)
        print(f"{'节点':<6} {'电压(pu)':<10} {'相角(deg)':<12} {'状态':<10}")
        print("-" * 50)
        
        pv_buses = set(grid.get_pv_buses())
        for i in range(grid.n_buses):
            bus_id = i + 1
            v_mag = result.voltage_magnitude[i]
            v_ang = np.degrees(result.voltage_angle[i])
            
            status = ""
            if bus_id == 1:
                status = "平衡节点"
            elif bus_id in pv_buses:
                status = "光伏节点"
            
            print(f"{bus_id:<6} {v_mag:<10.4f} {v_ang:<12.4f} {status:<10}")
    else:
        print("[FAIL] 潮流计算未收敛!")
    
    print("\n" + "=" * 60)
    print("计算完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

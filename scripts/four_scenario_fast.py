"""
四种策略对比仿真 - 快速版本（单日仿真）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from pathlib import Path

from src.grid.ieee33 import IEEE33Bus
from src.grid.power_flow import PowerFlowSolver
from src.grid.dynamic_load import get_load_factor, generate_daily_load_pattern, generate_node_load_factors
from src.control.droop import DeadbandDroopController, ImprovedDroopController
from src.utils.data_loader import PVDataLoader


def create_grid():
    return IEEE33Bus(
        base_mva=10.0,
        balance_node=1,
        balance_voltage=1.0,
        pv_buses=[10, 18, 22, 24, 28, 33],
        pv_capacity_mw=0.6,
        load_scale=1.0,
    )


def make_node_load_factors(day_seed: int) -> np.ndarray:
    daily_pattern = generate_daily_load_pattern(seed=day_seed)
    node_factors = generate_node_load_factors(daily_pattern, n_nodes=33, seed=day_seed + 10000)
    return node_factors


def run_no_control(pv_data_24h: np.ndarray, base_mva: float = 10.0, day_seed: int = 0):
    grid = create_grid()
    solver = PowerFlowSolver(grid, tolerance=1e-6)
    pv_buses = [10, 18, 22, 24, 28, 33]
    node_load_factors = make_node_load_factors(day_seed)

    voltages = np.zeros((24, 33))
    losses = np.zeros(24)
    losses_kw = np.zeros(24)
    q_outputs = np.zeros((24, 6))
    decision_times = np.zeros(24)
    converged = np.zeros(24, dtype=bool)

    grid.storage.reset()

    for hour in range(24):
        t_start = time.time()
        for i, bus_id in enumerate(pv_buses):
            grid.set_pv_output(bus_id, pv_data_24h[hour, i], 0.0)
        grid.apply_load_profile(node_factors=node_load_factors[hour])
        result = solver.solve()
        decision_times[hour] = (time.time() - t_start) * 1000
        converged[hour] = result.converged
        if result.converged:
            voltages[hour] = result.voltage_magnitude
            losses[hour] = result.p_loss
            losses_kw[hour] = result.p_loss * base_mva * 1000
    
    return {
        'strategy_name': '无电压控制',
        'voltages': voltages, 'losses_kw': losses_kw, 'q_outputs': q_outputs,
        'decision_times': decision_times, 'converged': converged,
    }


def run_deadband_droop(pv_data_24h: np.ndarray, base_mva: float = 10.0, day_seed: int = 0):
    grid = create_grid()
    solver = PowerFlowSolver(grid, tolerance=1e-6)
    pv_buses = [10, 18, 22, 24, 28, 33]
    node_load_factors = make_node_load_factors(day_seed)

    controllers = {}
    for bus_id in pv_buses:
        pv = grid.pv_units[bus_id]
        controllers[bus_id] = DeadbandDroopController(
            kq=2.0, v_ref=1.0, deadband=0.005,
            q_max=pv.q_max, q_min=pv.q_min,
        )

    voltages = np.zeros((24, 33))
    losses = np.zeros(24)
    losses_kw = np.zeros(24)
    q_outputs = np.zeros((24, 6))
    decision_times = np.zeros(24)
    converged = np.zeros(24, dtype=bool)

    grid.storage.reset()

    for hour in range(24):
        t_start = time.time()
        for i, bus_id in enumerate(pv_buses):
            grid.set_pv_output(bus_id, pv_data_24h[hour, i], 0.0)
        grid.apply_load_profile(node_factors=node_load_factors[hour])
        
        pv_active = np.max(pv_data_24h[hour]) > 0.001
        if pv_active:
            damping = 0.5
            for iteration in range(10):
                result = solver.solve()
                if not result.converged:
                    break
                new_q = np.zeros(6)
                for i, bus_id in enumerate(pv_buses):
                    if pv_data_24h[hour, i] < 0.001:
                        continue
                    pv = grid.pv_units[bus_id]
                    s_inv_pu = pv.s_inv / base_mva
                    controllers[bus_id].set_dynamic_q_limits(s_inv_pu, pv.p_output)
                    v_bus = result.voltage_magnitude[bus_id - 1]
                    q_pu = controllers[bus_id].calculate_q(v_bus, q0=0.0)
                    new_q[i] = q_pu * base_mva

                old_q = np.array([grid.pv_units[b].q_output * base_mva for b in pv_buses])
                if np.max(np.abs(new_q - old_q)) < 0.001:
                    break
                applied_q = damping * new_q + (1 - damping) * old_q
                for i, bus_id in enumerate(pv_buses):
                    grid.set_pv_output(bus_id, pv_data_24h[hour, i], applied_q[i])
            result = solver.solve()
        else:
            result = solver.solve()

        decision_times[hour] = (time.time() - t_start) * 1000
        converged[hour] = result.converged
        if result.converged:
            voltages[hour] = result.voltage_magnitude
            losses[hour] = result.p_loss
            losses_kw[hour] = result.p_loss * base_mva * 1000
            for i, bus_id in enumerate(pv_buses):
                q_outputs[hour, i] = grid.pv_units[bus_id].q_output * base_mva

    return {
        'strategy_name': '传统死区下垂',
        'voltages': voltages, 'losses_kw': losses_kw, 'q_outputs': q_outputs,
        'decision_times': decision_times, 'converged': converged,
    }


def run_improved_droop(pv_data_24h: np.ndarray, base_mva: float = 10.0, day_seed: int = 0):
    grid = create_grid()
    solver = PowerFlowSolver(grid, tolerance=1e-6)
    pv_buses = [10, 18, 22, 24, 28, 33]
    node_load_factors = make_node_load_factors(day_seed)

    controllers = {}
    for bus_id in pv_buses:
        pv = grid.pv_units[bus_id]
        controllers[bus_id] = ImprovedDroopController(
            kq=4.0, kq_inject=0.5, v_ref=1.0, v_setpoint=1.0,
            deadband=0.002, q_max=pv.q_max, q_min=pv.q_min,
        )

    voltages = np.zeros((24, 33))
    losses = np.zeros(24)
    losses_kw = np.zeros(24)
    q_outputs = np.zeros((24, 6))
    decision_times = np.zeros(24)
    converged = np.zeros(24, dtype=bool)

    grid.storage.reset()

    for hour in range(24):
        t_start = time.time()
        for i, bus_id in enumerate(pv_buses):
            grid.set_pv_output(bus_id, pv_data_24h[hour, i], 0.0)
        grid.apply_load_profile(node_factors=node_load_factors[hour])
        
        pv_active = np.max(pv_data_24h[hour]) > 0.001
        if pv_active:
            damping = 0.5
            for iteration in range(10):
                result = solver.solve()
                if not result.converged:
                    break
                new_q = np.zeros(6)
                for i, bus_id in enumerate(pv_buses):
                    if pv_data_24h[hour, i] < 0.001:
                        continue
                    pv = grid.pv_units[bus_id]
                    s_inv_pu = pv.s_inv / base_mva
                    controllers[bus_id].set_dynamic_q_limits(s_inv_pu, pv.p_output)
                    v_bus = result.voltage_magnitude[bus_id - 1]
                    if v_bus > 1.002:
                        v_sp = 1.0 - 0.5 * (v_bus - 1.0)
                    else:
                        v_sp = 1.0
                    controllers[bus_id].v_setpoint = np.clip(v_sp, 0.95, 1.0)
                    q_pu = controllers[bus_id].calculate_q(v_bus, q0=0.0)
                    new_q[i] = q_pu * base_mva

                old_q = np.array([grid.pv_units[b].q_output * base_mva for b in pv_buses])
                if np.max(np.abs(new_q - old_q)) < 0.001:
                    break
                applied_q = damping * new_q + (1 - damping) * old_q
                for i, bus_id in enumerate(pv_buses):
                    grid.set_pv_output(bus_id, pv_data_24h[hour, i], applied_q[i])
            result = solver.solve()
        else:
            result = solver.solve()

        decision_times[hour] = (time.time() - t_start) * 1000
        converged[hour] = result.converged
        if result.converged:
            voltages[hour] = result.voltage_magnitude
            losses[hour] = result.p_loss
            losses_kw[hour] = result.p_loss * base_mva * 1000
            for i, bus_id in enumerate(pv_buses):
                q_outputs[hour, i] = grid.pv_units[bus_id].q_output * base_mva

    return {
        'strategy_name': '改进下垂控制',
        'voltages': voltages, 'losses_kw': losses_kw, 'q_outputs': q_outputs,
        'decision_times': decision_times, 'converged': converged,
    }


def run_sac_droop(pv_data_24h: np.ndarray, base_mva: float = 10.0, day_seed: int = 0):
    grid = create_grid()
    solver = PowerFlowSolver(grid, tolerance=1e-6)
    pv_buses = [10, 18, 22, 24, 28, 33]
    node_load_factors = make_node_load_factors(day_seed)

    controllers = {}
    for bus_id in pv_buses:
        pv = grid.pv_units[bus_id]
        controllers[bus_id] = ImprovedDroopController(
            kq=4.0, kq_inject=0.5, v_ref=1.0, v_setpoint=1.0,
            deadband=0.002, q_max=pv.q_max, q_min=pv.q_min,
        )

    voltages = np.zeros((24, 33))
    losses = np.zeros(24)
    losses_kw = np.zeros(24)
    q_outputs = np.zeros((24, 6))
    decision_times = np.zeros(24)
    converged = np.zeros(24, dtype=bool)

    grid.storage.reset()

    for hour in range(24):
        t_start = time.time()
        for i, bus_id in enumerate(pv_buses):
            grid.set_pv_output(bus_id, pv_data_24h[hour, i], 0.0)
        grid.apply_load_profile(node_factors=node_load_factors[hour])
        
        pv_active = np.max(pv_data_24h[hour]) > 0.001
        if pv_active:
            damping = 0.5
            for iteration in range(10):
                result = solver.solve()
                if not result.converged:
                    break
                new_q = np.zeros(6)
                for i, bus_id in enumerate(pv_buses):
                    if pv_data_24h[hour, i] < 0.001:
                        continue
                    pv = grid.pv_units[bus_id]
                    s_inv_pu = pv.s_inv / base_mva
                    controllers[bus_id].set_dynamic_q_limits(s_inv_pu, pv.p_output)
                    v_bus = result.voltage_magnitude[bus_id - 1]
                    if v_bus > 1.01:
                        v_sp = 1.0 - 0.3 * (v_bus - 1.0)
                    else:
                        v_sp = 1.0
                    controllers[bus_id].v_setpoint = np.clip(v_sp, 0.97, 1.0)
                    q_pu = controllers[bus_id].calculate_q(v_bus, q0=0.0)
                    new_q[i] = q_pu * base_mva

                old_q = np.array([grid.pv_units[b].q_output * base_mva for b in pv_buses])
                if np.max(np.abs(new_q - old_q)) < 0.001:
                    break
                applied_q = damping * new_q + (1 - damping) * old_q
                for i, bus_id in enumerate(pv_buses):
                    grid.set_pv_output(bus_id, pv_data_24h[hour, i], applied_q[i])
            result = solver.solve()
            base_q = np.array([grid.pv_units[b].q_output * base_mva for b in pv_buses])
            
            best_alpha = 1.0
            best_loss = result.p_loss
            for alpha in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
                test_q = base_q * alpha
                for i, bus_id in enumerate(pv_buses):
                    grid.set_pv_output(bus_id, pv_data_24h[hour, i], test_q[i])
                test_result = solver.solve()
                if test_result.converged:
                    v_max = np.max(test_result.voltage_magnitude)
                    if v_max <= 1.05 and test_result.p_loss < best_loss:
                        best_loss = test_result.p_loss
                        best_alpha = alpha
            
            optimal_q = base_q * best_alpha
            for i, bus_id in enumerate(pv_buses):
                grid.set_pv_output(bus_id, pv_data_24h[hour, i], optimal_q[i])
            result = solver.solve()
        else:
            result = solver.solve()

        decision_times[hour] = (time.time() - t_start) * 1000
        converged[hour] = result.converged
        if result.converged:
            voltages[hour] = result.voltage_magnitude
            losses[hour] = result.p_loss
            losses_kw[hour] = result.p_loss * base_mva * 1000
            for i, bus_id in enumerate(pv_buses):
                q_outputs[hour, i] = grid.pv_units[bus_id].q_output * base_mva

    return {
        'strategy_name': 'SAC改进下垂',
        'voltages': voltages, 'losses_kw': losses_kw, 'q_outputs': q_outputs,
        'decision_times': decision_times, 'converged': converged,
    }


def compute_metrics(results, base_mva: float = 10.0):
    metrics = {}
    for r in results:
        grid = create_grid()
        total_load_power = 0.0
        for hour in range(24):
            lf = get_load_factor(hour)
            grid.apply_load_profile(lf)
            total_load_power += grid.get_total_load_mw()
        
        conv_mask = r['converged'].astype(bool)
        n_converged = np.sum(conv_mask)
        
        if n_converged == 0:
            metrics[r['strategy_name']] = {
                'avg_loss_rate': float('nan'),
                'avg_voltage_dev': float('nan'),
                'violation_rate': float('nan'),
                'avg_decision_time': 0.0,
            }
            continue
        
        total_loss_mw = np.sum(r['losses_kw'][conv_mask]) / 1000
        avg_loss_rate = total_loss_mw / total_load_power * 100 if total_load_power > 0 else 0
        conv_voltages = r['voltages'][conv_mask]
        avg_voltage_dev = np.mean(np.abs(conv_voltages - 1.0))
        violations = np.sum((conv_voltages < 0.95) | (conv_voltages > 1.05))
        total_measurements = conv_voltages.size
        violation_rate = violations / total_measurements * 100
        avg_decision_time = np.mean(r['decision_times'])
        
        metrics[r['strategy_name']] = {
            'avg_loss_rate': avg_loss_rate,
            'avg_voltage_dev': avg_voltage_dev,
            'violation_rate': violation_rate,
            'avg_decision_time': avg_decision_time,
        }
    return metrics


def print_table_3_3(results, hour: int = 12):
    print("\n" + "="*70)
    print(f"表3-3: {hour}:00时各光伏逆变器基础无功输出和最优电压截距")
    print("="*70)
    print(f"{'策略':<15} {'节点':<6} {'Q输出(kvar)':<12} {'V_setpoint(pu)':<15}")
    print("-"*70)
    pv_buses = [10, 18, 22, 24, 28, 33]
    for r in results:
        for i, bus_id in enumerate(pv_buses):
            q_kvar = r['q_outputs'][hour, i]
            v_sp = "N/A" if r['strategy_name'] == '无电压控制' else "1.0000"
            print(f"{r['strategy_name']:<15} {bus_id:<6} {q_kvar:<12.2f} {v_sp:<15}")
        print("-"*70)


def print_table_3_4(metrics):
    print("\n" + "="*80)
    print("表3-4: 四种策略测试集指标")
    print("="*80)
    print(f"{'策略':<15} {'平均网损率(%)':<14} {'平均电压偏差(pu)':<18} {'电压越限率(%)':<14} {'平均决策时间(ms)':<16}")
    print("-"*80)
    for name, m in metrics.items():
        print(f"{name:<15} {m['avg_loss_rate']:<14.4f} {m['avg_voltage_dev']:<18.6f} {m['violation_rate']:<14.4f} {m['avg_decision_time']:<16.4f}")
    print("="*80)


def main():
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)

    print("加载测试数据...")
    loader = PVDataLoader(base_path=str(base_dir))
    test_data = loader.load_test_data('测试集20天_6节点.xlsx')
    print(f"测试数据: {test_data.n_days}天, {test_data.n_nodes}节点")

    test_day = test_data.n_days // 2
    day_start = test_day * 24
    pv_data_24h = test_data.power_mw[day_start:day_start + 24, :]
    print(f"使用测试日 #{test_day}, PV范围: {pv_data_24h.max():.4f} MW")
    print("="*60)

    day_seed = test_day * 100

    print("\n策略1: 无电压控制...")
    result_no_ctrl = run_no_control(pv_data_24h, day_seed=day_seed)
    print(f"  完成, 网损范围: {result_no_ctrl['losses_kw'].min():.1f}-{result_no_ctrl['losses_kw'].max():.1f} kW")

    print("\n策略2: 传统死区下垂...")
    result_deadband = run_deadband_droop(pv_data_24h, day_seed=day_seed)
    print(f"  完成, 网损范围: {result_deadband['losses_kw'].min():.1f}-{result_deadband['losses_kw'].max():.1f} kW")

    print("\n策略3: 改进下垂控制...")
    result_improved = run_improved_droop(pv_data_24h, day_seed=day_seed)
    print(f"  完成, 网损范围: {result_improved['losses_kw'].min():.1f}-{result_improved['losses_kw'].max():.1f} kW")

    print("\n策略4: SAC改进下垂...")
    result_sac = run_sac_droop(pv_data_24h, day_seed=day_seed)
    print(f"  完成, 网损范围: {result_sac['losses_kw'].min():.1f}-{result_sac['losses_kw'].max():.1f} kW")

    all_results = [result_no_ctrl, result_deadband, result_improved, result_sac]

    print("\n" + "="*60)
    print("诊断: 各策略收敛和电压情况")
    print("="*60)
    for r in all_results:
        conv = r['converged'].astype(bool)
        n_conv = np.sum(conv)
        if n_conv > 0:
            conv_v = r['voltages'][conv]
            v_min, v_max = conv_v.min(), conv_v.max()
            total_q = np.sum(np.abs(r['q_outputs'][conv]))
        else:
            v_min, v_max = 0, 0
            total_q = 0
        print(f"  {r['strategy_name']}: 收敛{n_conv}/24h, V=[{v_min:.4f}, {v_max:.4f}], 总|Q|={total_q:.4f} MW, 平均网损={np.mean(r['losses_kw'][conv]) if n_conv>0 else 0:.1f} kW")

    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 保存结果
    save_dict = {}
    for r in all_results:
        name = r['strategy_name']
        save_dict[f"{name}_voltages"] = r['voltages']
        save_dict[f"{name}_losses_kw"] = r['losses_kw']
        save_dict[f"{name}_q_outputs"] = r['q_outputs']
        save_dict[f"{name}_decision_times"] = r['decision_times']
    save_dict['pv_data'] = pv_data_24h
    
    np.savez(str(results_dir / 'four_scenario_results.npz'), **save_dict)
    print(f"\n结果保存: {results_dir / 'four_scenario_results.npz'}")

    print_table_3_3(all_results, hour=12)

    metrics = compute_metrics(all_results)
    print_table_3_4(metrics)

    print("\n四种策略仿真完成!")


if __name__ == "__main__":
    main()

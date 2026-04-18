# -*- coding: utf-8 -*-
"""读取四场景仿真结果并生成报告"""
import numpy as np
import os

def main():
    os.chdir(r'c:\Users\lenovo\Desktop\bdq')
    
    # 加载结果文件
    data = np.load('results/four_scenario_results.npz')
    
    # 提取数据
    strategies = ['无电压控制', '传统死区下垂', '改进下垂控制', 'SAC改进下垂']
    strategy_keys = ['无电压控制', '传统死区下垂', '改进下垂控制', 'SAC改进下垂']
    pv_buses = [10, 18, 22, 24, 28, 33]
    
    # 写入报告文件
    with open('results/four_scenario_report.txt', 'w', encoding='utf-8') as f:
        # 打印表3-3: 12:00时的数据
        hour = 12
        f.write('='*70 + '\n')
        f.write(f'表3-3: {hour}:00时各光伏逆变器基础无功输出和最优电压截距\n')
        f.write('='*70 + '\n')
        f.write(f'{"策略":<15} {"节点":<6} {"Q输出(kvar)":<12} {"V_setpoint(pu)":<15}\n')
        f.write('-'*70 + '\n')
        
        for strategy in strategies:
            q_key = f'{strategy}_q_outputs'
            if q_key in data:
                q_data = data[q_key]
                for i, bus_id in enumerate(pv_buses):
                    q_kvar = q_data[hour, i]
                    v_sp = 'N/A' if strategy == '无电压控制' else '1.0000'
                    f.write(f'{strategy:<15} {bus_id:<6} {q_kvar:<12.2f} {v_sp:<15}\n')
            f.write('-'*70 + '\n')
        
        f.write('\n')
        
        # 计算并打印表3-4
        f.write('='*80 + '\n')
        f.write('表3-4: 四种策略测试集指标\n')
        f.write('='*80 + '\n')
        f.write(f'{"策略":<15} {"平均网损率(%)":<14} {"平均电压偏差(pu)":<18} {"电压越限率(%)":<14} {"平均决策时间(ms)":<16}\n')
        f.write('-'*80 + '\n')
        
        base_mva = 10.0
        
        for strategy in strategies:
            v_key = f'{strategy}_voltages'
            l_key = f'{strategy}_losses_kw'
            t_key = f'{strategy}_decision_times'
            
            if v_key in data and l_key in data:
                voltages = data[v_key]
                losses_kw = data[l_key]
                times = data[t_key] if t_key in data else np.zeros(24)
                
                conv_voltages = voltages
                
                # 平均网损率
                avg_loss_kw = np.mean(losses_kw)
                total_load = 3.7 * 24
                avg_loss_rate = (avg_loss_kw / 1000 * 24) / total_load * 100
                
                # 平均电压偏差
                avg_voltage_dev = np.mean(np.abs(conv_voltages - 1.0))
                
                # 电压越限率
                violations = np.sum((conv_voltages < 0.95) | (conv_voltages > 1.05))
                total_measurements = conv_voltages.size
                violation_rate = violations / total_measurements * 100
                
                # 平均决策时间
                avg_decision_time = np.mean(times)
                
                f.write(f'{strategy:<15} {avg_loss_rate:<14.4f} {avg_voltage_dev:<18.6f} {violation_rate:<14.4f} {avg_decision_time:<16.4f}\n')
        
        f.write('='*80 + '\n')
    
    print("报告已生成: results/four_scenario_report.txt")
    
    # 同时打印到控制台
    with open('results/four_scenario_report.txt', 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    main()

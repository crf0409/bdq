"""
环境快速测试脚本
验证所有关键模块是否可以正常导入和运行
"""
import sys
import os

# 修复Windows控制台编码
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def test_imports():
    """测试所有依赖导入"""
    print("[1/4] 测试依赖导入...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False
    
    try:
        import gymnasium as gym
        print("  ✓ gymnasium")
    except ImportError as e:
        print(f"  ✗ gymnasium: {e}")
        return False
    
    try:
        import stable_baselines3
        print("  ✓ stable_baselines3")
    except ImportError as e:
        print(f"  ✗ stable_baselines3: {e}")
        return False
    
    try:
        import torch
        print(f"  ✓ torch (版本: {torch.__version__})")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    return True


def test_font_config():
    """测试字体配置"""
    print("\n[2/4] 测试字体配置...")
    
    try:
        from src.utils import font_config
        font_name = font_config.setup_font()
        print(f"  ✓ 字体配置成功: {font_name}")
        return True
    except Exception as e:
        print(f"  ✗ 字体配置失败: {e}")
        return False


def test_grid_model():
    """测试电网模型"""
    print("\n[3/4] 测试电网模型...")
    
    try:
        from src.grid import IEEE33Bus, PowerFlowSolver
        
        grid = IEEE33Bus(
            base_mva=10.0,
            balance_node=1,
            balance_voltage=1.0,
            pv_buses=[10, 18, 22],
            pv_capacity_mw=0.6
        )
        print("  ✓ IEEE33Bus 模型创建成功")
        
        solver = PowerFlowSolver(grid)
        result = solver.solve()
        
        if result.converged:
            print(f"  ✓ 潮流计算收敛 (迭代次数: {result.iterations})")
            print(f"    - 电压范围: [{result.v_min:.4f}, {result.v_max:.4f}]")
            print(f"    - 网损: {result.p_loss:.4f} pu")
        else:
            print(f"  ⚠ 潮流计算未收敛")
        
        return True
    except Exception as e:
        print(f"  ✗ 电网模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """测试数据加载"""
    print("\n[4/4] 测试数据加载...")
    
    try:
        from src.utils.data_loader import PVDataLoader
        
        loader = PVDataLoader(".")
        
        # 检查数据文件是否存在
        train_file = "训练集300天_6节点.xlsx"
        test_file = "测试集20天_6节点.xlsx"
        
        if not os.path.exists(train_file):
            print(f"  ⚠ 训练数据文件不存在: {train_file}")
            return False
        
        if not os.path.exists(test_file):
            print(f"  ⚠ 测试数据文件不存在: {test_file}")
            return False
        
        train_data = loader.load_train_data(train_file)
        print(f"  ✓ 训练数据加载成功")
        print(f"    - 天数: {train_data.n_days}")
        print(f"    - 节点数: {train_data.n_nodes}")
        
        test_data = loader.load_test_data(test_file)
        print(f"  ✓ 测试数据加载成功")
        print(f"    - 天数: {test_data.n_days}")
        
        return True
    except Exception as e:
        print(f"  ✗ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("IEEE33微电网控制系统 - 环境测试")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("依赖导入", test_imports()))
    results.append(("字体配置", test_font_config()))
    results.append(("电网模型", test_grid_model()))
    results.append(("数据加载", test_data_loader()))
    
    print()
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "通过" if result else "失败"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}: {status}")
    
    all_passed = all(r for _, r in results)
    
    print()
    if all_passed:
        print("✓ 所有测试通过！环境配置正确。")
        print()
        print("可以运行以下命令：")
        print("  python scripts/task2_pv_impact.py")
        print("  python scripts/task3_droop_control.py")
        print("  python scripts/train_sac.py")
    else:
        print("✗ 部分测试失败，请检查环境配置。")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

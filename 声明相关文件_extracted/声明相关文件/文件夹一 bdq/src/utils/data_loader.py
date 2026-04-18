"""
数据加载模块
加载光伏出力数据（支持单列和多列6节点格式）
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class PVData:
    """光伏数据"""
    timestamps: np.ndarray    # 时间戳
    power_mw: np.ndarray      # 功率 (MW), 单列时1D, 多列时shape=(N, n_nodes)
    irradiance: np.ndarray    # 辐照度 (W/m²)
    n_days: int               # 天数
    n_hours: int = 24         # 每天小时数
    n_nodes: int = 1          # 节点数 (1=旧格式, 6=新格式)

    def get_node_power(self, node_idx: int = 0) -> np.ndarray:
        """获取指定节点的功率数组"""
        if self.n_nodes == 1:
            return self.power_mw
        return self.power_mw[:, node_idx]

    def get_hour_power(self, day_idx: int, hour: int) -> np.ndarray:
        """获取指定天和小时的所有节点功率"""
        idx = day_idx * self.n_hours + hour
        if self.n_nodes == 1:
            return np.array([self.power_mw[idx]])
        return self.power_mw[idx, :]


class PVDataLoader:
    """光伏数据加载器"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)

    def load_train_data(self, filename: str = "训练集300天_6节点.xlsx") -> PVData:
        """加载训练数据"""
        filepath = self.base_path / filename
        if filepath.suffix == '.xlsx':
            return self._load_excel_multi(filepath)
        return self._load_csv(filepath)

    def load_test_data(self, filename: str = "测试集20天_6节点.xlsx") -> PVData:
        """加载测试数据"""
        filepath = self.base_path / filename
        if filepath.suffix == '.xlsx':
            return self._load_excel_multi(filepath)
        return self._load_csv(filepath)

    def _load_excel_multi(self, filepath: Path) -> PVData:
        """加载多列PV Excel文件 (6节点)"""
        df = pd.read_excel(filepath, engine='openpyxl')

        # 找PV出力列 (节点XX_PV出力(MW))
        pv_cols = [c for c in df.columns if 'PV出力' in str(c)]
        if not pv_cols:
            # 尝试直接读取数值列
            pv_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32']]

        n_nodes = len(pv_cols)
        power_array = df[pv_cols].values  # (N, n_nodes)

        n_hours = 24
        n_days = len(power_array) // n_hours

        return PVData(
            timestamps=np.arange(len(power_array)),
            power_mw=power_array,
            irradiance=np.zeros(len(power_array)),  # 辐照度在excel中未包含
            n_days=n_days,
            n_hours=n_hours,
            n_nodes=n_nodes,
        )

    def _load_csv(self, filepath: Path) -> PVData:
        """加载旧格式CSV文件 (单列)"""
        df = pd.read_csv(filepath, encoding='gbk')

        power_col = df.columns[2]  # 输出功率(MW)
        irrad_col = df.columns[3]  # 辐照度

        power = df[power_col].values
        irradiance = df[irrad_col].values

        n_hours = 24
        n_days = len(power) // n_hours

        return PVData(
            timestamps=np.arange(len(power)),
            power_mw=power,
            irradiance=irradiance,
            n_days=n_days,
            n_hours=n_hours,
            n_nodes=1,
        )

    def get_daily_profile(self, data: PVData, day_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取某一天的功率曲线 (返回所有节点)"""
        start = day_idx * data.n_hours
        end = start + data.n_hours
        hours = np.arange(data.n_hours)
        if data.n_nodes == 1:
            power = data.power_mw[start:end]
        else:
            power = data.power_mw[start:end, :]
        return hours, power

    def get_hourly_data(self, data: PVData, day_idx: int, hour: int) -> np.ndarray:
        """获取某天某小时的功率 (返回所有节点)"""
        idx = day_idx * data.n_hours + hour
        if data.n_nodes == 1:
            return np.array([data.power_mw[idx]])
        return data.power_mw[idx, :]

    def normalize_power(self, power: np.ndarray, capacity: float = 0.2) -> np.ndarray:
        """归一化功率到 [0, 1]"""
        return power / capacity

    def get_random_day(self, data: PVData, seed: Optional[int] = None) -> int:
        """随机选择一天"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.randint(0, data.n_days)

    def statistics(self, data: PVData) -> dict:
        """数据统计"""
        if data.n_nodes == 1:
            power_flat = data.power_mw
        else:
            power_flat = data.power_mw.flatten()
        return {
            'n_days': data.n_days,
            'n_samples': len(power_flat),
            'n_nodes': data.n_nodes,
            'power_mean': np.mean(power_flat),
            'power_std': np.std(power_flat),
            'power_max': np.max(power_flat),
            'power_min': np.min(power_flat),
        }

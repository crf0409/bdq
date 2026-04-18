"""
24小时动态负荷模型
参考文件二：按时段系数范围随机生成日负荷模式 + 各节点±3%噪声
不同天/不同节点的负荷各不相同，通过seed保证可重复

时段划分及负荷系数范围（来自问题修改.docx）:
  低谷段 (0:00-6:00):  0.30 ~ 0.50
  早峰段 (7:00-9:00):  0.80 ~ 0.95
  平  段 (10:00-17:00): 0.70 ~ 0.85
  晚峰段 (18:00-22:00): 0.95 ~ 1.10
  回落段 (23:00):       0.60 ~ 0.70
"""

import numpy as np


# 时段定义: (start_hour, end_hour, coeff_min, coeff_max)
TIME_SEGMENTS = [
    (0, 6, 0.30, 0.50),    # 低谷
    (7, 9, 0.80, 0.95),    # 早峰
    (10, 17, 0.70, 0.85),  # 平段
    (18, 22, 0.95, 1.10),  # 晚峰
    (23, 23, 0.60, 0.70),  # 回落
]

# 固定基准系数（向后兼容）
HOURLY_LOAD_FACTORS = np.array([
    0.35, 0.33, 0.32, 0.30, 0.35, 0.40, 0.48,  # 0-6h  低谷
    0.82, 0.88, 0.93,                              # 7-9h  早峰
    0.84, 0.80, 0.76, 0.72, 0.70, 0.73, 0.78, 0.83,  # 10-17h 平段
    0.96, 1.02, 1.06, 1.08, 1.00,                 # 18-22h 晚峰
    0.65,                                           # 23h 回落
])


def _get_segment(hour: int):
    """获取小时所属时段的系数范围"""
    for start, end, cmin, cmax in TIME_SEGMENTS:
        if start <= hour <= end:
            return cmin, cmax
    return 0.50, 0.80  # fallback


def get_load_factor(hour: int) -> float:
    """获取固定基准负荷系数（向后兼容）"""
    return float(HOURLY_LOAD_FACTORS[hour % 24])


def get_daily_load_factors() -> np.ndarray:
    """获取固定24小时负荷系数"""
    return HOURLY_LOAD_FACTORS.copy()


def generate_daily_load_pattern(seed: int = 0) -> np.ndarray:
    """
    按时段系数范围随机生成24小时日负荷模式

    严格保证每个小时的系数在对应时段的 [coeff_min, coeff_max] 范围内。
    同一时段内有平滑过渡，不同天seed不同则曲线不同。

    Args:
        seed: 随机种子（不同天用不同seed）

    Returns:
        (24,) 负荷系数数组，每小时值在对应时段范围内
    """
    rng = np.random.RandomState(seed)
    pattern = np.zeros(24)

    for hour in range(24):
        cmin, cmax = _get_segment(hour)
        pattern[hour] = rng.uniform(cmin, cmax)

    # 平滑（窗口=3）使相邻小时有连续性
    smoothed = np.zeros_like(pattern)
    for i in range(24):
        start = max(0, i - 1)
        end = min(24, i + 2)
        smoothed[i] = np.mean(pattern[start:end])

    # 再次clip确保不超出时段范围
    for hour in range(24):
        cmin, cmax = _get_segment(hour)
        smoothed[hour] = np.clip(smoothed[hour], cmin, cmax)

    return smoothed


def generate_node_load_factors(daily_pattern: np.ndarray, n_nodes: int = 33,
                                seed: int = 0) -> np.ndarray:
    """
    为每个节点生成带±3%噪声的负荷系数
    参考文件二 generate_node_dynamic_loads(): noise = np.random.normal(0, 0.03, 24)

    Args:
        daily_pattern: (24,) 基础日负荷模式
        n_nodes: 节点数
        seed: 随机种子

    Returns:
        (24, n_nodes) 各节点各时刻的负荷系数
    """
    rng = np.random.RandomState(seed)
    node_factors = np.zeros((24, n_nodes))

    for node in range(n_nodes):
        noise = rng.normal(0, 0.03, 24)
        node_factors[:, node] = daily_pattern + noise

    # clip到合理范围（不低于0.20, 不超过1.20）
    node_factors = np.clip(node_factors, 0.20, 1.20)

    return node_factors


def get_load_period(hour: int) -> str:
    """获取时段名称"""
    if 0 <= hour <= 6:
        return "低谷"
    elif 7 <= hour <= 9:
        return "早峰"
    elif 10 <= hour <= 17:
        return "平段"
    elif 18 <= hour <= 22:
        return "晚峰"
    else:
        return "回落"

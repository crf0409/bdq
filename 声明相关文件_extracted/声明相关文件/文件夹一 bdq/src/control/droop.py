"""
下垂控制函数模块
实现无功-电压(Q-V)下垂控制
包含: 基础下垂、带死区下垂、改进下垂控制
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class DroopController:
    """
    无死区下垂控制器

    Q-V下垂控制公式:
    Q = Q0 + Kq * (V_ref - V)
    """
    kq: float = 2.0        # 下垂系数
    v_ref: float = 1.0     # 参考电压
    q_max: float = 0.5     # 最大无功 (pu)
    q_min: float = -0.5    # 最小无功 (pu)

    def calculate_q(self, v: float, q0: float = 0.0) -> float:
        """计算无功输出"""
        delta_v = self.v_ref - v
        q = q0 + self.kq * delta_v
        return np.clip(q, self.q_min, self.q_max)

    def get_characteristic(
        self,
        v_range: Tuple[float, float] = (0.9, 1.1),
        n_points: int = 100,
        q0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取下垂特性曲线"""
        v = np.linspace(v_range[0], v_range[1], n_points)
        q = np.array([self.calculate_q(vi, q0) for vi in v])
        return v, q


@dataclass
class DeadbandDroopController:
    """
    带死区的下垂控制器 (图3-2样式)

    死区内不进行无功调节，避免频繁动作
    Q_max/Q_min 基于逆变器容量和当前P动态计算
    配电网中R>X，Q注入提升电压效果有限，故采用非对称控制:
    - 过电压时: 吸收Q (Q<0)，系数kq
    - 欠电压时: 注入Q (Q>0)，系数kq_inject (较小)
    """
    kq: float = 2.0           # 吸收Q下垂系数 (过电压)
    kq_inject: float = 0.5    # 注入Q下垂系数 (欠电压，较小)
    v_ref: float = 1.0        # 参考电压
    deadband: float = 0.02    # 死区范围 (±0.02 pu)
    q_max: float = 0.5        # 最大无功 (pu), 可动态设置
    q_min: float = -0.5       # 最小无功 (pu), 可动态设置

    def set_dynamic_q_limits(self, s_inv_pu: float, p_current_pu: float):
        """
        基于逆变器容量和当前P动态计算Q限制
        Q_max = sqrt(S_inv² - P²)
        """
        p_abs = abs(p_current_pu)
        if p_abs < s_inv_pu:
            q_limit = np.sqrt(s_inv_pu**2 - p_abs**2)
        else:
            q_limit = 0.0
        self.q_max = q_limit
        self.q_min = -q_limit

    def calculate_q(self, v: float, q0: float = 0.0) -> float:
        """计算无功输出 (带死区, 非对称)"""
        v_low = self.v_ref - self.deadband
        v_high = self.v_ref + self.deadband

        if v_low <= v <= v_high:
            q = q0
        elif v < v_low:
            # 欠电压: 注入少量Q
            q = q0 + self.kq_inject * (v_low - v)
        else:
            # 过电压: 吸收Q (主要控制手段)
            q = q0 + self.kq * (v_high - v)

        return np.clip(q, self.q_min, self.q_max)

    def get_characteristic(
        self,
        v_range: Tuple[float, float] = (0.9, 1.1),
        n_points: int = 200,
        q0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取下垂特性曲线 (图3-2)"""
        v = np.linspace(v_range[0], v_range[1], n_points)
        q = np.array([self.calculate_q(vi, q0) for vi in v])
        return v, q


@dataclass
class ImprovedDroopController:
    """
    改进下垂控制器 (图3-3样式)

    特点:
    - 可调电压截距 v_setpoint (由DRL或优化器调节)
    - 死区范围 (比传统更小)
    - 非对称响应: 过电压时kq大, 欠电压时kq_inject小
    - 可行域边界约束
    """
    kq: float = 4.0            # 吸收Q下垂系数 (过电压)
    kq_inject: float = 0.5     # 注入Q下垂系数 (欠电压)
    v_ref: float = 1.0         # 参考电压
    v_setpoint: float = 1.0    # 可调电压截距
    deadband: float = 0.01     # 死区范围 (比传统更小)
    q_max: float = 0.5         # 最大无功 (pu)
    q_min: float = -0.5        # 最小无功 (pu)
    v_feasible_min: float = 0.95  # 可行域下边界
    v_feasible_max: float = 1.05  # 可行域上边界

    def set_dynamic_q_limits(self, s_inv_pu: float, p_current_pu: float):
        """基于逆变器容量和当前P动态计算Q限制"""
        p_abs = abs(p_current_pu)
        if p_abs < s_inv_pu:
            q_limit = np.sqrt(s_inv_pu**2 - p_abs**2)
        else:
            q_limit = 0.0
        self.q_max = q_limit
        self.q_min = -q_limit

    def calculate_q(self, v: float, q0: float = 0.0) -> float:
        """
        计算改进下垂无功输出

        以v_setpoint为中心的下垂特性，带死区，非对称响应
        """
        if abs(v - self.v_ref) <= self.deadband:
            q = q0
        elif v > self.v_ref + self.deadband:
            # 过电压: 以v_setpoint为基准吸收Q
            delta_v = self.v_setpoint - v
            q = q0 + self.kq * delta_v
        else:
            # 欠电压: 小幅注入Q
            delta_v = self.v_setpoint - v
            q = q0 + self.kq_inject * delta_v

        return np.clip(q, self.q_min, self.q_max)

    def get_characteristic(
        self,
        v_range: Tuple[float, float] = (0.9, 1.1),
        n_points: int = 200,
        q0: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取改进下垂特性曲线 (图3-3)"""
        v = np.linspace(v_range[0], v_range[1], n_points)
        q = np.array([self.calculate_q(vi, q0) for vi in v])
        return v, q

    def get_feasible_region(self, n_points: int = 200) -> dict:
        """获取可行域边界数据 (用于绘制虚线)"""
        v = np.linspace(self.v_feasible_min, self.v_feasible_max, n_points)
        return {
            'v': v,
            'q_upper': np.full(n_points, self.q_max),
            'q_lower': np.full(n_points, self.q_min),
            'v_min_line': self.v_feasible_min,
            'v_max_line': self.v_feasible_max,
        }


def improved_droop_control(
    v: float,
    v_ref: float,
    v_setpoint: float,
    kq: float,
    deadband: float = 0.02,
    q_max: float = 0.5
) -> float:
    """
    改进的无功电压下垂控制 (函数式接口)

    Args:
        v: 当前电压
        v_ref: 参考电压 (通常为1.0)
        v_setpoint: 电压截距设定值 (可调参数)
        kq: 下垂系数
        deadband: 死区范围
        q_max: 最大无功

    Returns:
        无功输出
    """
    delta_v = v_setpoint - v

    if abs(v - v_ref) <= deadband:
        q = 0.0
    else:
        q = kq * delta_v

    return np.clip(q, -q_max, q_max)

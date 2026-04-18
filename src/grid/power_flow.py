"""
牛顿-拉夫森法潮流计算引擎
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .ieee33 import IEEE33Bus


@dataclass
class PowerFlowResult:
    """潮流计算结果"""
    converged: bool
    iterations: int
    voltage: np.ndarray          # 复数电压
    voltage_magnitude: np.ndarray  # 电压幅值
    voltage_angle: np.ndarray    # 电压相角 (rad)
    p_slack: float               # 平衡节点有功
    q_slack: float               # 平衡节点无功
    p_loss: float                # 有功损耗
    q_loss: float                # 无功损耗
    max_mismatch: float          # 最大功率偏差

    @property
    def v_min(self) -> float:
        return float(np.min(self.voltage_magnitude))

    @property
    def v_max(self) -> float:
        return float(np.max(self.voltage_magnitude))

    @property
    def v_mean(self) -> float:
        return float(np.mean(self.voltage_magnitude))

    def voltage_deviation(self, v_ref: float = 1.0) -> float:
        """计算平均电压偏差"""
        return float(np.mean(np.abs(self.voltage_magnitude - v_ref)))

    def voltage_violation_rate(self, v_min: float = 0.95, v_max: float = 1.05) -> float:
        """计算电压越限率"""
        violations = np.sum((self.voltage_magnitude < v_min) |
                           (self.voltage_magnitude > v_max))
        return violations / len(self.voltage_magnitude)


class PowerFlowSolver:
    """牛顿-拉夫森法潮流计算求解器"""

    def __init__(
        self,
        grid: IEEE33Bus,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ):
        """
        初始化求解器

        Args:
            grid: IEEE33节点模型
            tolerance: 收敛精度
            max_iterations: 最大迭代次数
        """
        self.grid = grid
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self) -> PowerFlowResult:
        """执行潮流计算"""
        n = self.grid.n_buses
        G = self.grid.G
        B = self.grid.B
        slack_idx = self.grid.balance_node - 1

        # 获取注入功率
        P_spec, Q_spec = self.grid.get_all_injections()

        # 初始化电压（直角坐标）
        e = np.ones(n) * self.grid.balance_voltage
        f = np.zeros(n)

        # 非平衡节点索引
        non_slack = self.grid.get_non_slack_indices()
        n_pq = len(non_slack)

        converged = False
        iteration = 0
        max_mismatch = float('inf')

        while iteration < self.max_iterations:
            # 计算节点功率
            P_calc = np.zeros(n)
            Q_calc = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    P_calc[i] += e[i] * (G[i, j] * e[j] - B[i, j] * f[j]) + \
                                 f[i] * (G[i, j] * f[j] + B[i, j] * e[j])
                    Q_calc[i] += f[i] * (G[i, j] * e[j] - B[i, j] * f[j]) - \
                                 e[i] * (G[i, j] * f[j] + B[i, j] * e[j])

            # 计算功率偏差
            dP = np.zeros(n_pq)
            dQ = np.zeros(n_pq)
            for idx, i in enumerate(non_slack):
                dP[idx] = P_spec[i] - P_calc[i]
                dQ[idx] = Q_spec[i] - Q_calc[i]

            max_mismatch = max(np.max(np.abs(dP)), np.max(np.abs(dQ)))

            if max_mismatch < self.tolerance:
                converged = True
                break

            # 构建雅可比矩阵
            jacobi = self._build_jacobian(e, f, G, B, non_slack)

            # 构建偏差向量
            dW = np.zeros(2 * n_pq)
            for idx in range(n_pq):
                dW[2 * idx] = dP[idx]
                dW[2 * idx + 1] = dQ[idx]

            # 求解线性方程组
            try:
                dV = np.linalg.solve(jacobi, -dW)
            except np.linalg.LinAlgError:
                break

            # 更新电压
            for idx, i in enumerate(non_slack):
                e[i] += dV[2 * idx]
                f[i] += dV[2 * idx + 1]

            iteration += 1

        # 计算结果
        V_complex = e + 1j * f
        V_mag = np.abs(V_complex)
        V_angle = np.angle(V_complex)

        # 平衡节点功率
        p_slack = P_calc[slack_idx]
        q_slack = Q_calc[slack_idx]

        # 计算网损
        total_loss = self.grid.calculate_total_loss(V_complex)

        return PowerFlowResult(
            converged=converged,
            iterations=iteration,
            voltage=V_complex,
            voltage_magnitude=V_mag,
            voltage_angle=V_angle,
            p_slack=p_slack,
            q_slack=q_slack,
            p_loss=float(np.real(total_loss)),
            q_loss=float(np.imag(total_loss)),
            max_mismatch=max_mismatch
        )

    def _build_jacobian(
        self,
        e: np.ndarray,
        f: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        non_slack: list
    ) -> np.ndarray:
        """构建雅可比矩阵"""
        n = len(e)
        n_pq = len(non_slack)
        jacobi = np.zeros((2 * n_pq, 2 * n_pq))

        # 预计算
        A = np.zeros(n)
        S = np.zeros(n)
        for i in range(n):
            for j in range(n):
                A[i] += G[i, j] * e[j] - B[i, j] * f[j]
                S[i] += G[i, j] * f[j] + B[i, j] * e[j]

        for idx_i, i in enumerate(non_slack):
            for idx_j, j in enumerate(non_slack):
                if i != j:
                    # 非对角元素
                    jacobi[2*idx_i, 2*idx_j] = -G[i, j] * e[i] - B[i, j] * f[i]
                    jacobi[2*idx_i, 2*idx_j+1] = B[i, j] * e[i] - G[i, j] * f[i]
                    jacobi[2*idx_i+1, 2*idx_j] = B[i, j] * e[i] - G[i, j] * f[i]
                    jacobi[2*idx_i+1, 2*idx_j+1] = G[i, j] * e[i] + B[i, j] * f[i]
                else:
                    # 对角元素
                    jacobi[2*idx_i, 2*idx_i] = -A[i] - G[i, i] * e[i] - B[i, i] * f[i]
                    jacobi[2*idx_i, 2*idx_i+1] = -S[i] + B[i, i] * e[i] - G[i, i] * f[i]
                    jacobi[2*idx_i+1, 2*idx_i] = S[i] + B[i, i] * e[i] - G[i, i] * f[i]
                    jacobi[2*idx_i+1, 2*idx_i+1] = -A[i] + G[i, i] * e[i] + B[i, i] * f[i]

        return jacobi


def run_power_flow(
    pv_buses: list = None,
    pv_power_mw: float = 0.5,
    balance_node: int = 1,
    balance_voltage: float = 1.0,
    base_mva: float = 10.0
) -> PowerFlowResult:
    """
    便捷函数：运行潮流计算

    Args:
        pv_buses: 光伏节点列表
        pv_power_mw: 光伏有功出力 MW
        balance_node: 平衡节点
        balance_voltage: 平衡节点电压
        base_mva: 基准容量

    Returns:
        PowerFlowResult: 潮流计算结果
    """
    if pv_buses is None:
        pv_buses = [10, 18, 22, 24, 28, 33]

    grid = IEEE33Bus(
        base_mva=base_mva,
        balance_node=balance_node,
        balance_voltage=balance_voltage,
        pv_buses=pv_buses,
        pv_capacity_mw=1.0
    )

    # 设置光伏出力
    for bus_id in pv_buses:
        grid.set_pv_output(bus_id, pv_power_mw, pv_power_mw * 0.2)

    solver = PowerFlowSolver(grid)
    return solver.solve()

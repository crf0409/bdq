"""
IEEE 33节点微电网模型
支持孤岛运行模式，包含光伏接入、柴油发电机和储能
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class BranchData:
    """支路数据"""
    from_bus: int
    to_bus: int
    resistance: float  # 电阻 (pu)
    reactance: float   # 电抗 (pu)

    @property
    def impedance(self) -> complex:
        return self.resistance + 1j * self.reactance

    @property
    def admittance(self) -> complex:
        return 1.0 / self.impedance


@dataclass
class BusData:
    """节点数据"""
    bus_id: int
    bus_type: int  # 1=PQ, 2=PV, 3=Slack
    p_load: float  # 有功负荷 (pu)
    q_load: float  # 无功负荷 (pu)
    p_load_base: float = 0.0  # 基础有功负荷 (pu)，用于动态负荷
    q_load_base: float = 0.0  # 基础无功负荷 (pu)，用于动态负荷
    v_init: float = 1.0  # 初始电压
    theta_init: float = 0.0  # 初始相角


@dataclass
class PVUnit:
    """光伏发电单元"""
    bus_id: int
    capacity_mw: float  # 装机容量 MW
    s_inv: float = 0.0  # 逆变器额定容量 MVA
    p_output: float = 0.0  # 当前有功出力 (pu)
    q_output: float = 0.0  # 当前无功出力 (pu)
    q_max: float = 0.5  # 最大无功 (pu)
    q_min: float = -0.5  # 最小无功 (pu)
    power_factor: float = 0.85  # 额定功率因数

    def update_q_limits(self, base_mva: float):
        """
        根据当前有功出力动态计算无功限制
        Q_max = sqrt(S_inv² - P²), 其中P为当前有功出力(pu)
        """
        s_inv_pu = self.s_inv / base_mva
        p_pu = abs(self.p_output)
        if p_pu < s_inv_pu:
            q_limit = np.sqrt(s_inv_pu**2 - p_pu**2)
        else:
            q_limit = 0.0
        self.q_max = q_limit
        self.q_min = -q_limit


@dataclass
class DieselGenerator:
    """柴油发电机 (接Bus1平衡节点)"""
    bus_id: int = 1
    capacity_mw: float = 1.8  # 额定容量 MW
    r_pu: float = 0.01  # 电阻 (pu)
    x_pu: float = 0.1   # 电抗 (pu)
    p_output: float = 0.0  # 当前有功出力 (pu)
    q_output: float = 0.0  # 当前无功出力 (pu)


@dataclass
class EnergyStorage:
    """储能系统 (接Bus1平衡节点)"""
    bus_id: int = 1
    capacity_mw: float = 1.8   # 额定功率 MW
    energy_mwh: float = 7.2    # 额定能量 MWh
    soc_min: float = 0.20      # SOC下限
    soc_max: float = 0.90      # SOC上限
    soc_init: float = 0.60     # 初始SOC
    efficiency: float = 0.94   # 充放电效率
    soc: float = 0.60          # 当前SOC
    p_output: float = 0.0      # 当前出力 (pu), 正为放电，负为充电

    def reset(self):
        """重置SOC到初始值"""
        self.soc = self.soc_init
        self.p_output = 0.0

    def update_soc(self, p_mw: float, dt_hours: float = 1.0):
        """
        更新SOC
        Args:
            p_mw: 功率 (MW), 正为放电，负为充电
            dt_hours: 时间步长 (小时)
        """
        if p_mw > 0:
            # 放电
            energy_change = p_mw * dt_hours / self.efficiency
        else:
            # 充电
            energy_change = p_mw * dt_hours * self.efficiency

        new_soc = self.soc - energy_change / self.energy_mwh
        self.soc = np.clip(new_soc, self.soc_min, self.soc_max)

    def get_charge_power(self, pv_total_mw: float, load_total_mw: float, hour: int) -> float:
        """
        储能充放电逻辑：白天光伏出力时充电，夜间放电
        Args:
            pv_total_mw: 总PV出力 MW
            load_total_mw: 总负荷 MW
            hour: 当前小时
        Returns:
            功率 MW (正为放电，负为充电)
        """
        if 6 <= hour <= 18 and pv_total_mw > 0.05:
            # 白天有光伏时充电
            charge_power = min(pv_total_mw * 0.3, self.capacity_mw)
            # 检查SOC上限
            if self.soc >= self.soc_max:
                return 0.0
            return -charge_power
        elif hour >= 18 or hour < 6:
            # 夜间放电
            discharge_power = min(load_total_mw * 0.2, self.capacity_mw)
            # 检查SOC下限
            if self.soc <= self.soc_min:
                return 0.0
            return discharge_power
        else:
            return 0.0


class IEEE33Bus:
    """IEEE 33节点微电网模型"""

    # IEEE33标准支路数据 (from, to, R_ohm, X_ohm) - Baran & Wu 1989
    # 仅包含32条辐射状支路 (联络线默认断开)
    BRANCH_DATA = [
        (1, 2, 0.0922, 0.0470), (2, 3, 0.4930, 0.2511), (3, 4, 0.3660, 0.1864),
        (4, 5, 0.3811, 0.1941), (5, 6, 0.8190, 0.7070), (6, 7, 0.1872, 0.6188),
        (7, 8, 0.7114, 0.2351), (8, 9, 1.0300, 0.7400), (9, 10, 1.0440, 0.7400),
        (10, 11, 0.1966, 0.0650), (11, 12, 0.3744, 0.1238), (12, 13, 1.4680, 1.1550),
        (13, 14, 0.5416, 0.7129), (14, 15, 0.5910, 0.5260), (15, 16, 0.7463, 0.5450),
        (16, 17, 1.2890, 1.7210), (17, 18, 0.7320, 0.5740), (2, 19, 0.1640, 0.1565),
        (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784), (21, 22, 0.7089, 0.9373),
        (3, 23, 0.4512, 0.3083), (23, 24, 0.8980, 0.7091), (24, 25, 0.8960, 0.7011),
        (6, 26, 0.2030, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.0590, 0.9337),
        (28, 29, 0.8042, 0.7006), (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.9630),
        (31, 32, 0.3105, 0.3619), (32, 33, 0.3410, 0.5302),
    ]

    # IEEE33标准节点负荷数据 (bus_id, P_load, Q_load) in MW/MVar
    LOAD_DATA = [
        (1, 0.0, 0.0), (2, 0.10, 0.06), (3, 0.09, 0.04), (4, 0.12, 0.08),
        (5, 0.06, 0.03), (6, 0.06, 0.02), (7, 0.20, 0.10), (8, 0.20, 0.10),
        (9, 0.06, 0.02), (10, 0.06, 0.02), (11, 0.045, 0.03), (12, 0.06, 0.035),
        (13, 0.06, 0.035), (14, 0.12, 0.08), (15, 0.06, 0.01), (16, 0.06, 0.02),
        (17, 0.06, 0.02), (18, 0.09, 0.04), (19, 0.09, 0.04), (20, 0.09, 0.04),
        (21, 0.09, 0.04), (22, 0.09, 0.04), (23, 0.09, 0.05), (24, 0.42, 0.20),
        (25, 0.42, 0.20), (26, 0.06, 0.025), (27, 0.06, 0.025), (28, 0.06, 0.02),
        (29, 0.12, 0.07), (30, 0.20, 0.60), (31, 0.15, 0.07), (32, 0.21, 0.10),
        (33, 0.06, 0.04)
    ]

    def __init__(
        self,
        base_mva: float = 10.0,
        balance_node: int = 1,
        balance_voltage: float = 1.0,
        pv_buses: Optional[List[int]] = None,
        pv_capacity_mw: float = 1.0,
        load_scale: float = 1.0
    ):
        """
        初始化IEEE33节点模型

        Args:
            base_mva: 基准容量 MVA (论文中10MVA)
            balance_node: 平衡节点编号 (1-33)
            balance_voltage: 平衡节点电压 (pu)
            pv_buses: 光伏接入节点列表
            pv_capacity_mw: 每个光伏装机容量 MW
            load_scale: 负荷缩放系数
        """
        self.n_buses = 33
        self.n_branches = 32
        self.base_mva = base_mva
        self.base_kv = 12.66
        self.balance_node = balance_node
        self.balance_voltage = balance_voltage
        self.load_scale = load_scale

        # 阻抗基准: Z_base = V_base² / S_base (ohm → pu)
        z_base = self.base_kv ** 2 / self.base_mva

        # 初始化支路 (将ohm值转换为标幺值)
        self.branches: List[BranchData] = []
        for from_bus, to_bus, r_ohm, x_ohm in self.BRANCH_DATA:
            self.branches.append(BranchData(
                from_bus, to_bus, r_ohm / z_base, x_ohm / z_base
            ))

        # 初始化节点 (负荷 × load_scale)
        self.buses: Dict[int, BusData] = {}
        for bus_id, p, q in self.LOAD_DATA:
            bus_type = 3 if bus_id == balance_node else 1
            p_scaled = p * load_scale
            q_scaled = q * load_scale
            p_pu = -p_scaled / base_mva  # 负荷为负
            q_pu = -q_scaled / base_mva
            v_init = balance_voltage if bus_id == balance_node else 1.0
            self.buses[bus_id] = BusData(
                bus_id, bus_type, p_pu, q_pu, p_pu, q_pu, v_init
            )

        # 初始化光伏 (容配比1.15, 额定0.2MVA, 功率因数0.85)
        self.pv_units: Dict[int, PVUnit] = {}
        if pv_buses:
            s_inv = pv_capacity_mw * 1.15  # 容配比1.15
            for bus_id in pv_buses:
                pv = PVUnit(
                    bus_id=bus_id,
                    capacity_mw=pv_capacity_mw,
                    s_inv=s_inv,
                    power_factor=0.85,
                )
                # 初始动态Q限制
                pv.update_q_limits(base_mva)
                self.pv_units[bus_id] = pv

        # 柴油发电机 (Bus1)
        self.diesel = DieselGenerator(
            bus_id=1,
            capacity_mw=1.8,
            r_pu=0.01,
            x_pu=0.1
        )

        # 储能系统 (Bus1)
        self.storage = EnergyStorage(
            bus_id=1,
            capacity_mw=1.8,
            energy_mwh=7.2,
            soc_min=0.20,
            soc_max=0.90,
            soc_init=0.60,
            efficiency=0.94,
            soc=0.60
        )

        # 构建导纳矩阵
        self._build_admittance_matrix()

    def _build_admittance_matrix(self) -> None:
        """构建节点导纳矩阵"""
        self.Y = np.zeros((self.n_buses, self.n_buses), dtype=complex)

        for branch in self.branches:
            i = branch.from_bus - 1
            j = branch.to_bus - 1
            y = branch.admittance

            self.Y[i, i] += y
            self.Y[j, j] += y
            self.Y[i, j] -= y
            self.Y[j, i] -= y

        self.G = np.real(self.Y)
        self.B = np.imag(self.Y)

    def set_pv_output(self, bus_id: int, p_mw: float, q_mvar: float = 0.0) -> None:
        """设置光伏出力并更新动态Q限制"""
        if bus_id in self.pv_units:
            self.pv_units[bus_id].p_output = p_mw / self.base_mva
            self.pv_units[bus_id].q_output = q_mvar / self.base_mva
            # 更新动态Q限制
            self.pv_units[bus_id].update_q_limits(self.base_mva)

    def set_all_pv_output(self, p_mw: float, q_mvar: float = 0.0) -> None:
        """设置所有光伏相同出力"""
        for bus_id in self.pv_units:
            self.set_pv_output(bus_id, p_mw, q_mvar)

    def apply_load_profile(self, load_factor=None, node_factors: np.ndarray = None) -> None:
        """
        应用动态负荷系数
        Args:
            load_factor: 全局负荷系数 (float) 或 per-node系数 (ndarray of shape (33,))
            node_factors: 可选的per-node系数 (33,)，与load_factor叠加使用
        """
        for bus_id, bus in self.buses.items():
            if bus_id != self.balance_node:
                idx = bus_id - 1
                if node_factors is not None:
                    factor = node_factors[idx]
                elif isinstance(load_factor, np.ndarray):
                    factor = load_factor[idx]
                else:
                    factor = load_factor
                bus.p_load = bus.p_load_base * factor
                bus.q_load = bus.q_load_base * factor

    def get_bus_injection(self, bus_id: int) -> Tuple[float, float]:
        """获取节点净注入功率 (P, Q)"""
        bus = self.buses[bus_id]
        p = bus.p_load
        q = bus.q_load

        if bus_id in self.pv_units:
            pv = self.pv_units[bus_id]
            p += pv.p_output
            q += pv.q_output

        return p, q

    def get_all_injections(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有节点注入功率"""
        P = np.zeros(self.n_buses)
        Q = np.zeros(self.n_buses)

        for i in range(self.n_buses):
            bus_id = i + 1
            if bus_id != self.balance_node:
                P[i], Q[i] = self.get_bus_injection(bus_id)

        return P, Q

    def calculate_branch_power(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算支路功率"""
        S_from = np.zeros(self.n_branches, dtype=complex)
        S_to = np.zeros(self.n_branches, dtype=complex)

        for idx, branch in enumerate(self.branches):
            i = branch.from_bus - 1
            j = branch.to_bus - 1

            I_ij = (V[i] - V[j]) * branch.admittance
            I_ji = -I_ij

            S_from[idx] = V[i] * np.conj(I_ij)
            S_to[idx] = V[j] * np.conj(I_ji)

        return S_from, S_to

    def calculate_total_loss(self, V: np.ndarray) -> complex:
        """计算系统总损耗"""
        S_from, S_to = self.calculate_branch_power(V)
        return np.sum(S_from + S_to)

    def get_pv_buses(self) -> List[int]:
        """获取光伏节点列表"""
        return list(self.pv_units.keys())

    def get_non_slack_indices(self) -> List[int]:
        """获取非平衡节点索引 (0-based)"""
        return [i for i in range(self.n_buses) if i + 1 != self.balance_node]

    def get_total_load_mw(self) -> float:
        """获取总负荷有功 (MW)"""
        total = 0.0
        for bus_id, bus in self.buses.items():
            if bus_id != self.balance_node:
                total += abs(bus.p_load) * self.base_mva
        return total

    def get_total_pv_mw(self) -> float:
        """获取总PV出力 (MW)"""
        total = 0.0
        for pv in self.pv_units.values():
            total += pv.p_output * self.base_mva
        return total

    def __repr__(self) -> str:
        return (f"IEEE33Bus(balance_node={self.balance_node}, "
                f"pv_buses={self.get_pv_buses()}, "
                f"base_mva={self.base_mva}, "
                f"load_scale={self.load_scale})")

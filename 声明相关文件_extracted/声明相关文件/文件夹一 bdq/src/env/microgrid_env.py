"""
微电网强化学习环境
基于Gymnasium实现
支持动态负荷、多节点PV数据、储能SOC追踪
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.grid import IEEE33Bus, PowerFlowSolver
from src.grid.dynamic_load import get_load_factor
from src.utils.data_loader import PVDataLoader, PVData


class MicrogridEnv(gym.Env):
    """
    微电网电压控制环境

    状态空间 (41维):
    - 各节点电压 (33维)
    - 各光伏当前出力 (6维)
    - 当前时刻 (1维, 归一化)
    - 储能SOC (1维)

    动作空间:
    - "q_only": 6维 (各PV无功输出)
    - "q_and_v": 12维 (6Q + 6V_setpoint)

    奖励函数:
    - 电压偏差惩罚
    - 电压越限惩罚
    - 网络损耗惩罚
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        pv_data: PVData,
        pv_buses: List[int] = None,
        pv_capacity: float = 0.6,
        base_mva: float = 10.0,
        load_scale: float = 1.0,
        v_min: float = 0.95,
        v_max: float = 1.05,
        v_ref: float = 1.0,
        reward_weights: Dict[str, float] = None,
        max_steps: int = 24,
        action_mode: str = "q_and_v",
    ):
        """
        初始化环境

        Args:
            pv_data: 光伏数据 (支持多列6节点)
            pv_buses: 光伏接入节点
            pv_capacity: 光伏容量 (MW)
            base_mva: 基准容量 (MVA)
            load_scale: 负荷缩放系数
            v_min: 电压下限
            v_max: 电压上限
            v_ref: 参考电压
            reward_weights: 奖励权重
            max_steps: 最大步数 (一天24小时)
            action_mode: 动作模式 "q_only" or "q_and_v"
        """
        super().__init__()

        self.pv_data = pv_data
        self.pv_buses = pv_buses or [10, 18, 22, 24, 28, 33]
        self.n_pv = len(self.pv_buses)
        self.pv_capacity = pv_capacity
        self.base_mva = base_mva
        self.load_scale = load_scale
        self.v_min = v_min
        self.v_max = v_max
        self.v_ref = v_ref
        self.max_steps = max_steps
        self.action_mode = action_mode

        # 奖励权重
        self.reward_weights = reward_weights or {
            'voltage_deviation': 100.0,
            'voltage_violation': 200.0,
            'power_loss': 10.0,
        }

        # 创建电网模型
        self.grid = IEEE33Bus(
            base_mva=base_mva,
            balance_node=1,
            balance_voltage=1.0,
            pv_buses=self.pv_buses,
            pv_capacity_mw=pv_capacity,
            load_scale=load_scale,
        )
        self.solver = PowerFlowSolver(self.grid, tolerance=1e-6)

        # 定义动作空间
        if action_mode == "q_only":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_pv,), dtype=np.float32
            )
        else:
            # q_and_v: 6(Q) + 6(V_setpoint) = 12维
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.n_pv * 2,), dtype=np.float32
            )

        # 定义状态空间: 33(电压) + 6(PV) + 1(时间) + 1(SOC) = 41维
        n_state = 33 + self.n_pv + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_state,), dtype=np.float32
        )

        # 内部状态
        self.current_step = 0
        self.current_day = 0
        self.voltage = np.ones(33)
        self.pv_power = np.zeros(self.n_pv)

        # 记录历史
        self.history = {
            'voltage': [],
            'pv_power': [],
            'q_action': [],
            'reward': [],
            'loss': [],
            'soc': [],
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 随机选择一天
        if options and 'day' in options:
            self.current_day = options['day']
        else:
            self.current_day = self.np_random.integers(0, self.pv_data.n_days)

        self.current_step = 0
        self.history = {k: [] for k in self.history}

        # 重置储能SOC
        self.grid.storage.reset()

        # 应用动态负荷
        load_factor = get_load_factor(self.current_step)
        self.grid.apply_load_profile(load_factor)

        # 获取初始光伏出力
        self._update_pv_power()

        # 运行初始潮流计算
        self._run_power_flow(np.zeros(self.n_pv))

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 解析动作
        action = np.clip(action, -1.0, 1.0)

        if self.action_mode == "q_only":
            # Q缩放: 基于逆变器动态Q限制
            q_action = np.zeros(self.n_pv)
            for i, bus_id in enumerate(self.pv_buses):
                pv = self.grid.pv_units[bus_id]
                q_limit = pv.q_max * self.base_mva  # pu -> MW
                q_action[i] = action[i] * q_limit
            v_setpoint = np.ones(self.n_pv) * 1.0
        else:
            # Q + V_setpoint
            q_action = np.zeros(self.n_pv)
            for i, bus_id in enumerate(self.pv_buses):
                pv = self.grid.pv_units[bus_id]
                q_limit = pv.q_max * self.base_mva
                q_action[i] = action[i] * q_limit
            v_setpoint = 1.0 + action[self.n_pv:] * 0.05  # V ∈ [0.95, 1.05]

        # 应用动态负荷
        hour = self.current_step % 24
        load_factor = get_load_factor(hour)
        self.grid.apply_load_profile(load_factor)

        # 运行潮流计算
        result = self._run_power_flow(q_action)

        # 计算奖励
        reward = self._compute_reward(result)

        # 更新储能SOC
        pv_total = self.grid.get_total_pv_mw()
        load_total = self.grid.get_total_load_mw()
        storage_power = self.grid.storage.get_charge_power(pv_total, load_total, hour)
        self.grid.storage.update_soc(storage_power)

        # 记录历史
        self.history['voltage'].append(self.voltage.copy())
        self.history['pv_power'].append(self.pv_power.copy())
        self.history['q_action'].append(q_action.copy())
        self.history['reward'].append(reward)
        self.history['loss'].append(result.p_loss if result.converged else 0)
        self.history['soc'].append(self.grid.storage.soc)

        # 更新步数
        self.current_step += 1
        self._update_pv_power()

        # 检查终止条件
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info['power_flow_result'] = result

        return obs, reward, terminated, truncated, info

    def _update_pv_power(self):
        """更新当前光伏出力 (支持多列PV数据)"""
        hour = self.current_step % 24
        day_start = self.current_day * 24
        idx = day_start + hour

        if self.pv_data.n_nodes >= self.n_pv:
            # 多列数据: 各节点使用各自的PV出力
            if idx < len(self.pv_data.power_mw):
                for i in range(self.n_pv):
                    self.pv_power[i] = self.pv_data.power_mw[idx, i]
            else:
                self.pv_power = np.zeros(self.n_pv)
        elif self.pv_data.n_nodes == 1:
            # 单列数据: 所有PV使用相同出力
            if idx < len(self.pv_data.power_mw):
                power = self.pv_data.power_mw[idx]
                self.pv_power = np.full(self.n_pv, power)
            else:
                self.pv_power = np.zeros(self.n_pv)

    def _run_power_flow(self, q_action: np.ndarray):
        """运行潮流计算"""
        for i, bus_id in enumerate(self.pv_buses):
            p_mw = self.pv_power[i]
            q_mvar = q_action[i]
            self.grid.set_pv_output(bus_id, p_mw, q_mvar)

        result = self.solver.solve()

        if result.converged:
            self.voltage = result.voltage_magnitude

        return result

    def _compute_reward(self, result) -> float:
        """计算奖励"""
        if not result.converged:
            return -1000.0

        reward = 0.0

        # 1. 电压偏差惩罚
        voltage_dev = np.mean(np.abs(self.voltage - self.v_ref))
        reward -= self.reward_weights['voltage_deviation'] * voltage_dev

        # 2. 电压越限惩罚
        violations = np.sum((self.voltage < self.v_min) | (self.voltage > self.v_max))
        reward -= self.reward_weights['voltage_violation'] * violations

        # 3. 网络损耗惩罚
        reward -= self.reward_weights['power_loss'] * result.p_loss * 100

        return reward

    def _get_obs(self) -> np.ndarray:
        """获取观测 (41维)"""
        time_norm = self.current_step / 24.0
        soc = self.grid.storage.soc

        obs = np.concatenate([
            self.voltage,                          # 33维
            self.pv_power / self.pv_capacity,      # 6维 (归一化)
            [time_norm],                           # 1维
            [soc],                                 # 1维
        ])

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """获取信息"""
        return {
            'day': self.current_day,
            'step': self.current_step,
            'v_min': np.min(self.voltage),
            'v_max': np.max(self.voltage),
            'v_mean': np.mean(self.voltage),
            'pv_power': self.pv_power.copy(),
            'soc': self.grid.storage.soc,
        }

    def get_baseline_result(self) -> Dict:
        """获取无控制基线结果 (Q=0)"""
        self.current_step = 0
        self.grid.storage.reset()
        self._update_pv_power()

        baseline_voltages = []
        baseline_losses = []
        baseline_socs = []

        for step in range(self.max_steps):
            self.current_step = step
            self._update_pv_power()

            # 动态负荷
            load_factor = get_load_factor(step)
            self.grid.apply_load_profile(load_factor)

            # 无功输出为0
            result = self._run_power_flow(np.zeros(self.n_pv))
            baseline_voltages.append(self.voltage.copy())
            baseline_losses.append(result.p_loss if result.converged else 0)

            # 更新储能
            pv_total = self.grid.get_total_pv_mw()
            load_total = self.grid.get_total_load_mw()
            storage_power = self.grid.storage.get_charge_power(pv_total, load_total, step)
            self.grid.storage.update_soc(storage_power)
            baseline_socs.append(self.grid.storage.soc)

        return {
            'voltages': np.array(baseline_voltages),
            'losses': np.array(baseline_losses),
            'socs': np.array(baseline_socs),
        }

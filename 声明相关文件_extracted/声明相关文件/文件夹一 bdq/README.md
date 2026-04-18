# IEEE33 微电网深度强化学习电压控制系统

基于 Soft Actor-Critic (SAC) 算法的 IEEE33 节点微电网电压控制项目。

## 项目简介

本项目实现了基于深度强化学习的微电网电压控制系统，主要功能包括：

- **潮流计算**：牛顿-拉夫森法求解 IEEE33 节点微电网
- **光伏建模**：多节点光伏接入与动态无功控制
- **强化学习**：SAC 算法实现智能电压控制
- **可视化分析**：全面的结果可视化与对比分析

## 项目结构

```
.
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── agents/              # RL 智能体
│   │   └── sac_agent.py     # SAC 智能体封装
│   ├── control/             # 控制策略
│   │   └── droop.py         # 下垂控制
│   ├── env/                 # 强化学习环境
│   │   └── microgrid_env.py # 微电网环境
│   ├── grid/                # 电网模型
│   │   ├── ieee33.py        # IEEE33 节点模型
│   │   ├── power_flow.py    # 潮流计算
│   │   └── dynamic_load.py  # 动态负荷
│   └── utils/               # 工具函数
│       ├── data_loader.py   # 数据加载
│       └── visualization.py # 可视化
├── scripts/                 # 运行脚本
│   ├── setup_environment.py # 环境初始化
│   ├── train_sac.py         # 训练脚本
│   ├── evaluate_sac.py      # 评估脚本
│   ├── task2_pv_impact.py   # 光伏影响分析
│   └── task3_droop_control.py # 下垂控制
├── 训练集300天_6节点.xlsx   # 训练数据
├── 测试集20天_6节点.xlsx    # 测试数据
├── requirements.txt         # Python 依赖
└── setup.py                 # 安装配置
```

## 环境要求

- Python >= 3.9
- CUDA (可选，用于GPU加速训练)

## 快速开始

### 1. 安装依赖

```bash
# 方式1：使用 pip
pip install -r requirements.txt

# 方式2：使用 setup.py
pip install -e .

# 方式3：运行环境初始化脚本（推荐）
python scripts/setup_environment.py
```

### 2. 运行脚本

```bash
# 任务2：光伏影响分析
python scripts/task2_pv_impact.py

# 任务3：下垂控制仿真
python scripts/task3_droop_control.py

# 任务4：训练 SAC 智能体
python scripts/train_sac.py

# 评估模型
python scripts/evaluate_sac.py
```

## 配置说明

编辑 `config/config.yaml` 修改系统参数：

```yaml
# 电网参数
grid:
  n_buses: 33          # 节点数
  base_mva: 10.0       # 基准容量
  base_kv: 12.66       # 基准电压

# 光伏配置
pv:
  buses: [10, 18, 22, 24, 28, 33]  # 接入节点
  capacity_mw: 0.6     # 单机容量

# SAC 训练参数
sac:
  learning_rate: 3e-4
  buffer_size: 100000
  batch_size: 256
  gamma: 0.99
```

## 输出结果

所有结果保存在 `results/` 目录：

- `task2_pv_impact.png` - 光伏影响分析图
- `task4_training_curve.png` - 训练曲线
- `fig3_6_voltage_distribution.png` - 电压分布图
- `fig3_7_loss_distribution.png` - 网损分布图
- `evaluation_stats.json` - 评估统计数据

## 常见问题

### Q: matplotlib 中文显示乱码？

A: 运行环境初始化脚本自动配置字体：
```bash
python scripts/setup_environment.py
```

### Q: 如何切换 CPU/GPU 训练？

A: 修改训练脚本中的 `device` 参数：
```python
agent = SACAgent(env=env, device="cpu")  # 使用 CPU
agent = SACAgent(env=env, device="cuda")  # 使用 GPU
```

### Q: 训练时间太长？

A: 可以调整训练参数：
```python
total_timesteps = 50000  # 减少训练步数
batch_size = 512        # 增大 batch size
```

## 技术栈

- **数值计算**: NumPy, SciPy
- **数据处理**: Pandas, OpenPyXL
- **机器学习**: PyTorch, Stable-Baselines3
- **强化学习**: Gymnasium
- **可视化**: Matplotlib

## 许可证

MIT License

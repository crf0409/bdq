# IEEE33微电网控制系统 - 安装指南

## 系统要求

- **操作系统**: Windows 10/11, macOS, Linux
- **Python**: 3.9 或更高版本
- **内存**: 至少 4GB RAM（推荐 8GB）
- **存储**: 至少 2GB 可用空间

## 快速安装（Windows）

### 方法一：使用批处理脚本（推荐）

双击运行 `install.bat`，脚本会自动完成：
1. 升级 pip
2. 安装所有依赖
3. 配置中文字体
4. 创建必要的目录

### 方法二：手动安装

```bash
# 1. 升级 pip
python -m pip install --upgrade pip

# 2. 安装依赖
pip install -r requirements.txt

# 3. 初始化环境
python scripts/setup_environment.py
```

## 验证安装

```bash
# 运行环境测试
python test_env.py
```

预期输出：
```
============================================================
IEEE33微电网控制系统 - 环境测试
============================================================

[1/4] 测试依赖导入...
  ✓ numpy
  ✓ pandas
  ✓ matplotlib
  ✓ gymnasium
  ✓ stable_baselines3
  ✓ torch

[2/4] 测试字体配置...
  ✓ 字体配置成功: Microsoft YaHei

[3/4] 测试电网模型...
  ✓ IEEE33Bus 模型创建成功
  ✓ 潮流计算收敛 (迭代次数: 3)

[4/4] 测试数据加载...
  ✓ 训练数据加载成功
  ✓ 测试数据加载成功

✓ 所有测试通过！环境配置正确。
```

## 运行项目

### 使用菜单脚本（Windows）

双击 `run.bat`，选择要运行的任务：

```
[1] 任务2: 光伏影响分析 (60秒仿真)
[2] 任务3: 下垂控制仿真
[3] 任务4: 训练SAC智能体
[4] 评估SAC模型
[5] 运行全部任务
[6] 环境检查
[0] 退出
```

### 使用命令行

```bash
# 任务2: 光伏影响分析
python scripts/task2_pv_impact.py

# 任务3: 下垂控制仿真
python scripts/task3_droop_control.py

# 任务4: 训练SAC智能体
python scripts/train_sac.py

# 评估SAC模型
python scripts/evaluate_sac.py
```

## 常见问题

### 1. 中文显示乱码

**解决方案**: 运行环境初始化脚本
```bash
python scripts/setup_environment.py
```

脚本会自动检测系统中可用的中文字体并配置 matplotlib。

### 2. 安装速度慢

**解决方案**: 使用国内镜像

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或使用 setup_environment.py，它会自动尝试多个国内镜像。

### 3. 提示缺少某个模块

**解决方案**: 重新安装依赖
```bash
pip install -r requirements.txt --force-reinstall
```

### 4. GPU 训练问题

如果需要使用 GPU 训练，请确保：
1. 安装 CUDA 工具包
2. 安装对应版本的 PyTorch

```bash
# 使用GPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 目录结构

安装完成后，项目目录结构如下：

```
bdq/
├── config/
│   └── config.yaml          # 配置文件
├── src/                     # 源代码
│   ├── agents/              # RL智能体
│   ├── control/             # 控制策略
│   ├── env/                 # 强化学习环境
│   ├── grid/                # 电网模型
│   └── utils/               # 工具函数
├── scripts/                 # 运行脚本
├── results/                 # 输出结果
│   ├── data/               
│   ├── figures/
│   └── models/
├── logs/                    # 日志文件
├── 训练集300天_6节点.xlsx   # 训练数据
├── 测试集20天_6节点.xlsx    # 测试数据
├── requirements.txt         # 依赖列表
├── install.bat              # Windows安装脚本
├── run.bat                  # Windows运行菜单
└── test_env.py              # 环境测试
```

## 依赖列表

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | >=1.24.0 | 数值计算 |
| pandas | >=2.0.0 | 数据处理 |
| matplotlib | >=3.7.0 | 可视化 |
| pyyaml | >=6.0 | 配置解析 |
| openpyxl | >=3.1.0 | Excel读写 |
| gymnasium | >=0.29.0 | 强化学习环境 |
| stable-baselines3 | >=2.2.0 | SAC算法 |
| torch | >=2.0.0 | 深度学习框架 |

## 技术支持

如有问题，请：
1. 先运行 `python test_env.py` 检查环境
2. 查看 `logs/` 目录下的日志文件
3. 检查数据文件是否存在

## 更新日志

### v1.0.0 (2024)
- 初始版本
- 支持 Windows 本地化
- 自动字体配置
- 一键安装脚本

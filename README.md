# bdq | 气象水文时序预测管道

基于深度学习的气象/水文多节点时序数据预测系统，支持多站点联合训练与评估。

## 数据规格

- 训练集：300 天历史数据
- 测试集：20 天独立数据
- 节点数：6 个监测站点

## 功能特性

- 多站点时序数据清洗与对齐
- 滑动窗口特征工程
- 多模型对比（LSTM / GRU / Transformer）
- 自动化结果汇报与可视化

## 快速开始

```bash
pip install -r requirements.txt

# 数据预处理
python scripts/preprocess.py --config config/default.yaml

# 训练
python scripts/train.py --config config/default.yaml

# 评估
python scripts/evaluate.py --config config/default.yaml
```

## 目录结构

```
bdq/
├── src/          # 核心模型与训练代码
├── scripts/      # 数据处理与实验脚本
├── config/       # 配置文件（YAML）
├── results/      # 实验结果
└── gj/           # 辅助工具
```

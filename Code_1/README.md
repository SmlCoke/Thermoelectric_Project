# 时间序列预测项目 - 基于CNN-LSTM的云量/辐射短期预报

本目录包含基于深度学习的时间序列预测模型，用于预测热电芯片采集的多通道电压数据，实现云量变化和太阳能辐射的短期预报。

## 🎯 项目目标

利用热电芯片采集的高频时间序列数据（VSW黑/红/蓝/绿，VLW黑/红/蓝/绿），建立深度学习模型：
- **输入**：过去10小时的8通道电压数据（采样间隔10秒）
- **输出**：未来6小时的8通道电压预测值
- **应用**：
  - 预测云量变化（VSW↓ + VLW↑ = 云来了）
  - 太阳能发电波动预警
  - 体感温度变化预警

## 📁 目录结构

```
Code_1/
├── README.md                    # 本文件
├── lightweight_version/         # 轻量级版本（适合个人PC）
│   ├── config.py               # 配置文件
│   ├── model.py                # CNN-LSTM模型
│   ├── dataset.py              # 数据加载器
│   ├── generate_data.py        # 生成模拟数据
│   ├── train.py                # 训练脚本
│   ├── test.py                 # 测试脚本
│   ├── requirements.txt        # 依赖列表
│   └── README.md              # 详细说明
└── full_version/               # 完整版本（适合服务器）
    ├── config.py               # 配置文件（服务器级参数）
    ├── model.py                # 高级模型（含注意力机制）
    ├── dataset.py              # 数据加载器（含增强）
    ├── generate_data.py        # 生成大规模数据
    ├── train.py                # 训练脚本（混合精度+多GPU）
    ├── test.py                 # 测试脚本
    ├── requirements.txt        # 依赖列表
    └── README.md              # 详细说明
```

## 🚀 快速开始

### 选择合适的版本

#### 轻量级版本 - 推荐新手和PC用户

**适用场景**：
- ✅ 个人PC（NVIDIA RTX 4060 / 3060 等）
- ✅ 显存 8GB 左右
- ✅ 系统内存 16GB
- ✅ 快速实验和学习
- ✅ 首次接触深度学习时间序列预测

**特点**：
- 模型参数量：~140K
- 训练时间：5-10分钟
- 数据规模：7天
- 简单易懂，适合学习

**使用方法**：
```bash
cd lightweight_version
python generate_data.py  # 生成数据
python train.py          # 训练模型
python test.py           # 测试评估
```

#### 完整版本 - 推荐有经验用户和服务器

**适用场景**：
- ✅ 服务器GPU（V100, A100, RTX 3090等）
- ✅ 显存 16GB 以上
- ✅ 系统内存 32GB 以上
- ✅ 追求最佳性能
- ✅ 生产级应用

**特点**：
- 模型参数量：~2.5M
- 训练时间：30-60分钟
- 数据规模：30天
- 包含注意力机制、混合精度训练等高级特性

**使用方法**：
```bash
cd full_version
python generate_data.py  # 生成数据
python train.py          # 训练模型
python test.py           # 测试评估
```

## 📊 两个版本对比

| 特性 | 轻量级版本 | 完整版本 |
|------|-----------|---------|
| **目标用户** | 新手、学习者 | 有经验的用户 |
| **硬件要求** | RTX 4060 (8GB) | RTX 3090 (24GB) |
| **模型大小** | ~140K 参数 | ~2.5M 参数 |
| **训练时间** | 5-10 分钟 | 30-60 分钟 |
| **数据规模** | 7天 (~15MB) | 30天 (~65MB) |
| **批次大小** | 16 | 64 |
| **模型深度** | 浅层网络 | 深层网络 |
| **注意力机制** | ❌ | ✅ |
| **混合精度** | ❌ | ✅ |
| **数据增强** | ❌ | ✅ |
| **多GPU支持** | ❌ | ✅ |
| **预期性能** | 良好 | 优秀 (提升20-30%) |
| **推荐场景** | 学习、快速原型 | 生产、最佳性能 |

## 🎓 学习路径

如果你是新手，建议按以下顺序学习：

1. **阅读学习指南**：`../Initial/Study.md`
2. **理解项目背景**：`../Initial/完整方案书.md`
3. **从轻量级版本开始**：
   - 运行 `generate_data.py` 理解数据结构
   - 阅读 `model.py` 理解CNN-LSTM架构
   - 运行 `train.py` 体验训练流程
   - 运行 `test.py` 查看预测结果
4. **实验和调参**：
   - 修改 `config.py` 中的参数
   - 观察不同参数对性能的影响
5. **升级到完整版本**：
   - 熟悉轻量级版本后
   - 学习注意力机制等高级概念
   - 在服务器上训练完整版本

## 🔧 环境配置

### Python环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
cd lightweight_version  # 或 full_version
pip install -r requirements.txt
```

### 验证CUDA

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

## 📈 典型工作流程

### 1. 使用模拟数据训练（学习阶段）

```bash
# 步骤1：生成模拟数据
python generate_data.py

# 步骤2：训练模型
python train.py

# 步骤3：评估和可视化
python test.py

# 步骤4：查看训练曲线
tensorboard --logdir=logs/
```

### 2. 使用真实数据训练（应用阶段）

```python
# 准备你的真实数据
# 格式: numpy数组，shape = (时间步数, 8通道)
import numpy as np

# 假设你已经采集了数据
real_data = your_collected_data  # shape: (N, 8)

# 保存为.npz格式
np.savez('data/real_data.npz', data=real_data)

# 修改config.py中的DATA_PATH
# DATA_PATH = 'data/real_data.npz'

# 然后正常训练
python train.py
```

## 🎯 预期结果

### 训练过程

- 训练损失逐渐下降
- 验证损失先下降后平稳
- 早停法自动停止训练

### 测试性能

**轻量级版本**：
- MAE: 0.03-0.05
- RMSE: 0.04-0.07

**完整版本**：
- MAE: 0.02-0.04 (提升20-30%)
- RMSE: 0.03-0.05

### 可视化结果

- 预测曲线 vs 真实曲线
- 各通道误差分布
- 云遮挡事件识别示例

## 📝 代码说明

### 核心文件

1. **config.py**：所有超参数配置
2. **model.py**：神经网络模型定义
3. **dataset.py**：数据加载和预处理
4. **train.py**：训练循环和优化
5. **test.py**：测试和可视化
6. **generate_data.py**：生成模拟数据

### 关键概念

- **滑动窗口**：将时间序列转换为监督学习样本
- **CNN层**：提取局部时序特征
- **LSTM层**：捕捉长期时间依赖
- **注意力机制**（完整版）：关注重要时间步
- **数据归一化**：StandardScaler标准化
- **早停法**：防止过拟合

## 🐛 常见问题

### Q: CUDA Out of Memory?
**A**: 减小 `BATCH_SIZE`，或使用轻量级版本

### Q: 训练很慢?
**A**: 确认使用了GPU，检查 `torch.cuda.is_available()`

### Q: 模型不收敛?
**A**: 降低学习率，检查数据归一化，增加训练数据

### Q: 如何在真实数据上训练?
**A**: 准备numpy数组格式数据，修改config.py中的DATA_PATH

### Q: 两个版本可以互相转换吗?
**A**: 数据格式兼容，但模型架构不同，权重不能直接迁移

## 📚 相关文档

- **学习指南**：`../Initial/Study.md` - 完整的学习路径和课程推荐
- **项目方案**：`../Initial/完整方案书.md` - 项目背景和目标
- **数据采集**：`../Initial/数据采集方案.md` - 硬件方案说明

## 🤝 贡献

欢迎改进代码！可以尝试：
- 添加新的模型架构（Transformer等）
- 优化超参数
- 改进数据增强策略
- 添加更多评估指标
- 优化训练速度

## 📄 许可

本项目用于课程学习和研究目的。

---

**开始你的时间序列预测之旅！** 🚀

如有问题，请查看各版本目录下的详细README文档。

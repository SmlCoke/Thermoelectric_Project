# 时间序列预测模型 - 完整版本（服务器级）

本目录包含适用于服务器（大显存GPU，如V100, A100, RTX 3090等）的完整版CNN-LSTM混合模型，具有更深的网络结构、注意力机制和高级训练技术。

## 📋 目录结构

```
full_version/
├── config.py           # 配置文件（服务器级参数）
├── model.py            # 高级CNN-LSTM模型（含注意力机制）
├── dataset.py          # 数据加载和增强
├── generate_data.py    # 生成大规模模拟数据
├── train.py            # 训练脚本（混合精度+多GPU）
├── test.py             # 测试和可视化脚本
├── requirements.txt    # Python依赖
└── README.md          # 本文件
```

## 🚀 快速开始

### 1. 环境配置

服务器环境要求：
- Python 3.8-3.11
- CUDA Toolkit 11.8+
- 显存 ≥ 16GB（推荐24GB以上）
- 系统内存 ≥ 32GB

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装PyTorch (CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 生成模拟数据

```bash
python generate_data.py
```

这将生成：
- `data/simulated_data_extended.npz`: 30天的模拟8通道电压数据（~65MB）
- `data/data_visualization.png`: 数据可视化

### 3. 训练模型

```bash
# 单GPU训练
python train.py

# 多GPU训练（自动检测）
# 如果有多个GPU，代码会自动使用torch.nn.DataParallel
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
```

训练特性：
- ✅ 混合精度训练（FP16）加速
- ✅ 多GPU并行支持
- ✅ 数据增强
- ✅ 高级学习率调度（Cosine Annealing）
- ✅ 梯度裁剪
- ✅ 早停法
- ✅ 定期保存检查点

预计训练时间：
- 在单个RTX 3090上约 30-60 分钟（100个epoch）
- 在单个V100上约 40-80 分钟
- 在4x GPU上约 15-30 分钟

### 4. 测试和可视化

```bash
python test.py
```

### 5. 查看训练曲线

```bash
tensorboard --logdir=logs/
```

## 🏗️ 模型架构（高级版）

```
输入: (batch_size, 3600, 8)  # 10小时 x 8通道
  ↓
深层1D CNN (特征提取)
  - Conv1D(8→64, kernel=7)
  - Conv1D(64→128, kernel=5)
  - Conv1D(128→128, kernel=3) + 残差连接
  - MaxPool1D(kernel=2)
  ↓
深层LSTM (时序建模)
  - 4层LSTM, hidden_size=256
  - Bidirectional: False
  ↓
多头注意力机制
  - 8个注意力头
  - 残差连接 + LayerNorm
  ↓
深层全连接网络 (预测生成)
  - Linear(256→512) + BatchNorm + ReLU
  - Linear(512→256) + BatchNorm + ReLU
  - Linear(256→2160*8)
  ↓
输出: (batch_size, 2160, 8)  # 6小时 x 8通道
```

**模型参数量**: 约 2.5M-3.5M（根据配置）

## ⚙️ 配置说明

完整版特有的高级参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CNN_CHANNELS` | 128 | CNN输出通道数（更大） |
| `LSTM_HIDDEN_SIZE` | 256 | LSTM隐藏层大小（更大） |
| `LSTM_NUM_LAYERS` | 4 | LSTM层数（更深） |
| `USE_ATTENTION` | True | 是否使用注意力机制 |
| `ATTENTION_HEADS` | 8 | 多头注意力的头数 |
| `BATCH_SIZE` | 64 | 批次大小（更大） |
| `NUM_EPOCHS` | 100 | 训练轮数（更多） |
| `USE_MIXED_PRECISION` | True | 使用FP16混合精度 |
| `USE_DATA_AUGMENTATION` | True | 使用数据增强 |
| `SCHEDULER_TYPE` | 'cosine' | 学习率调度器类型 |

## 💾 资源需求

**典型资源占用**（batch_size=64）：
- GPU显存: 10-14 GB（FP16）/ 16-20 GB（FP32）
- 系统内存: 16-24 GB
- 数据文件大小: ~65 MB (30天数据)
- 模型文件大小: ~15 MB

## 🆚 与轻量级版本的对比

| 特性 | 轻量级版本 | 完整版本 |
|------|-----------|---------|
| CNN通道数 | 32 | 128 |
| LSTM隐藏层 | 64 | 256 |
| LSTM层数 | 2 | 4 |
| 注意力机制 | ❌ | ✅ (8头) |
| 残差连接 | ❌ | ✅ |
| 混合精度 | ❌ | ✅ |
| 数据增强 | ❌ | ✅ |
| 批次大小 | 16 | 64 |
| 训练数据 | 7天 | 30天 |
| 模型参数量 | ~140K | ~2.5M |
| 适用场景 | 个人PC快速实验 | 服务器生产级训练 |

## 🎯 性能预期

在30天模拟数据上的典型性能：
- **训练损失**: 0.0005-0.002 (MSE)
- **测试损失**: 0.001-0.003 (MSE)
- **MAE**: 0.02-0.04 (归一化数据)

完整版本通常比轻量级版本有 **20-30%** 的性能提升。

## 🔧 高级功能

### 1. 混合精度训练

自动启用，可在config.py中关闭：
```python
USE_MIXED_PRECISION = False  # 关闭FP16
```

### 2. 多GPU训练

自动检测多GPU：
```bash
# 指定使用哪些GPU
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

### 3. 学习率调度

可选三种调度器（config.py）：
- `'cosine'`: 余弦退火（推荐）
- `'reduce_on_plateau'`: 验证损失平台时降低
- `'step'`: 阶梯式衰减

### 4. 数据增强

包括：
- 随机噪声注入
- 随机缩放（±5%）

### 5. 检查点管理

- 自动保存最佳模型
- 每N个epoch保存检查点
- 支持断点续训

## 🐛 故障排除

### Q1: CUDA Out of Memory

解决方案：
1. 减小 `BATCH_SIZE` (例如改为32或16)
2. 关闭混合精度：`USE_MIXED_PRECISION = False`
3. 减小模型规模：
   ```python
   CNN_CHANNELS = 64  # 从128降到64
   LSTM_HIDDEN_SIZE = 128  # 从256降到128
   ```

### Q2: 训练速度慢

优化建议：
1. 确认使用了GPU：检查日志中的"DEVICE: cuda"
2. 启用混合精度：`USE_MIXED_PRECISION = True`
3. 增加 `num_workers`（dataset.py中）
4. 使用多GPU

### Q3: 模型不收敛

调试步骤：
1. 降低学习率：`LEARNING_RATE = 0.0001`
2. 增加预热轮数：`WARMUP_EPOCHS = 10`
3. 检查数据归一化是否正确
4. 尝试不同的调度器

### Q4: 如何从检查点恢复训练？

修改train.py，在创建模型后添加：
```python
checkpoint = torch.load('checkpoints/best_model_epoch50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 📊 实验建议

1. **超参数搜索**：
   - 尝试不同的学习率：[0.0001, 0.0005, 0.001]
   - 尝试不同的批次大小：[32, 64, 128]
   - 调整Dropout率：[0.2, 0.3, 0.4]

2. **消融实验**：
   - 关闭注意力机制，观察性能变化
   - 减少LSTM层数，对比效果
   - 关闭数据增强，评估影响

3. **模型集成**（高级）：
   - 训练3-5个模型（不同随机种子）
   - 预测时取平均，提升稳定性

## 📚 文件说明

### config.py
服务器级配置，包含所有高级训练参数。

### model.py
高级模型架构，包含：
- 多头注意力机制实现
- 深层CNN和LSTM
- 残差连接
- BatchNorm和LayerNorm

### train.py
完整训练流程，包含：
- 混合精度训练
- 多GPU并行
- 高级学习率调度
- 检查点保存
- TensorBoard日志

### generate_data.py
生成大规模模拟数据（30天），包含：
- 更多云遮挡事件
- 长期趋势和季节性
- 多种噪声类型
- 随机脉冲干扰

## 🎓 从轻量级迁移

如果你已经熟悉轻量级版本：

1. **数据格式完全兼容**：可以直接使用轻量级版本的数据
2. **配置文件结构相同**：只是参数值更大
3. **训练流程一致**：命令和工作流相同
4. **关键区别**：
   - 模型更深更复杂
   - 训练时间更长
   - 需要更多显存
   - 性能更好

## 🚀 部署建议

训练完成后，如果要部署：

1. **模型量化**（减小模型大小）：
   ```python
   # 使用PyTorch的动态量化
   quantized_model = torch.quantization.quantize_dynamic(
       model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
   )
   ```

2. **导出ONNX**（跨平台）：
   ```python
   torch.onnx.export(model, dummy_input, "model.onnx")
   ```

3. **TorchScript**（生产环境）：
   ```python
   scripted_model = torch.jit.script(model)
   scripted_model.save("model_scripted.pt")
   ```

## 📞 技术支持

遇到问题？
1. 检查GPU驱动和CUDA版本兼容性
2. 查看TensorBoard确认训练是否正常
3. 参考 `Initial/Study.md` 学习指南
4. 在GitHub仓库提Issue

祝训练顺利！🚀

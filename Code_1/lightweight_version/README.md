# 时间序列预测模型 - 轻量级版本

本目录包含适用于个人PC（NVIDIA RTX 4060 + 16GB RAM）的轻量级CNN-LSTM混合模型。

## 📋 目录结构

```
lightweight_version/
├── config.py           # 配置文件（所有超参数）
├── model.py            # CNN-LSTM模型定义
├── dataset.py          # 数据加载和预处理
├── generate_data.py    # 生成模拟数据
├── train.py            # 训练脚本
├── test.py             # 测试和可视化脚本
├── requirements.txt    # Python依赖
└── README.md          # 本文件
```

## 🚀 快速开始

### 1. 环境配置

确保你的PC已经安装了：
- Python 3.8-3.11
- CUDA Toolkit 11.8+ (如果使用GPU)
- NVIDIA驱动程序

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

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
- `data/simulated_data.npz`: 7天的模拟8通道电压数据
- `data/data_visualization.png`: 数据可视化
- `data/cloud_event_example.png`: 云遮挡事件示例

### 3. 训练模型

```bash
python train.py
```

训练过程：
- 自动划分训练集/验证集/测试集 (70%/15%/15%)
- 使用早停法防止过拟合
- 自动保存最佳模型到 `checkpoints/best_model.pth`
- 训练日志保存到 `logs/` (可用TensorBoard查看)

预计训练时间：
- 在NVIDIA RTX 4060上约 5-10 分钟（50个epoch）
- 在CPU上约 30-60 分钟

### 4. 测试和可视化

```bash
python test.py
```

这将生成：
- 测试集性能指标（MAE, RMSE, MAPE）
- 多个预测样本的可视化图
- 各通道误差分布图
- 云遮挡事件预测示例

结果保存在 `results/` 目录。

### 5. 查看训练曲线（可选）

```bash
tensorboard --logdir=logs/
```

然后在浏览器打开 http://localhost:6006

## 📊 模型架构

```
输入: (batch_size, 3600, 8)  # 10小时 x 8通道
  ↓
1D CNN层 (特征提取)
  - Conv1D(8→32, kernel=7)
  - Conv1D(32→32, kernel=5)
  - MaxPool1D(kernel=2)
  ↓
LSTM层 (时序建模)
  - 2层LSTM, hidden_size=64
  ↓
全连接层 (预测生成)
  - Linear(64→64)
  - Linear(64→2160*8)
  ↓
输出: (batch_size, 2160, 8)  # 6小时 x 8通道
```

**模型参数量**: 约 140K (非常轻量)

## ⚙️ 配置说明

主要超参数在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `INPUT_HOURS` | 10 | 输入序列长度（小时） |
| `OUTPUT_HOURS` | 6 | 输出序列长度（小时） |
| `BATCH_SIZE` | 16 | 批次大小 |
| `NUM_EPOCHS` | 50 | 训练轮数 |
| `LEARNING_RATE` | 0.001 | 初始学习率 |
| `CNN_CHANNELS` | 32 | CNN输出通道数 |
| `LSTM_HIDDEN_SIZE` | 64 | LSTM隐藏层大小 |

你可以根据自己的硬件调整这些参数。

## 💾 内存和显存使用

**典型资源占用**（batch_size=16）：
- GPU显存: ~2-3 GB
- 系统内存: ~4-6 GB
- 数据文件大小: ~15 MB (7天数据)

如果遇到 `CUDA out of memory` 错误：
1. 减小 `BATCH_SIZE` (例如改为8)
2. 关闭其他占用GPU的程序
3. 在 `config.py` 中减小 `CNN_CHANNELS` 或 `LSTM_HIDDEN_SIZE`

## 📈 性能预期

在模拟数据上的典型性能：
- **训练损失**: 0.001-0.005 (MSE)
- **测试损失**: 0.002-0.008 (MSE)
- **MAE**: 0.03-0.05 (归一化数据)

真实数据的性能会因数据质量而异。

## 🔧 常见问题

### Q1: 如何在真实数据上训练？

修改 `config.py` 中的 `DATA_PATH`，然后准备你的数据为以下格式：
```python
# 保存为 .npz 文件
np.savez('your_data.npz', data=your_array)
# your_array shape: (时间步数, 8)
```

### Q2: 如何调整输入/输出长度？

修改 `config.py` 中的 `INPUT_HOURS` 和 `OUTPUT_HOURS`，然后重新生成数据和训练。

### Q3: 模型不收敛怎么办？

1. 检查数据是否正确归一化
2. 降低学习率（例如改为0.0001）
3. 增加batch size
4. 检查GPU是否正常工作：`print(torch.cuda.is_available())`

### Q4: 如何使用CPU训练？

模型会自动检测。如果没有GPU，会使用CPU（速度较慢）。

## 📚 文件说明

### config.py
包含所有配置参数。修改此文件来调整模型和训练设置。

### model.py
CNN-LSTM混合模型的PyTorch实现。可以运行 `python model.py` 测试模型架构。

### dataset.py
数据加载器，包括：
- 滑动窗口生成训练样本
- 数据归一化
- 训练/验证/测试集划分

### generate_data.py
生成模拟数据，包含：
- 日周期变化
- 云遮挡事件
- 不同颜色芯片的响应差异
- 随机噪声

### train.py
完整训练流程，包括：
- 模型训练
- 验证和早停
- 最佳模型保存
- TensorBoard日志

### test.py
测试和可视化，包括：
- 性能指标计算
- 预测结果可视化
- 误差分析
- 云遮挡事件识别示例

## 🎯 下一步

1. **理解代码**: 阅读每个文件，理解数据流和模型架构
2. **实验调参**: 尝试调整超参数，观察性能变化
3. **真实数据**: 接入真实的热电芯片数据
4. **模型改进**: 尝试添加注意力机制或其他高级特性
5. **部署应用**: 将模型部署到树莓派进行实时预测

## 📞 技术支持

遇到问题？
1. 查看各脚本的注释
2. 参考 `Initial/Study.md` 学习指南
3. 在GitHub仓库提Issue

祝训练顺利！🚀

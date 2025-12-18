# 热电芯片天空辐射时间序列预测项目

基于 PyTorch 的 LSTM/GRU 时间序列预测系统，用于分析热电芯片在不同波段下的辐射强度数据。

## 项目简介

本项目旨在对热电芯片天空辐射采集实验数据进行时间序列预测。我们使用八个相同的热电芯片配合不同波段的滤光片，在太阳灶焦点位置测量不同波段的辐射强度。

### 数据特点

- **8个通道**：8个热电芯片对应8个电压通道
- **固定采样间隔**：约10秒/次
- **片段式数据**：每天的数据是独立的时间序列片段
- **不连续**：不同日期之间的数据完全不连续，不进行插值

### 技术栈

- **深度学习框架**：PyTorch
- **模型**：LSTM / GRU
- **加速**：CUDA (RTX 4060)
- **可视化**：TensorBoard, Matplotlib
- **数据处理**：NumPy, Pandas, scikit-learn

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+ (可选，推荐)
- GPU: NVIDIA RTX 4060 或同等性能

### 安装依赖

```bash
cd TimeSeries
pip install -r requirements.txt
```

### 快速训练

```bash
# 进入源代码目录
cd src

# 训练GRU模型（推荐，快速）
python train.py --model gru

# 或训练LSTM模型
python train.py --model lstm
```

训练完成后，模型会保存在 `checkpoints/` 目录。

### 快速预测

```bash
# 使用训练好的模型进行预测
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../../1122.csv \
    --start_idx 100 \
    --plot
```

## 项目结构

```
TimeSeries/
├── README.md              # 本文件
├── requirements.txt       # Python依赖
├── 1122.csv              # 示例数据1
├── 1129.csv              # 示例数据2
├── src/                  # 源代码
│   ├── dataset.py        # 数据加载
│   ├── model_gru.py      # GRU模型
│   ├── model_lstm.py     # LSTM模型
│   ├── train.py          # 训练脚本
│   └── predict.py        # 预测脚本
├── docs/                 # 文档
│   ├── time_series_intro.md
│   ├── dataset.md
│   ├── model_gru.md
│   ├── model_lstm.md
│   ├── train.md
│   ├── predict.md
│   └── structure.md
├── checkpoints/          # 模型保存（训练后生成）
└── logs/                 # 训练日志（训练后生成）
```

## 数据格式说明

### CSV文件格式

每个CSV文件代表一天的测量数据：

```csv
Timestamp,DateTime,TEC1_Optimal(V),TEC2_Optimal(V),...,TEC8_Optimal(V)
1763784663,12:11:02,0.006125,0.007125,...,0.00303125
1763784673,12:11:13,0.00575,0.007125,...,0.00325
...
```

**重要说明**：
- 每个文件是独立的时间序列片段
- 不同文件之间的数据不连续
- 模型不会跨文件建立时间依赖关系

### 添加新数据

只需将新的CSV文件放到 `TimeSeries/` 目录下即可：

```bash
# 复制新数据
cp your_new_data.csv TimeSeries/

# 重新训练（会自动加载所有CSV）
cd src
python train.py --model gru
```

## 使用指南

### 1. 训练模型

#### 基本训练

```bash
cd src
python train.py --model gru
```

#### 自定义参数

```bash
python train.py \
    --model gru \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 0.001
```

#### 训练时间建议

**快速原型（5分钟）**：
```bash
python train.py --model gru --hidden_size 64 --num_epochs 50
```

**推荐配置（15分钟）**：
```bash
python train.py --model gru --hidden_size 128 --num_epochs 100
```

**高性能配置（30分钟）**：
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 150
```

### 2. 监控训练

使用TensorBoard实时查看训练进度：

```bash
# 在新的终端窗口中
tensorboard --logdir=logs

# 浏览器访问
http://localhost:6006
```

### 3. 切换模型（GRU / LSTM）

#### 使用GRU

```bash
python train.py --model gru
```

**优点**：
- 训练速度快（约快30%）
- 参数少，不易过拟合
- 适合中短序列

#### 使用LSTM

```bash
python train.py --model lstm
```

**优点**：
- 记忆能力强
- 适合长序列
- 可能获得更好的性能

**建议**：先尝试GRU，如果效果不够好再尝试LSTM。

### 4. 进行预测

#### 基本预测

```bash
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../../1122.csv \
    --start_idx 100
```

#### 保存预测结果

```bash
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../../1122.csv \
    --start_idx 100 \
    --save_path predictions.npy
```

#### 可视化预测结果

```bash
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../../1122.csv \
    --start_idx 100 \
    --plot
```

会生成两个图像文件：
- `predictions_plot.png`：所有8个通道的预测对比
- `predictions_channel1.png`：第一个通道的详细预测

### 5. 评估模型

预测脚本会自动计算以下指标（如果有真实值）：

- **MSE (均方误差)**：平均预测误差的平方
- **MAE (平均绝对误差)**：平均预测误差的绝对值

```bash
python predict.py --model_path ../checkpoints/best_model.pth \
                  --csv_path ../../1122.csv --start_idx 100
```

输出示例：
```
均方误差 (MSE): 0.000123
平均绝对误差 (MAE): 0.008456
```

## 主要参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `../TimeSeries` | CSV文件所在目录 |
| `--window_size` | 60 | 输入序列长度（时间步） |
| `--predict_steps` | 10 | 预测未来的步数 |
| `--batch_size` | 64 | 批次大小 |
| `--stride` | 5 | 滑动窗口步长 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `gru` | 模型类型：`gru` 或 `lstm` |
| `--hidden_size` | 128 | 隐藏层大小 |
| `--num_layers` | 1 | RNN层数 |
| `--dropout` | 0.2 | Dropout比率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 100 | 训练轮数 |
| `--learning_rate` | 0.001 | 学习率 |
| `--early_stopping_patience` | 20 | Early stopping耐心值 |

## 常见问题

### 训练相关

**Q: 训练时显示 CUDA out of memory？**  
A: 减小 batch_size 或 hidden_size：
```bash
python train.py --batch_size 32 --hidden_size 64
```

**Q: 训练很慢？**  
A: 
1. 确保使用了GPU（检查是否显示 "使用设备: cuda"）
2. 增加 stride 减少样本数
3. 使用 GRU 代替 LSTM
4. 减小 hidden_size

**Q: 验证损失不下降？**  
A:
1. 降低学习率：`--learning_rate 0.0001`
2. 检查数据是否正确加载
3. 尝试更大的模型：`--hidden_size 256`

### 预测相关

**Q: 预测结果不准确？**  
A:
1. 确保使用了最佳模型（`best_model.pth`）
2. 增加训练数据
3. 调整模型参数
4. 尝试不同的 start_idx

**Q: 如何选择 start_idx？**  
A:
1. 避开数据开头和结尾
2. 确保后续有足够的真实值用于对比
3. 可以随机选择多个位置进行测试

### 其他问题

**Q: 可以在CPU上运行吗？**  
A: 可以，但速度会慢10-20倍。训练时间可能超过30分钟。

**Q: 如何保存和恢复训练？**  
A: 模型会自动保存检查点。如需恢复训练，需要修改代码加载检查点。

**Q: 如何对比不同模型的性能？**  
A: 
1. 训练多个模型并保存
2. 使用相同的测试数据进行预测
3. 对比MSE、MAE等指标
4. 可视化预测结果进行定性分析

## 性能基准

**测试环境**：RTX 4060, 16GB RAM

| 配置 | 模型 | Hidden | Layers | 训练时间 | 参数量 | 显存 |
|------|------|--------|--------|----------|--------|------|
| 快速 | GRU | 64 | 1 | ~5分钟 | ~30K | ~300MB |
| 推荐 | GRU | 128 | 1 | ~15分钟 | ~60K | ~500MB |
| 高性能 | LSTM | 256 | 2 | ~30分钟 | ~400K | ~1GB |

## 进阶使用

### Python API

可以直接在Python脚本中使用：

```python
# 数据加载
from dataset import create_dataloaders
train_loader, val_loader, dataset = create_dataloaders(
    data_dir='../TimeSeries',
    batch_size=32
)

# 模型创建
from model_gru import GRUModel
model = GRUModel(input_size=8, hidden_size=128)

# 训练
from train import Trainer
trainer = Trainer(model, train_loader, val_loader, device, config)
trainer.train()

# 预测
from predict import Predictor
predictor = Predictor('./checkpoints/best_model.pth')
predictions = predictor.predict(input_sequence)
```

### 自定义修改

- **新的损失函数**：修改 `train.py` 中的 `self.criterion`
- **新的模型架构**：创建新的模型文件并在 `train.py` 中注册
- **数据增强**：在 `dataset.py` 的 `__getitem__` 方法中添加

## 文档

详细文档请参考 `docs/` 目录：

- [时间序列与LSTM/GRU入门](docs/time_series_intro.md)
- [数据加载详解](docs/dataset.md)
- [GRU模型详解](docs/model_gru.md)
- [LSTM模型详解](docs/model_lstm.md)
- [训练指南](docs/train.md)
- [预测指南](docs/predict.md)
- [项目结构](docs/structure.md)

## 许可证

本项目用于学术研究和教育目的。

## 致谢

感谢所有为热电芯片天空辐射采集实验提供数据和支持的研究人员。

---

**如有问题或建议，请查阅文档或提出Issue。**

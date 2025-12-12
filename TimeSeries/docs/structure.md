# 项目文件结构说明

## 目录树

```
TimeSeries/
├── README.md                    # 项目主README
├── 1122.csv                     # 示例数据：11月22日测量数据
├── 1129.csv                     # 示例数据：11月29日测量数据
├── prompt.md                    # 项目需求文档
│
├── src/                         # 源代码目录
│   ├── dataset.py              # 数据加载和预处理模块
│   ├── model_lstm.py           # LSTM模型实现
│   ├── model_gru.py            # GRU模型实现
│   ├── train.py                # 训练脚本
│   └── predict.py              # 预测脚本
│
├── docs/                        # 文档目录
│   ├── time_series_intro.md    # 时间序列和LSTM/GRU入门教程
│   ├── dataset.md              # dataset.py详细说明
│   ├── model_lstm.md           # LSTM模型详细说明
│   ├── model_gru.md            # GRU模型详细说明
│   ├── train.md                # 训练脚本详细说明
│   ├── predict.md              # 预测脚本详细说明
│   └── structure.md            # 本文件：项目结构说明
│
├── checkpoints/                 # 模型保存目录（训练后生成）
│   ├── best_model.pth          # 最佳模型
│   ├── final_model.pth         # 最终模型
│   ├── checkpoint_epoch_XX.pth # 定期检查点
│   ├── scaler.pkl              # 数据标准化器
│   └── training_history.npy    # 训练历史记录
│
└── logs/                        # 训练日志目录（训练后生成）
    └── events.out.tfevents.xxx # TensorBoard日志文件
```

## 文件说明

### 数据文件

#### CSV数据格式

每个CSV文件包含一天的测量数据，格式如下：

```
Timestamp,DateTime,TEC1_Optimal(V),TEC2_Optimal(V),...,TEC8_Optimal(V)
1763784663,12:11:02,0.006125,0.007125,...,0.00303125
1763784673,12:11:13,0.00575,0.007125,...,0.00325
...
```

**列说明**：
- `Timestamp`: Unix时间戳
- `DateTime`: 人类可读的时间
- `TEC1_Optimal(V)` ~ `TEC8_Optimal(V)`: 8个热电芯片的电压值

### 源代码文件

#### dataset.py

**功能**：数据加载和预处理

**主要类/函数**：
- `ThermoelectricDataset`: 数据集类
- `create_dataloaders()`: 创建DataLoader的便捷函数

**输入**：CSV文件目录  
**输出**：PyTorch DataLoader

**使用**：
```python
from dataset import create_dataloaders
train_loader, val_loader, dataset = create_dataloaders(
    data_dir='../TimeSeries',
    batch_size=32
)
```

#### model_gru.py

**功能**：GRU模型实现

**主要类**：
- `GRUModel`: 主要的GRU预测模型
- `GRUEncoder`: 编码器（高级用法）
- `GRUDecoder`: 解码器（高级用法）

**特点**：
- 参数少，训练快
- 适合中短序列
- 推荐用于快速原型

**使用**：
```python
from model_gru import GRUModel
model = GRUModel(input_size=8, hidden_size=128)
```

#### model_lstm.py

**功能**：LSTM模型实现

**主要类**：
- `LSTMModel`: 主要的LSTM预测模型
- `LSTMEncoder`: 编码器（高级用法）
- `LSTMDecoder`: 解码器（高级用法）

**特点**：
- 记忆能力强
- 适合长序列
- 参数稍多

**使用**：
```python
from model_lstm import LSTMModel
model = LSTMModel(input_size=8, hidden_size=128)
```

#### train.py

**功能**：模型训练

**主要类/函数**：
- `Trainer`: 训练器类
- `main()`: 主函数

**使用**：
```bash
# 命令行
python train.py --model gru --hidden_size 128 --num_epochs 100

# Python脚本
from train import Trainer
trainer = Trainer(model, train_loader, val_loader, device, config)
trainer.train()
```

#### predict.py

**功能**：模型预测和可视化

**主要类/函数**：
- `Predictor`: 预测器类
- `plot_predictions()`: 绘制预测结果
- `plot_all_channels()`: 绘制所有通道

**使用**：
```bash
# 命令行
python predict.py --model_path ./checkpoints/best_model.pth \
                  --csv_path ../1122.csv --plot

# Python脚本
from predict import Predictor
predictor = Predictor('./checkpoints/best_model.pth')
predictions = predictor.predict(input_seq)
```

## 推荐的工作流程

### 第一次使用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练GRU模型（快速）
cd src
python train.py --model gru

# 3. 查看训练过程
tensorboard --logdir=../logs

# 4. 使用最佳模型预测
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../../1122.csv \
    --start_idx 100 \
    --plot

# 5. 查看结果
# 预测图像会保存在当前目录
```

### 添加新数据

```bash
# 1. 将新的CSV文件放到TimeSeries目录
cp new_data.csv ../TimeSeries/

# 2. 重新训练（会自动加载所有CSV）
python train.py --model gru

# 3. 对比新旧模型性能
python predict.py --model_path ../checkpoints/best_model.pth \
                  --csv_path ../new_data.csv --plot
```

### 切换模型

```bash
# 训练LSTM模型
python train.py --model lstm --hidden_size 256 --num_layers 2

# 对比GRU和LSTM
# （使用相同的测试数据）
python predict.py --model_path ../checkpoints/gru_best.pth ...
python predict.py --model_path ../checkpoints/lstm_best.pth ...
```

### 超参数调优

```bash
# 方案1：快速（5-10分钟）
python train.py --model gru --hidden_size 64 --batch_size 128

# 方案2：平衡（15分钟）
python train.py --model gru --hidden_size 128 --batch_size 64

# 方案3：强大（30分钟）
python train.py --model lstm --hidden_size 256 --num_layers 2

# 对比不同方案的预测结果
```

## 数据流程图

```
CSV文件 (原始数据)
    ↓
dataset.py (数据加载)
    ↓
DataLoader (批次迭代)
    ↓
model_*.py (模型定义)
    ↓
train.py (训练)
    ↓
checkpoints/ (保存模型)
    ↓
predict.py (预测)
    ↓
预测结果 + 可视化图像
```

## 常用命令速查

### 训练

```bash
# GRU快速训练
python train.py --model gru

# LSTM完整训练
python train.py --model lstm --hidden_size 256 --num_epochs 150

# 自定义所有参数
python train.py \
    --model gru \
    --data_dir ../../TimeSeries \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 0.001
```

### 预测

```bash
# 基本预测
python predict.py --model_path ../checkpoints/best_model.pth \
                  --csv_path ../../1122.csv

# 保存结果
python predict.py --model_path ../checkpoints/best_model.pth \
                  --csv_path ../../1122.csv \
                  --save_path result.npy

# 可视化
python predict.py --model_path ../checkpoints/best_model.pth \
                  --csv_path ../../1122.csv \
                  --plot
```

### 监控

```bash
# 启动TensorBoard
tensorboard --logdir=../logs

# 浏览器访问
# http://localhost:6006
```

## 开发建议

### 添加新功能

1. **新的模型架构**：
   - 在`src/`创建新的模型文件，如`model_transformer.py`
   - 参考`model_gru.py`的结构
   - 在`train.py`中添加对新模型的支持

2. **新的损失函数**：
   - 在`train.py`的`Trainer`类中修改`self.criterion`

3. **新的数据增强**：
   - 在`dataset.py`的`__getitem__`方法中添加

### 性能优化

1. **加速数据加载**：
   - 增加`num_workers`
   - 使用`pin_memory=True`

2. **加速训练**：
   - 使用混合精度训练（AMP）
   - 增大batch size
   - 使用GRU代替LSTM

3. **减少显存占用**：
   - 减小batch size
   - 减小hidden size
   - 使用梯度累积

### 调试技巧

1. **检查数据**：
   ```python
   dataset = ThermoelectricDataset(...)
   x, y = dataset[0]
   print(x.shape, y.shape, x.min(), x.max())
   ```

2. **检查模型**：
   ```python
   model = GRUModel(...)
   x = torch.randn(1, 60, 8)
   output, _ = model(x)
   print(output.shape)
   ```

3. **检查训练**：
   - 观察loss是否下降
   - 检查学习率是否合适
   - 使用TensorBoard可视化

## 扩展阅读

- `docs/time_series_intro.md`: 时间序列基础知识
- `docs/dataset.md`: 数据处理详解
- `docs/model_gru.md`: GRU原理和使用
- `docs/model_lstm.md`: LSTM原理和使用
- `docs/train.md`: 训练技巧和调优
- `docs/predict.md`: 预测和评估方法

## 常见问题

**Q: 如何选择GRU还是LSTM？**  
A: 先尝试GRU（更快），如果效果不够好再试LSTM。

**Q: 训练需要多久？**  
A: 根据配置，5-30分钟不等。推荐配置约15分钟。

**Q: 如何评估模型好坏？**  
A: 
1. 查看验证损失
2. 使用predict.py生成预测并可视化
3. 计算MSE、MAE等指标

**Q: 可以在CPU上训练吗？**  
A: 可以，但会慢很多（10-20倍）。

**Q: 如何恢复中断的训练？**  
A: 从检查点恢复（需要修改代码加载检查点）。

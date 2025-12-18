# model_lstm.py 解释文档

## 概述

`model_lstm.py` 实现了基于LSTM（长短期记忆网络）的时间序列预测模型。LSTM是最经典的循环神经网络变体，擅长捕获长期依赖关系。

## LSTM原理简介

### 门控机制

LSTM使用三个门来控制信息流：

1. **遗忘门 (Forget Gate)**：决定丢弃哪些历史信息
2. **输入门 (Input Gate)**：决定保存哪些新信息
3. **输出门 (Output Gate)**：决定输出哪些信息

### 数学公式

```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)      # 遗忘门
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)      # 输入门
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)      # 输出门
c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)   # 候选细胞状态
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ         # 新细胞状态
hₜ = oₜ ⊙ tanh(cₜ)               # 新隐藏状态
```

### 关键概念

- **细胞状态 (Cell State)**：长期记忆，像传送带一样贯穿整个序列
- **隐藏状态 (Hidden State)**：短期记忆，当前时刻的输出

## 模型架构

### LSTMModel类

```
输入 [batch, seq_len, 8]
    ↓
LSTM层 (多层可选)
    ↓
取最后时间步输出 [batch, hidden_size]
    ↓
Dropout层
    ↓
全连接层 [batch, predict_steps * 8]
    ↓
重塑 [batch, predict_steps, 8]
    ↓
输出预测
```

### 初始化参数

```python
LSTMModel(
    input_size=8,        # 输入特征数（8个电压通道）
    hidden_size=128,     # 隐藏层大小
    num_layers=1,        # LSTM层数
    output_size=8,       # 输出特征数
    predict_steps=10,    # 预测步数
    dropout=0.2          # Dropout比率
)
```

## 核心方法详解

### 1. `__init__()`

初始化模型组件。

```python
# LSTM层
self.lstm = nn.LSTM(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    batch_first=True,
    dropout=0.2 if num_layers > 1 else 0
)

# Dropout层
self.dropout_layer = nn.Dropout(0.2)

# 全连接层
self.fc = nn.Linear(hidden_size, predict_steps * output_size)
```

### 2. `forward(x, hidden=None)`

前向传播，核心预测逻辑。

**输入**：
- `x`: [batch_size, seq_len, 8] - 输入序列
- `hidden`: tuple (h_0, c_0) - 初始隐藏状态和细胞状态（可选）
  - `h_0`: [num_layers, batch_size, hidden_size]
  - `c_0`: [num_layers, batch_size, hidden_size]

**输出**：
- `output`: [batch_size, predict_steps, 8] - 预测结果
- `hidden`: tuple (h_n, c_n) - 最终隐藏状态和细胞状态

**维度变化示意图**：

```
输入 x: [32, 60, 8]
    ↓
LSTM处理
    ↓
lstm_out: [32, 60, 128]  # 每个时间步的输出
h_n: [1, 32, 128]        # 最终隐藏状态
c_n: [1, 32, 128]        # 最终细胞状态
    ↓
取最后时间步 last_output: [32, 128]
    ↓
Dropout
    ↓
全连接层 fc_out: [32, 80]  # 80 = 10 * 8
    ↓
重塑 output: [32, 10, 8]
```

**代码逻辑**：

```python
# LSTM前向传播
lstm_out, hidden = self.lstm(x, hidden)
# lstm_out: [batch, seq, hidden]
# hidden: (h_n, c_n)

# 取最后一个时间步
last_output = lstm_out[:, -1, :]  # [batch, hidden]

# Dropout
last_output = self.dropout_layer(last_output)

# 全连接层预测
fc_out = self.fc(last_output)

# 重塑
output = fc_out.view(batch_size, self.predict_steps, self.output_size)
```

### 3. `_init_hidden(batch_size, device)`

初始化隐藏状态和细胞状态为零。

```python
h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
return (h_0, c_0)
```

注意：LSTM返回的是元组 `(h, c)`，而GRU只返回 `h`。

### 4. `predict_multi_step(x, steps)`

多步迭代预测。

```python
# 获取初始状态
_, hidden = self.lstm(x, hidden)

# 初始输入
current_input = x[:, -1:, :]

# 迭代预测
for _ in range(steps):
    _, hidden = self.lstm(current_input, hidden)
    h_n, c_n = hidden
    prediction = self.fc(h_n[-1])
    next_step = prediction[:, 0:1, :]
    all_predictions.append(next_step)
    current_input = next_step
```

## LSTM特有特性

### 双重状态机制

LSTM同时维护两个状态：

1. **细胞状态 (c)**：
   - 长期记忆
   - 信息可以在整个序列中流动
   - 通过遗忘门和输入门更新

2. **隐藏状态 (h)**：
   - 短期记忆
   - 当前时刻的输出表示
   - 通过输出门和细胞状态计算

```python
# LSTM返回两个状态
lstm_out, (h_n, c_n) = self.lstm(x)

# GRU只返回一个状态
gru_out, h_n = self.gru(x)
```

### 梯度流动

LSTM通过细胞状态实现更好的梯度流动：

```
传统RNN: 梯度 = dL/dh₁ * dh₁/dh₀
         容易梯度消失（连乘）

LSTM:    梯度 = dL/dc₁ * dc₁/dc₀
         细胞状态通过加法更新
         梯度流动更稳定
```

## 参数量分析

### 单层LSTM

```
参数量 = 4 * (input_size + hidden_size + 1) * hidden_size
       = 4 * (8 + 128 + 1) * 128
       = 70,144
```

LSTM有4组参数（遗忘门、输入门、输出门、候选状态）。

### GRU vs LSTM 参数对比

```
GRU:  3 * (input_size + hidden_size + 1) * hidden_size
LSTM: 4 * (input_size + hidden_size + 1) * hidden_size

LSTM参数量 ≈ GRU参数量 * 1.33
```

## 使用示例

### 基本使用

```python
from model_lstm import LSTMModel
import torch

# 创建模型
model = LSTMModel(
    input_size=8,
    hidden_size=128,
    num_layers=2,
    predict_steps=10
)

# 输入数据
x = torch.randn(32, 60, 8)

# 前向传播
predictions, (h_n, c_n) = model(x)
print(predictions.shape)  # [32, 10, 8]
print(h_n.shape)          # [2, 32, 128]
print(c_n.shape)          # [2, 32, 128]
```

### 使用初始状态

```python
# 自定义初始状态
h_0 = torch.zeros(2, 32, 128)
c_0 = torch.zeros(2, 32, 128)
hidden_0 = (h_0, c_0)

# 前向传播
predictions, hidden_n = model(x, hidden_0)
```

### 多步预测

```python
# 预测未来30步
long_predictions = model.predict_multi_step(x, steps=30)
print(long_predictions.shape)  # [32, 30, 8]
```

## 高级架构（可选）

### LSTMEncoder-Decoder

```python
# 编码器
encoder = LSTMEncoder(input_size=8, hidden_size=128)
encoder_outputs, encoder_hidden = encoder(input_seq)

# 解码器
decoder = LSTMDecoder(hidden_size=128, output_size=8)
decoder_output, decoder_hidden = decoder(decoder_input, encoder_hidden)
```

**适用场景**：
- 序列到序列任务
- 变长输入和输出
- 需要注意力机制的场景

## 训练技巧

### 1. 梯度裁剪

LSTM虽然缓解了梯度消失，但仍可能梯度爆炸。

```python
# 裁剪梯度范数
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. 层归一化

对于深层LSTM，可以使用层归一化。

```python
class LSTMWithLayerNorm(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(...)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out, hidden
```

### 3. 变分Dropout

在时间步之间共享dropout mask。

```python
# PyTorch 1.x+
lstm = nn.LSTM(..., dropout=0.2)  # 自动使用变分dropout
```

## LSTM vs GRU 选择指南

### 使用LSTM的场景

✅ 序列较长（>100步）  
✅ 需要捕获长期依赖  
✅ 数据量充足  
✅ 对训练时间不敏感  
✅ GPU显存充足  

### 使用GRU的场景

✅ 序列较短（<100步）  
✅ 需要快速训练  
✅ 数据量有限  
✅ GPU显存受限  
✅ 快速原型验证  

### 性能对比（示例）

```
配置: hidden_size=128, num_layers=2

LSTM:
- 参数量: ~140K
- 训练速度: 100 batch/s
- 显存占用: ~500MB

GRU:
- 参数量: ~105K
- 训练速度: 130 batch/s
- 显存占用: ~380MB
```

## 超参数调优

### hidden_size

```
64:   快速，表达能力有限
128:  推荐，平衡性能和速度
256:  强大，需要更多数据和时间
512:  很强，容易过拟合
```

### num_layers

```
1:    简单模式，快速
2:    复杂模式，推荐
3:    很复杂，需要大量数据
4+:   深度网络，难以训练
```

### dropout

```
0.0:   无正则化
0.1:   轻度正则化
0.2:   标准正则化（推荐）
0.3:   强正则化
0.5+:  可能欠拟合
```

## 常见问题

**Q: LSTM为什么有两个状态？**  
A: 细胞状态（c）是长期记忆，隐藏状态（h）是短期输出。这种设计使LSTM能更好地捕获长期依赖。

**Q: 何时需要多层LSTM？**  
A: 当单层无法捕获数据的复杂模式时。通常2层足够，3层以上需要大量数据。

**Q: LSTM一定比GRU好吗？**  
A: 不一定。对于较短序列，GRU可能表现相当甚至更好，且训练更快。

**Q: 如何处理变长序列？**  
A: 使用`pack_padded_sequence`和`pad_packed_sequence`，或使用固定窗口切分。

**Q: LSTM可以并行训练吗？**  
A: 同一序列的时间步不能并行，但批次内的不同序列可以并行。

# model_gru.py 解释文档

## 概述

`model_gru.py` 实现了基于GRU（门控循环单元）的时间序列预测模型。GRU是LSTM的简化版本，参数更少，训练速度更快。

## GRU原理简介

### 门控机制

GRU使用两个门来控制信息流：

1. **重置门 (Reset Gate)**：控制如何将新输入与历史记忆结合
2. **更新门 (Update Gate)**：控制历史信息的保留程度

### 数学公式

```
rₜ = σ(Wr·[hₜ₋₁, xₜ])           # 重置门
zₜ = σ(Wz·[hₜ₋₁, xₜ])           # 更新门
h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])   # 候选隐藏状态
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ  # 新隐藏状态
```

其中：
- σ: sigmoid函数
- ⊙: 逐元素乘法
- hₜ: 隐藏状态
- xₜ: 当前输入

## 模型架构

### GRUModel类

```
输入 [batch, seq_len, 8]
    ↓
GRU层 (多层可选)
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
GRUModel(
    input_size=8,        # 输入特征数（8个电压通道）
    hidden_size=128,     # 隐藏层大小
    num_layers=1,        # GRU层数
    output_size=8,       # 输出特征数
    predict_steps=10,    # 预测步数
    dropout=0.2          # Dropout比率
)
```

## 核心方法详解

### 1. `__init__()`

初始化模型组件。

```python
# GRU层
self.gru = nn.GRU(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    batch_first=True,  # 输入形状: [batch, seq, feature]
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
- `hidden`: [num_layers, batch_size, hidden_size] - 初始隐藏状态（可选）

**输出**：
- `output`: [batch_size, predict_steps, 8] - 预测结果
- `hidden`: [num_layers, batch_size, hidden_size] - 最终隐藏状态

**维度变化示意图**：

```
输入 x: [32, 60, 8]
    ↓
GRU处理
    ↓
gru_out: [32, 60, 128]  # 每个时间步的输出
hidden: [1, 32, 128]    # 最终隐藏状态
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
# GRU前向传播
gru_out, hidden = self.gru(x, hidden)  # [batch, seq, hidden], [layers, batch, hidden]

# 取最后一个时间步
last_output = gru_out[:, -1, :]  # [batch, hidden]

# Dropout
last_output = self.dropout_layer(last_output)

# 全连接层预测
fc_out = self.fc(last_output)  # [batch, predict_steps * output_size]

# 重塑为正确的输出形状
output = fc_out.view(batch_size, self.predict_steps, self.output_size)
```

### 3. `_init_hidden(batch_size, device)`

初始化隐藏状态为零。

```python
return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
```

### 4. `predict_multi_step(x, steps)`

多步迭代预测，用于长期预测。

**工作原理**：
1. 使用输入序列获取初始隐藏状态
2. 从最后一个时间步开始
3. 迭代预测每一步
4. 将预测结果作为下一步的输入

```python
# 获取初始隐藏状态
_, hidden = self.gru(x, hidden)

# 初始输入
current_input = x[:, -1:, :]  # [batch, 1, 8]

# 迭代预测
for _ in range(steps):
    _, hidden = self.gru(current_input, hidden)
    prediction = self.fc(hidden[-1])
    next_step = prediction[:, 0:1, :]
    all_predictions.append(next_step)
    current_input = next_step  # 更新输入
```

## 高级架构（可选）

### GRUEncoder-Decoder

编码器-解码器架构适合更复杂的序列到序列任务。

**GRUEncoder**：
```python
encoder = GRUEncoder(input_size=8, hidden_size=128)
outputs, hidden = encoder(x)
```

**GRUDecoder**：
```python
decoder = GRUDecoder(hidden_size=128, output_size=8)
output, hidden = decoder(x, hidden)
```

## 参数量分析

### 单层GRU

```
参数量 = 3 * (input_size + hidden_size + 1) * hidden_size
       = 3 * (8 + 128 + 1) * 128
       = 52,608
```

GRU有3组参数（重置门、更新门、候选状态），每组包含：
- 输入权重：input_size × hidden_size
- 隐藏权重：hidden_size × hidden_size
- 偏置：hidden_size

### 完整模型（示例配置）

```python
GRUModel(input_size=8, hidden_size=128, num_layers=1)

总参数量：
- GRU层：52,608
- 全连接层：128 * 80 + 80 = 10,320
- 总计：62,928
```

## 使用示例

### 基本使用

```python
from model_gru import GRUModel
import torch

# 创建模型
model = GRUModel(
    input_size=8,
    hidden_size=128,
    num_layers=1,
    predict_steps=10
)

# 输入数据
x = torch.randn(32, 60, 8)  # [batch, seq_len, input_size]

# 前向传播
predictions, hidden = model(x)
print(predictions.shape)  # [32, 10, 8]
```

### 训练模式

```python
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for batch_x, batch_y in train_loader:
    # 前向传播
    predictions, _ = model(batch_x)
    
    # 计算损失
    loss = criterion(predictions, batch_y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 推理模式

```python
model.eval()
with torch.no_grad():
    predictions, _ = model(test_input)
```

### 多步预测

```python
# 预测未来20步
long_predictions = model.predict_multi_step(x, steps=20)
print(long_predictions.shape)  # [32, 20, 8]
```

## GRU vs LSTM

| 特性 | GRU | LSTM |
|------|-----|------|
| 门的数量 | 2个 | 3个 |
| 状态 | 隐藏状态 | 隐藏状态 + 细胞状态 |
| 参数量 | 较少 | 较多 |
| 训练速度 | 更快 | 较慢 |
| 记忆能力 | 良好 | 更强 |
| 适用场景 | 中短序列 | 长序列 |

**何时选择GRU？**
- 数据量较小
- 序列较短（<100步）
- 需要快速训练
- GPU显存有限

**何时选择LSTM？**
- 数据量充足
- 序列较长
- 需要更强的记忆能力
- 对训练时间不敏感

## 超参数调优建议

### hidden_size（隐藏层大小）

- **64**: 快速原型，5分钟内训练完成
- **128**: 平衡性能和速度，推荐配置
- **256**: 更强的表达能力，需要更多时间

### num_layers（层数）

- **1**: 最快，适合简单模式
- **2**: 增强表达能力，适合复杂模式
- **3+**: 容易过拟合，不推荐

### dropout

- **0.0**: 无正则化
- **0.1-0.2**: 轻度正则化，推荐
- **0.3-0.5**: 强正则化，防止严重过拟合

## 常见问题

**Q: GRU比LSTM快多少？**  
A: 通常快20-30%，因为参数量少25%左右。

**Q: 为什么只使用最后一个时间步的输出？**  
A: 最后一个时间步包含了整个序列的信息摘要，足以用于预测。

**Q: 可以使用所有时间步的输出吗？**  
A: 可以，但需要修改全连接层的输入维度，通常用于序列标注任务。

**Q: 如何防止过拟合？**  
A: 使用dropout、early stopping、减小模型容量、增加训练数据。

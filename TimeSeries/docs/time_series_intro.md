# 时间序列基础 & LSTM/GRU 入门

## 1. 时间序列数据概述

### 1.1 什么是时间序列数据？

时间序列数据是按时间顺序排列的一系列数据点。在我们的热电芯片辐射采集项目中，每10秒采集一次8个通道的电压值，这些数据随时间变化，形成了时间序列。

**特点：**
- 时间依赖性：前后数据点之间存在关联
- 顺序性：数据的顺序不能随意改变
- 连续性：在同一测量段内，数据是连续采集的

### 1.2 不等间隔片段组成的数据集

在实际实验中，我们的数据具有特殊性：

```
11月22日: 12:00~17:00 (连续片段1)
         [缺失7天]
11月29日: 15:00~16:00 (连续片段2)
         [缺失6天]
12月05日: 13:00~16:00 (连续片段3)
```

**关键特性：**
- 每天的数据是一个独立的连续片段（Segment）
- 不同日期之间的数据完全不连续
- 不能通过插值连接不同日期的数据
- 每个片段内部的采样间隔固定（约10秒）

## 2. 片段式时间序列（Segment-based Sequence Modeling）

### 2.1 核心概念

片段式时间序列是指由多个独立的时间序列片段组成的数据集，每个片段在时间上是连续的，但片段之间存在时间间隔。

**在我们的项目中：**
- 每个片段 = 一天的测量数据
- 片段内：时间连续，可以建立时间依赖关系
- 片段间：时间不连续，不应建立依赖关系

### 2.2 数据表示

```
片段1 (11月22日): 
  形状: [N1, 8]  # N1 = 该天的采样点数, 8 = 8个通道
  
片段2 (11月29日):
  形状: [N2, 8]  # N2 可能与 N1 不同
  
片段3 (12月05日):
  形状: [N3, 8]
```

### 2.3 训练策略

在训练时，我们需要：
1. 将每个片段视为独立的序列
2. 在每个片段内进行序列建模
3. 不跨片段传递隐藏状态
4. 可以从不同片段中采样训练样本

## 3. RNN/LSTM/GRU 核心思想

### 3.1 循环神经网络 (RNN) 基础

RNN 是专门处理序列数据的神经网络，其核心思想是：
- 维护一个隐藏状态（hidden state），捕获历史信息
- 在处理每个时间步时，同时考虑当前输入和历史信息

**基本结构：**
```
输入序列: x₁, x₂, x₃, ..., xₜ
隐藏状态: h₁, h₂, h₃, ..., hₜ

hₜ = f(xₜ, hₜ₋₁)
```

**缺点：**
- 梯度消失/爆炸问题
- 难以捕获长期依赖关系

### 3.2 长短期记忆网络 (LSTM)

LSTM 通过引入门控机制解决了 RNN 的梯度消失问题。

**核心组件：**
1. **遗忘门 (Forget Gate)**：决定丢弃哪些历史信息
2. **输入门 (Input Gate)**：决定保存哪些新信息
3. **输出门 (Output Gate)**：决定输出哪些信息

**维度变化示意：**
```
输入: xₜ [batch_size, input_size]
隐藏状态: hₜ [batch_size, hidden_size]
细胞状态: cₜ [batch_size, hidden_size]

LSTM内部计算:
  遗忘门: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
  输入门: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
  输出门: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
  
  候选值: c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
  细胞状态: cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ
  隐藏状态: hₜ = oₜ ⊙ tanh(cₜ)
```

**优点：**
- 能够学习长期依赖关系
- 通过门控机制有效防止梯度消失
- 适合处理较长的序列

**适用场景：**
- 需要记住长期历史信息的任务
- 序列较长的时间序列预测
- 复杂的时序模式识别

### 3.3 门控循环单元 (GRU)

GRU 是 LSTM 的简化版本，参数更少，训练更快。

**核心组件：**
1. **重置门 (Reset Gate)**：控制如何将新输入与历史记忆结合
2. **更新门 (Update Gate)**：控制历史信息的保留程度

**维度变化示意：**
```
输入: xₜ [batch_size, input_size]
隐藏状态: hₜ [batch_size, hidden_size]

GRU内部计算:
  重置门: rₜ = σ(Wr·[hₜ₋₁, xₜ])
  更新门: zₜ = σ(Wz·[hₜ₋₁, xₜ])
  
  候选隐藏状态: h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
  新隐藏状态: hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

**优点：**
- 参数比 LSTM 少（无细胞状态）
- 训练速度更快
- 性能与 LSTM 相当

**对比：**
| 特性 | LSTM | GRU |
|------|------|-----|
| 参数量 | 较多 | 较少 |
| 训练速度 | 较慢 | 较快 |
| 记忆能力 | 强 | 较强 |
| 适用场景 | 长序列、复杂模式 | 中短序列、快速原型 |

## 4. 输入输出维度详解

### 4.1 单层 LSTM/GRU

**PyTorch 中的标准用法：**

```python
# 定义
lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=1, batch_first=True)
gru = nn.GRU(input_size=8, hidden_size=64, num_layers=1, batch_first=True)

# 输入
input: [batch_size, seq_len, input_size]
例如: [32, 100, 8]  # 32个样本，每个100个时间步，每步8个特征

# 输出
output: [batch_size, seq_len, hidden_size]
例如: [32, 100, 64]  # 每个时间步的输出

h_n: [num_layers, batch_size, hidden_size]
例如: [1, 32, 64]  # 最后一个时间步的隐藏状态

# LSTM 还额外有细胞状态
c_n: [num_layers, batch_size, hidden_size]
```

### 4.2 多层 LSTM/GRU

```python
# 2层LSTM
lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, batch_first=True)

# 隐藏状态形状
h_n: [2, batch_size, 64]  # 两层，每层的最终隐藏状态
c_n: [2, batch_size, 64]  # 两层，每层的最终细胞状态
```

### 4.3 在我们项目中的应用

**数据维度：**
```
原始数据: 
  - 8个通道的电压值
  - 时间序列长度取决于测量时长

模型输入:
  input_size = 8  # 8个电压通道
  
模型配置:
  hidden_size = 64/128/256  # 隐藏层大小（可调）
  num_layers = 1/2/3        # 层数（可调）
  
预测输出:
  output_size = 8  # 预测未来的8个电压值
```

## 5. 不插值条件下构建模型的注意事项

### 5.1 为什么不插值？

在我们的实验中，不同日期之间的数据来自完全不同的实验条件：
- 天气条件可能不同
- 太阳辐射强度不同
- 测量时间段不同

**插值的问题：**
- 人为创造不存在的数据点
- 破坏数据的真实性
- 可能引入错误的时间依赖关系

### 5.2 正确的处理方式

**1. 片段独立处理**
```python
# 错误做法：将所有数据连接成一个长序列
all_data = np.concatenate([day1_data, day2_data, day3_data])  # ❌

# 正确做法：保持片段独立
segments = [day1_data, day2_data, day3_data]  # ✓
```

**2. 重置隐藏状态**
```python
# 在每个新片段开始时
for segment in segments:
    h_0 = torch.zeros(num_layers, batch_size, hidden_size)  # 重置隐藏状态
    output, h_n = model(segment, h_0)
```

**3. 不跨片段预测**
```python
# 错误：从day1的末尾预测day2的开始
prediction = model(day1_end)  # 预测day2_start  ❌

# 正确：只在片段内部预测
prediction = model(day1_part1)  # 预测day1_part2  ✓
```

### 5.3 训练数据准备

**滑动窗口方法（在单个片段内）：**
```python
# 对于一个片段 [seq_len, 8]
window_size = 50
predict_steps = 10

for i in range(len(segment) - window_size - predict_steps):
    # 输入：前50个时间步
    x = segment[i:i+window_size]  # [50, 8]
    
    # 目标：后10个时间步
    y = segment[i+window_size:i+window_size+predict_steps]  # [10, 8]
```

## 6. 实验数据打包成 Batch

### 6.1 基本概念

Batch（批次）是将多个样本组合在一起同时训练，可以：
- 提高训练效率（GPU 并行计算）
- 稳定梯度更新
- 加速收敛

### 6.2 固定长度序列的 Batch

**最简单的情况（所有序列等长）：**
```python
# 假设我们有32个长度为100的序列
sequences = [seq1, seq2, ..., seq32]  # 每个seq: [100, 8]

# 直接堆叠成batch
batch = torch.stack(sequences, dim=0)  # [32, 100, 8]
```

### 6.3 变长序列的 Batch

由于不同天的测量时长不同，序列长度可能不一样。

**方法1：填充 (Padding)**
```python
from torch.nn.utils.rnn import pad_sequence

sequences = [seq1, seq2, seq3]  # 长度分别为 [100, 8], [150, 8], [80, 8]

# 填充到最长序列的长度
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
# 输出: [3, 150, 8]  # 较短的序列用0填充
```

**方法2：固定窗口切分**
```python
# 从每个片段中提取固定长度的子序列
window_size = 100

def extract_windows(segment):
    windows = []
    for i in range(len(segment) - window_size):
        windows.append(segment[i:i+window_size])
    return windows

# 这样所有窗口都是固定长度
all_windows = []
for segment in segments:
    all_windows.extend(extract_windows(segment))

# 组成batch
batch = torch.stack(all_windows[:32])  # [32, 100, 8]
```

### 6.4 使用 PyTorch DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, segments, window_size=100, predict_steps=10):
        self.samples = []
        for segment in segments:
            # 从每个片段提取样本
            for i in range(len(segment) - window_size - predict_steps):
                x = segment[i:i+window_size]
                y = segment[i+window_size:i+window_size+predict_steps]
                self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# 创建DataLoader
dataset = TimeSeriesDataset(segments)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用
for batch_x, batch_y in dataloader:
    # batch_x: [32, 100, 8]
    # batch_y: [32, 10, 8]
    predictions = model(batch_x)
    loss = criterion(predictions, batch_y)
```

## 7. 训练流程总结

### 7.1 完整流程

```
1. 数据加载
   ├── 读取多个CSV文件
   ├── 按日期划分片段
   └── 标准化/归一化

2. 数据准备
   ├── 滑动窗口提取样本
   ├── 划分训练集/验证集
   └── 创建DataLoader

3. 模型训练
   ├── 初始化模型
   ├── 定义损失函数和优化器
   ├── 训练循环
   │   ├── 前向传播
   │   ├── 计算损失
   │   ├── 反向传播
   │   └── 更新参数
   └── 验证评估

4. 模型预测
   ├── 加载训练好的模型
   ├── 输入测试序列
   └── 生成预测结果
```

### 7.2 关键参数设置（30分钟内训练完成）

**数据参数：**
- `window_size`: 50-100 （输入序列长度）
- `predict_steps`: 5-20 （预测步数）
- `batch_size`: 32-128 （根据GPU显存调整）

**模型参数：**
- `input_size`: 8 （固定）
- `hidden_size`: 64-256 （越大越慢）
- `num_layers`: 1-2 （层数太多会很慢）
- `dropout`: 0.1-0.3 （防止过拟合）

**训练参数：**
- `learning_rate`: 0.001-0.01
- `num_epochs`: 50-200 （根据数据量调整）
- `early_stopping`: 验证集10轮不改善则停止

**推荐配置（15分钟内训练）：**
```python
config = {
    'window_size': 60,
    'predict_steps': 10,
    'batch_size': 64,
    'hidden_size': 128,
    'num_layers': 1,
    'learning_rate': 0.001,
    'num_epochs': 100
}
```

## 8. 总结

### 8.1 核心要点

1. **片段式时间序列**：每天的数据是独立片段，不跨日期建模
2. **不插值原则**：保持数据真实性，不人为补点
3. **LSTM vs GRU**：LSTM记忆能力更强，GRU训练更快
4. **维度理解**：清楚输入输出的形状变化
5. **批次处理**：使用固定窗口或填充处理变长序列

### 8.2 实践建议

1. **从简单开始**：先用GRU + 小模型快速验证
2. **逐步调优**：观察训练曲线，调整超参数
3. **GPU加速**：确保使用CUDA加速训练
4. **监控过拟合**：使用验证集和dropout
5. **保存检查点**：定期保存模型，防止训练中断

### 8.3 常见问题

**Q: 序列太长导致显存不足？**  
A: 减小batch_size或window_size

**Q: 训练速度太慢？**  
A: 减小hidden_size或num_layers，优先使用GRU

**Q: 模型不收敛？**  
A: 检查数据归一化，降低学习率，增加训练轮数

**Q: 如何评估模型？**  
A: 使用MSE、MAE等指标，观察预测曲线与真实值的拟合程度

---

**参考资源：**
- PyTorch 官方文档: https://pytorch.org/docs/stable/nn.html#recurrent-layers
- Understanding LSTM Networks: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Time Series Forecasting with PyTorch: 各种在线教程和示例

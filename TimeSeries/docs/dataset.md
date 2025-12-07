# dataset.py 解释文档

## 概述

`dataset.py` 是数据加载和预处理模块，负责将热电芯片测量的CSV文件转换为PyTorch模型可以使用的格式。

## 核心功能

1. **加载多个CSV文件**：每个CSV文件代表一天的独立测量数据
2. **数据标准化**：对电压值进行标准化处理，加速训练收敛
3. **滑动窗口采样**：从连续序列中提取固定长度的训练样本
4. **数据集划分**：将数据划分为训练集和验证集
5. **批次加载**：使用DataLoader实现高效的批次数据加载

## 类和函数说明

### ThermoelectricDataset类

这是核心的数据集类，继承自`torch.utils.data.Dataset`。

#### 初始化参数

```python
ThermoelectricDataset(
    data_dir,           # CSV文件所在目录
    window_size=60,     # 输入序列长度
    predict_steps=10,   # 预测步数
    stride=1,           # 滑动窗口步长
    normalize=True,     # 是否标准化
    train_ratio=0.8     # 训练集比例
)
```

#### 关键方法

**1. `_load_segments()`**

加载所有CSV文件，每个文件作为一个独立片段。

```python
# 读取CSV
df = pd.read_csv(csv_file)

# 提取8个电压通道
voltage_data = df[voltage_columns].values  # [seq_len, 8]

# 存储为片段
segments.append(voltage_data)
```

**维度变化**：
- 输入：CSV文件
- 输出：List of numpy arrays, 每个形状 `[seq_len, 8]`

**2. `_fit_scaler()`**

在所有数据上拟合标准化器（StandardScaler）。

```python
# 合并所有片段数据
all_data = np.vstack(self.segments)  # [total_samples, 8]

# 拟合标准化器
self.scaler = StandardScaler()
self.scaler.fit(all_data)
```

**标准化公式**：
```
normalized_value = (value - mean) / std
```

**3. `_extract_samples()`**

使用滑动窗口从每个片段提取训练样本。

```python
for segment in segments:
    for i in range(0, len(segment) - window_size - predict_steps + 1, stride):
        # 输入窗口
        x = segment[i:i + window_size]  # [window_size, 8]
        
        # 目标窗口
        y = segment[i + window_size:i + window_size + predict_steps]  # [predict_steps, 8]
        
        samples.append((x, y, segment_idx))
```

**滑动窗口示意图**：
```
时间序列：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]

window_size=5, predict_steps=2, stride=1

样本1: x=[0,1,2,3,4], y=[5,6]
样本2: x=[1,2,3,4,5], y=[6,7]
样本3: x=[2,3,4,5,6], y=[7,8]
...
```

**4. `__getitem__(idx)`**

PyTorch Dataset的标准方法，返回一个样本。

```python
def __getitem__(self, idx):
    x, y, segment_idx = self.samples[idx]
    x = torch.FloatTensor(x)  # [window_size, 8]
    y = torch.FloatTensor(y)  # [predict_steps, 8]
    return x, y
```

**5. `split_train_val()`**

按片段划分训练集和验证集，确保同一片段的数据不会同时出现在训练集和验证集中。

```python
# 随机打乱片段索引
segment_indices = np.random.permutation(num_segments)

# 前80%的片段用于训练
train_indices = segment_indices[:num_train_segments]

# 后20%的片段用于验证
val_indices = segment_indices[num_train_segments:]
```

**6. `inverse_transform(data)`**

将标准化后的数据转换回原始尺度。

```python
# 反标准化公式
original_value = normalized_value * std + mean
```

### create_dataloaders函数

便捷函数，一次性创建训练和验证DataLoader。

```python
train_loader, val_loader, dataset = create_dataloaders(
    data_dir='../TimeSeries',
    batch_size=32,
    window_size=60,
    predict_steps=10
)
```

**返回值**：
- `train_loader`: 训练数据加载器
- `val_loader`: 验证数据加载器
- `dataset`: 原始数据集对象（包含scaler等信息）

## 数据流程图

```
CSV文件 (多个)
    ↓
加载为片段 (list of arrays)
    ↓
标准化处理
    ↓
滑动窗口提取样本 (list of (x, y) pairs)
    ↓
划分训练/验证集
    ↓
DataLoader (批次迭代器)
    ↓
模型训练
```

## 维度变化详解

### 单个样本

```
CSV原始数据:
  [N_samples, 10] (Timestamp, DateTime, 8个电压)

提取电压:
  [N_samples, 8]

滑动窗口采样:
  输入 x: [window_size, 8]      例如 [60, 8]
  目标 y: [predict_steps, 8]    例如 [10, 8]
```

### 批次数据

```
DataLoader输出:
  batch_x: [batch_size, window_size, 8]      例如 [32, 60, 8]
  batch_y: [batch_size, predict_steps, 8]    例如 [32, 10, 8]
```

## 使用示例

### 基本使用

```python
from dataset import ThermoelectricDataset, create_dataloaders

# 方法1：直接创建数据集
dataset = ThermoelectricDataset(
    data_dir='../TimeSeries',
    window_size=60,
    predict_steps=10,
    normalize=True
)

# 获取一个样本
x, y = dataset[0]
print(x.shape)  # [60, 8]
print(y.shape)  # [10, 8]

# 方法2：使用便捷函数创建DataLoader
train_loader, val_loader, dataset = create_dataloaders(
    data_dir='../TimeSeries',
    batch_size=32,
    window_size=60,
    predict_steps=10
)

# 遍历批次
for batch_x, batch_y in train_loader:
    print(batch_x.shape)  # [32, 60, 8]
    print(batch_y.shape)  # [32, 10, 8]
    break
```

### 保存和加载标准化器

```python
# 保存
dataset.save_scaler('scaler.pkl')

# 加载
dataset.load_scaler('scaler.pkl')

# 反标准化
original_data = dataset.inverse_transform(normalized_data)
```

## 重要注意事项

### 1. 片段独立性

不同日期的数据片段是完全独立的，不应该跨日期连接数据。

```python
# ❌ 错误做法
all_data = np.concatenate([day1, day2, day3])

# ✓ 正确做法
segments = [day1, day2, day3]
# 从每个片段独立提取样本
```

### 2. 滑动窗口步长

`stride` 参数控制滑动窗口的步长：
- `stride=1`: 最密集的采样，样本数最多，训练时间最长
- `stride=5`: 适中的采样，平衡样本数和训练时间
- `stride=10`: 稀疏采样，样本数少，训练快但可能欠拟合

### 3. 数据标准化

标准化对于神经网络训练非常重要：
- 加速收敛
- 避免梯度爆炸/消失
- 使不同通道的数值范围一致

**重要**：预测时必须使用训练时的scaler进行反标准化！

### 4. 训练/验证集划分

按片段划分而不是按样本划分，确保：
- 验证集的片段完全独立
- 更真实地评估模型的泛化能力

## 性能优化建议

1. **增加stride**：如果训练时间过长，可以增加stride减少样本数
2. **减小window_size**：较小的窗口可以减少计算量
3. **使用num_workers**：DataLoader可以使用多进程加速数据加载
4. **pin_memory**：如果使用GPU，设置`pin_memory=True`加速数据传输

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # 使用4个进程
    pin_memory=True     # GPU加速
)
```

## 常见问题

**Q: 为什么不同CSV文件的数据点数量不同？**  
A: 每天的测量时长不同，导致数据点数量不同。这是正常的，代码已处理。

**Q: 如何添加新的数据文件？**  
A: 直接将新的CSV文件放到data_dir目录下，保持相同的列名格式即可。

**Q: 标准化器的统计量是在所有数据上计算的吗？**  
A: 是的，虽然片段不连续，但我们希望所有数据使用统一的标准化参数。

**Q: 可以不标准化吗？**  
A: 可以设置`normalize=False`，但不推荐，可能导致训练不稳定。

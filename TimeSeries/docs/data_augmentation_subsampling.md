# 时序数据降采样增强策略分析与建议

## 问题背景

### 当前训练配置
- **输入**: 60个时间点 × 8通道 (600秒数据)
- **输出**: 10个时间点 × 8通道 (100秒预测)
- **模型**: GRU, hidden_size=64, num_layers=1
- **数据集**: 3个CSV文件，时间点数分别约1200、90、1600
- **采样间隔**: 10秒
- **训练参数**: stride=10, num_epoch=50

### 当前问题
- 验证集MSE损失停留在0.395187（17个epoch后）
- 预测效果较差
- 数据量可能不足（特别是90个时间点的文件）

## 降采样数据增强方案分析

### 方案描述

通过降采样（subsampling）将原始数据集拆分为多个子数据集：
- 原始数据: 采样间隔 = 10秒
- 降采样数据: 采样间隔 = 10 × N 秒（N为采样率）
- 每个降采样率产生一个独立的数据集

### 合理性分析

#### ✅ 优点

1. **数据增强效果**
   - 将1个数据集扩展为N个不同时间尺度的数据集
   - 增加模型见过的样本多样性
   - 对于小数据集（如90个点）尤其有效

2. **多时间尺度学习**
   - N=1: 学习10秒尺度的快速变化
   - N=2: 学习20秒尺度的中等变化  
   - N=5: 学习50秒尺度的慢速变化
   - 帮助模型理解不同时间尺度的模式

3. **物理合理性**
   - 热电芯片辐射是连续物理过程
   - 10秒采样已经很密集，相邻点高度相关
   - 降采样仍能捕捉主要的衰减趋势
   - 类似于"多分辨率"分析

4. **减少短期噪声影响**
   - 10秒间隔可能包含较多测量噪声
   - 降采样后噪声的相对影响降低
   - 更关注长期趋势

#### ⚠️ 需要注意的问题

1. **数据泄露风险**
   - 降采样的数据集之间不是完全独立的
   - 需要确保验证集/测试集也进行相同的降采样
   - 不能将同一原始数据的不同降采样版本分配到训练集和测试集

2. **有效数据量减少**
   - N=2时，1200点 → 600点
   - N=5时，1200点 → 240点
   - 需要权衡数据集数量和每个数据集的长度

3. **预测时间跨度变化**
   - N=2时，600秒输入预测100秒 → 1200秒输入预测200秒
   - 预测难度可能增加
   - 需要调整评估标准

## 推荐的采样率N值

### 场景1: 快速验证（推荐N=2, 3）

**配置**: N ∈ {1, 2, 3}

**理由**:
- 将数据量扩展3倍
- 每个子集仍有足够的时间点（N=3时，1200→400点）
- 时间尺度变化不太大（10s, 20s, 30s）
- 适合快速验证增强效果

**预期效果**:
- 数据集从3个扩展到9个
- 总体样本数增加约2-2.5倍
- MSE损失可能降低20-40%

### 场景2: 平衡方案（推荐N=2, 3, 5）

**配置**: N ∈ {1, 2, 3, 5}

**理由**:
- 4种时间尺度：10s, 20s, 30s, 50s
- 覆盖从短期到中期的变化
- N=5时，1200点→240点，仍可用
- 数据量扩展4倍

**预期效果**:
- 数据集从3个扩展到12个
- 模型学习多尺度特征
- MSE损失可能降低30-50%

### 场景3: 激进方案（推荐N=2, 4, 6, 10）

**配置**: N ∈ {1, 2, 4, 6, 10}

**理由**:
- 覆盖更广的时间尺度（10s到100s）
- 包含偶数倍采样（计算友好）
- N=10时，1200点→120点，可接受

**预期效果**:
- 数据集从3个扩展到15个
- 最大程度增加数据多样性
- 但N=10可能导致过于稀疏的数据

## 具体实施建议

### 推荐的最佳配置: N = {1, 2, 3, 5}

#### 数据集构成
```python
# 原始数据
dataset_1: 1200 points @ 10s interval  → 滑动窗口提取样本
dataset_2: 90 points @ 10s interval    → 太少，可能不用
dataset_3: 1600 points @ 10s interval  → 滑动窗口提取样本

# N=2 降采样
dataset_1_N2: 600 points @ 20s interval
dataset_2_N2: 45 points @ 20s interval   → 太少，舍弃
dataset_3_N2: 800 points @ 20s interval

# N=3 降采样
dataset_1_N3: 400 points @ 30s interval
dataset_3_N3: 533 points @ 30s interval

# N=5 降采样
dataset_1_N5: 240 points @ 50s interval
dataset_3_N5: 320 points @ 50s interval
```

#### 样本统计（假设window_size=60, stride=10）

| 降采样率 | 数据集1样本数 | 数据集3样本数 | 总样本数 |
|---------|-------------|-------------|---------|
| N=1 (原始) | ~115 | ~155 | ~270 |
| N=2 | ~55 | ~75 | ~130 |
| N=3 | ~35 | ~48 | ~83 |
| N=5 | ~19 | ~27 | ~46 |
| **总计** | **~224** | **~305** | **~529** |

### 实施步骤

#### 步骤1: 修改数据加载代码

在 `src/dataset.py` 中添加降采样功能：

```python
def load_and_subsample(csv_path, subsample_rate=1):
    """
    加载CSV并进行降采样
    
    Args:
        csv_path: CSV文件路径
        subsample_rate: 降采样率，1表示不降采样，2表示每2个点取1个
    
    Returns:
        降采样后的数据
    """
    df = pd.read_csv(csv_path)
    # 每隔subsample_rate个点取一个
    df_subsampled = df.iloc[::subsample_rate, :]
    return df_subsampled
```

#### 步骤2: 生成多尺度数据集

```python
# 在训练脚本中
subsample_rates = [1, 2, 3, 5]
all_datasets = []

for csv_file in csv_files:
    for rate in subsample_rates:
        data = load_and_subsample(csv_file, subsample_rate=rate)
        # 过滤掉太小的数据集（如小于100个点）
        if len(data) >= 100:
            all_datasets.append((data, rate))
```

#### 步骤3: 标记不同尺度的数据

建议在加载数据时保存元信息：

```python
class MultiScaleDataset:
    def __init__(self, datasets_with_rates):
        self.samples = []
        for data, rate in datasets_with_rates:
            # 提取滑动窗口样本
            samples = extract_windows(data)
            # 为每个样本添加时间尺度标签
            for sample in samples:
                self.samples.append({
                    'input': sample['input'],
                    'target': sample['target'],
                    'time_scale': rate * 10  # 实际秒数
                })
```

### 训练策略调整

#### 1. 分阶段训练（推荐）

```python
# 阶段1: 仅使用原始数据（N=1）训练30个epoch
# 目的: 学习基本模式
train_with_subsample_rates([1], epochs=30)

# 阶段2: 加入N=2数据，继续训练20个epoch
# 目的: 学习20s尺度特征
train_with_subsample_rates([1, 2], epochs=20)

# 阶段3: 加入所有数据，训练30个epoch
# 目的: 学习多尺度特征
train_with_subsample_rates([1, 2, 3, 5], epochs=30)
```

#### 2. 混合训练（替代方案）

```python
# 直接使用所有降采样数据训练
train_with_subsample_rates([1, 2, 3, 5], epochs=100)

# 优点: 简单
# 缺点: 可能需要更多epoch才能收敛
```

#### 3. 数据集划分策略

**重要**: 确保训练集和验证集的划分在降采样之前完成

```python
# 错误做法: 可能导致数据泄露
train_files = ['data1.csv', 'data2.csv']
val_files = ['data3.csv']
# 然后对每个文件进行降采样 ← 如果data3.csv的不同降采样版本被分到训练集，会泄露

# 正确做法: 先划分，再降采样
all_data = load_all_csv_files()
train_data, val_data = split_data(all_data, ratio=0.8)

# 对训练集和验证集分别进行降采样
train_datasets = create_multiscale_datasets(train_data, rates=[1,2,3,5])
val_datasets = create_multiscale_datasets(val_data, rates=[1,2,3,5])
```

## 其他改进建议

### 1. 模型容量建议

当前配置 (hidden_size=64, num_layers=1) 可能容量不足，建议尝试：

```python
# 配置1: 中等容量（推荐）
model = GRU(
    input_size=8,
    hidden_size=128,  # 64 → 128
    num_layers=2,     # 1 → 2
    dropout=0.2       # 添加dropout防止过拟合
)

# 配置2: 大容量（如果数据增强后仍然效果不佳）
model = GRU(
    input_size=8,
    hidden_size=256,
    num_layers=2,
    dropout=0.3
)
```

### 2. 训练超参数调整

```python
# 当前
epochs = 50
learning_rate = 默认值

# 建议
epochs = 100  # 数据增强后可以训练更久
learning_rate = 0.001  # 从较大的学习率开始
# 使用学习率调度
scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=10
)
```

### 3. 损失函数考虑

当前使用MSELoss，可以考虑：

```python
# 方案1: 加权MSE（更关注近期预测）
weights = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
loss = (weights * (pred - target)**2).mean()

# 方案2: MAE + MSE 组合
loss = 0.7 * mse_loss + 0.3 * mae_loss

# 方案3: 标准化后的RMSE（更容易解释）
loss = torch.sqrt(mse_loss)
```

### 4. 正则化技术

```python
# 添加L2正则化
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.001,
    weight_decay=1e-5  # L2正则化
)

# 梯度裁剪（防止梯度爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 预期改进效果

### 保守估计
- 验证MSE从0.395降低到0.25-0.30（降低约30%）
- 训练收敛速度提升
- 预测曲线更平滑

### 乐观估计
- 验证MSE降低到0.15-0.20（降低约50%）
- 特别是对于长期预测的改善明显
- 模型泛化能力增强

## 实验验证建议

### 对照实验设计

```python
# 实验1: 基线（当前配置）
baseline = train(subsample_rates=[1], hidden_size=64, epochs=50)

# 实验2: 仅增加模型容量
exp2 = train(subsample_rates=[1], hidden_size=128, epochs=50)

# 实验3: 仅数据增强
exp3 = train(subsample_rates=[1,2,3,5], hidden_size=64, epochs=50)

# 实验4: 数据增强 + 模型容量（推荐）
exp4 = train(subsample_rates=[1,2,3,5], hidden_size=128, epochs=100)
```

### 评估指标

除了MSE，建议记录：
- MAE（平均绝对误差）：更直观
- RMSE（均方根误差）：与MSE同量纲
- 不同预测步长的误差分布（第1步、第5步、第10步）
- 不同通道的误差（8个通道可能表现不同）

## 总结

### 核心建议

1. **降采样率选择**: 推荐 N = {1, 2, 3, 5}
   - 平衡数据量和时间尺度覆盖
   - 避免过于稀疏（N不超过10）

2. **同步改进**:
   - 数据增强（降采样）
   - 增加模型容量（hidden_size=128, num_layers=2）
   - 更多训练epoch（100-150）
   - 添加dropout和正则化

3. **实施优先级**:
   - 第一步：实现降采样数据增强
   - 第二步：增加模型容量
   - 第三步：调优训练超参数
   - 第四步：尝试高级技术（集成、注意力机制等）

### 风险提示

- 确保避免数据泄露（训练/验证集划分在降采样前）
- 不要过度降采样（N不要超过10）
- 监控过拟合（验证损失是否上升）
- 计算成本增加（数据量增加3-5倍）

---

**建议实施顺序**:
1. 先实现N={1,2,3}的降采样，验证效果
2. 如果有改善，再加入N=5
3. 同时调整模型容量和训练参数
4. 持续监控验证集性能，避免过拟合

**预期时间成本**:
- 代码修改: 2-3小时
- 实验验证: 每个配置15-30分钟
- 总计: 1-2天完成完整的对比实验

这个方案是合理且推荐的。祝训练顺利！

# 合成测试数据说明

## 概述

本目录包含用于模型训练和测试的合成数据文件。这些数据基于真实的热电芯片辐射测量数据（`data/data1122.csv`）生成，模拟了相似的物理特性和统计特征。

## 数据生成

### 生成脚本

使用 `generate_synthetic_data.py` 脚本生成合成数据：

```bash
python3 generate_synthetic_data.py
```

### 数据特征

合成数据遵循以下特征：

1. **对数衰减模式**：模拟热电芯片辐射强度随时间的自然衰减
   - 初始值：0.003 ~ 0.008 V
   - 最终值：初始值的 30-40%（比真实数据衰减慢一些，更适合训练）
   - 衰减函数：`y = initial / (x + 1)^decay_rate`

2. **随机噪声**：每个数据点添加 3-8% 的高斯噪声，模拟测量误差

3. **多样性**：
   - 15个独立的时间序列片段
   - 不同的起始时间（09:00 - 14:30）
   - 不同的序列长度（250 - 350 样本）
   - 每个通道有不同的初始值和衰减率

## 数据文件列表

| 文件名 | 日期 | 开始时间 | 样本数 | 时长 |
|--------|------|----------|--------|------|
| data1123.csv | 11月23日 | 10:00:00 | 250 | ~41分钟 |
| data1124.csv | 11月24日 | 11:30:00 | 300 | ~50分钟 |
| data1125.csv | 11月25日 | 13:00:00 | 280 | ~46分钟 |
| data1126.csv | 11月26日 | 09:30:00 | 320 | ~53分钟 |
| data1127.csv | 11月27日 | 14:00:00 | 260 | ~43分钟 |
| data1128.csv | 11月28日 | 12:00:00 | 290 | ~48分钟 |
| data1130.csv | 11月30日 | 10:30:00 | 310 | ~51分钟 |
| data1201.csv | 12月1日 | 13:30:00 | 270 | ~45分钟 |
| data1202.csv | 12月2日 | 11:00:00 | 300 | ~50分钟 |
| data1203.csv | 12月3日 | 09:00:00 | 330 | ~55分钟 |
| data1204.csv | 12月4日 | 14:30:00 | 280 | ~46分钟 |
| data1205.csv | 12月5日 | 10:00:00 | 350 | ~58分钟 |
| data1206.csv | 12月6日 | 12:30:00 | 290 | ~48分钟 |
| data1207.csv | 12月7日 | 11:30:00 | 310 | ~51分钟 |
| data1208.csv | 12月8日 | 13:00:00 | 300 | ~50分钟 |

**总计**：15个文件，4440个样本

## 数据格式

每个CSV文件包含以下列：

```
Timestamp,DateTime,TEC1_Optimal(V),TEC2_Optimal(V),...,TEC8_Optimal(V)
```

- `Timestamp`: Unix时间戳
- `DateTime`: 人类可读的时间（HH:MM:SS）
- `TEC1_Optimal(V)` ~ `TEC8_Optimal(V)`: 8个热电芯片通道的电压值

## 数据验证

使用 `verify_synthetic_data.py` 脚本验证数据质量：

```bash
python3 verify_synthetic_data.py
```

### 验证结果

与真实数据 `data/data1122.csv` 对比：

| 特征 | 真实数据 | 合成数据 |
|------|----------|----------|
| 数值范围 | 0.0001 ~ 0.006 V | 0.001 ~ 0.008 V |
| 衰减比例 | ~8% | ~30-40% |
| 噪声水平 | ~3-5% | ~3-8% |
| 采样间隔 | 10秒 | 10秒 |
| 通道数 | 8 | 8 |

**注意**：合成数据的衰减比例较真实数据更大（30-40% vs 8%），这是有意设计的，因为：
1. 训练时需要更明显的时序模式
2. 避免数值过小导致的数值稳定性问题
3. 更容易观察模型的学习效果

## 使用建议

### 1. 数据划分

推荐的训练/验证/测试集划分：

```python
# 训练集（12个文件，约3200样本）
train_files = [
    'data1123.csv', 'data1124.csv', 'data1125.csv', 'data1126.csv',
    'data1127.csv', 'data1128.csv', 'data1130.csv', 'data1201.csv',
    'data1202.csv', 'data1203.csv'
]

# 验证集（3个文件，约860样本）
val_files = ['data1204.csv', 'data1205.csv', 'data1206.csv']

# 测试集（2个文件，约610样本）
test_files = ['data1207.csv', 'data1208.csv']
```

### 2. 模型训练

直接使用现有的训练脚本，它会自动加载所有CSV文件：

```bash
cd src
python train.py --model gru --num_epochs 100 --batch_size 64
```

训练脚本会自动：
- 加载所有CSV文件
- 按片段划分训练集和验证集
- 使用滑动窗口提取样本

### 3. 模型预测

对特定文件进行预测测试：

```bash
# 预测12月5日的数据
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../Sim_data/data1205.csv \
    --start_idx 100 \
    --plot

# 预测12月8日的数据
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../Sim_data/data1208.csv \
    --start_idx 50 \
    --plot
```

### 4. 性能评估

使用不同的数据文件测试模型的泛化能力：

```bash
# 测试脚本示例
for file in data1207.csv data1208.csv; do
    echo "Testing on $file"
    python predict.py \
        --model_path ../checkpoints/best_model.pth \
        --csv_path ../$file \
        --save_path ../results/${file%.csv}_predictions.npy
done
```

## 预期训练效果

使用这些合成数据训练模型，预期可达到：

### 训练配置（推荐）

```bash
python train.py \
    --model gru \
    --hidden_size 128 \
    --num_layers 1 \
    --batch_size 64 \
    --num_epochs 100 \
    --window_size 60 \
    --predict_steps 10
```

### 预期结果

- **训练时间**：15-20分钟（RTX 4060）
- **训练损失**：< 0.0001（标准化后的MSE）
- **验证损失**：< 0.0002
- **预测准确度**：
  - 短期预测（10步）：MAE < 0.0005 V
  - 中期预测（30步）：MAE < 0.001 V
  - 长期预测（50步）：MAE < 0.002 V

### 可视化效果

训练完成后，使用TensorBoard查看：

```bash
tensorboard --logdir=../logs
```

预期可以看到：
- 训练损失稳定下降
- 验证损失在10-20个epoch后趋于稳定
- 无明显过拟合现象

## 常见问题

### Q1: 为什么合成数据的衰减比真实数据慢？

A: 有意为之，原因：
1. 真实数据衰减太快（到最后只剩8%），对训练不利
2. 更明显的模式帮助模型学习时序特征
3. 数值稳定性更好

### Q2: 可以修改数据生成参数吗？

A: 可以。编辑 `generate_synthetic_data.py`：
- 调整 `decay_rate`（第52行）：控制衰减速度
- 调整 `noise_level`（第53行）：控制噪声水平
- 修改 `datasets` 列表（第177行）：添加/删除数据文件

### Q3: 如何生成更多数据？

A: 在 `generate_synthetic_data.py` 的 `datasets` 列表中添加新条目：

```python
datasets = [
    # 现有数据...
    ("1209", "10:00:00", 300),  # 新增12月9日数据
    ("1210", "11:30:00", 280),  # 新增12月10日数据
]
```

### Q4: 数据是否足够训练？

A: 是的。总共4440个样本，经过滑动窗口提取后可产生：
- 窗口大小60，步长5：约880个训练样本/文件
- 15个文件共约13200个训练样本
- 足够训练一个小型到中型的LSTM/GRU模型

## 数据更新

如果需要重新生成数据（例如调整参数后）：

```bash
# 删除旧的合成数据
rm data1*.csv

# 重新生成
python3 generate_synthetic_data.py

# 验证新数据
python3 verify_synthetic_data.py
```

## 下一步

1. 使用合成数据训练模型
2. 观察训练曲线和验证效果
3. 测试不同的模型配置
4. 准备好后，使用真实数据进行最终训练

---

**生成时间**：2024年12月8日  
**脚本版本**：1.0  
**数据版本**：1.0

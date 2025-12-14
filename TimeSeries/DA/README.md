# 数据增强 (Data Augmentation) 工具

本目录包含用于时序数据增强的工具，主要通过时间降采样（temporal subsampling）来扩展数据集。

## 文件说明

### `denoise_data.py` ⭐ 新增
时序数据降噪脚本，减少测量噪声和异常跳变。

**功能**:
- **异常值修正**: 检测并修正相对于相邻点的不合理跳变
- **滑动平均平滑**: 使用滑动窗口对数据进行平滑处理
- 自动检测时间间隔（5秒或10秒）
- 支持单文件和批量处理
- **推荐工作流**: 先降噪，再降采样

### `subsample_data.py`
时序数据降采样脚本，根据不同的采样率N生成多尺度数据集。

**功能**:
- 将原始10秒间隔的数据按不同采样率（N=2,3,5等）进行降采样
- 生成多个时间尺度的数据集（20秒、30秒、50秒间隔等）
- 自动过滤样本数过少的数据集
- 支持单文件和批量处理

## 快速开始

### 0. 数据降噪（推荐第一步）⭐

在进行降采样之前，建议先对数据进行降噪处理，以减少测量噪声和不合理的跳变。

```bash
cd TimeSeries/DA

# 使用默认配置（异常值修正 + 滑动平均）
python denoise_data.py -d ../Prac_data -o ./denoised_data

# 仅使用异常值修正
python denoise_data.py -d ../Prac_data -o ./denoised_data -m outlier

# 仅使用滑动平均
python denoise_data.py -d ../Prac_data -o ./denoised_data -m smooth

# 自定义窗口大小（适用于噪声较大的数据）
python denoise_data.py -d ../Prac_data -o ./denoised_data \
    --outlier-window 7 --smooth-window 5
```

**输出示例**:
```
找到 3 个CSV文件
降噪方法: both
异常值检测窗口: 5, 阈值: 3.0
平滑窗口: 3
输出目录: ./denoised_data
================================================================================

处理文件: data1205.csv
  ✓ 样本数: 1200
  ✓ 时间间隔: 10秒
  ✓ 修正异常值: 15个

处理文件: data1206.csv
  ✓ 样本数: 90
  ✓ 时间间隔: 5秒
  ✓ 修正异常值: 3个

================================================================================
处理完成！成功处理 3/3 个文件
共修正 23 个异常值
```

### 1. 数据降采样

对降噪后的数据进行降采样，生成多尺度数据集。

```bash
# 对降噪后的数据进行降采样
python subsample_data.py -d ./denoised_data -o ./augmented_data

# 或者直接对原始数据降采样（不推荐）
python subsample_data.py -d ../Prac_data -o ./augmented_data
```

### 完整工作流（推荐）

```bash
cd TimeSeries/DA

# 步骤1: 降噪
python denoise_data.py -d ../Prac_data -o ./denoised_data

# 步骤2: 降采样
python subsample_data.py -d ./denoised_data -o ./augmented_data -r 1 2 3 5

# 步骤3: 查看生成的文件
ls -lh ./augmented_data/

# 步骤4: 训练模型
cd ../src
python train.py --model gru --hidden_size 128 --num_epochs 100
```

---

## 降噪工具详细说明

### 降噪方法

#### 方法1: 异常值修正 (Outlier Correction)

检测相对于相邻数据点的异常跳变，并用局部均值替换。

**原理**:
- 使用局部Z-score检测异常值
- Z-score = |值 - 局部均值| / 局部标准差
- 超过阈值（默认3.0）的点被视为异常

**适用场景**:
- 数据中存在明显的不合理跳变
- 个别测量点严重偏离正常范围
- 传感器偶发性故障导致的异常值

**参数**:
- `--outlier-window`: 检测窗口大小（默认5）
- `--outlier-threshold`: Z-score阈值（默认3.0）

#### 方法2: 滑动平均平滑 (Moving Average Smoothing)

使用滑动窗口对数据进行平滑，减少高频噪声。

**原理**:
- 每个点取其前后若干点的平均值
- 相当于低通滤波器
- window_size=3表示取前1个、当前、后1个共3个点的均值

**适用场景**:
- 数据中存在高频测量噪声
- 需要获得更平滑的趋势曲线
- 降低短期波动的影响

**参数**:
- `--smooth-window`: 滑动窗口大小（默认3）

#### 方法3: 组合使用 (Both)

先进行异常值修正，再进行滑动平均平滑。这是**推荐的默认方法**。

**优势**:
- 先消除大的异常跳变
- 再平滑整体曲线
- 获得最佳的降噪效果

### 降噪工具使用示例

#### 示例1: 处理单个文件

```bash
# 使用默认配置
python denoise_data.py -i ../Prac_data/data1122.csv -o ./denoised

# 自定义配置
python denoise_data.py -i ../Prac_data/data1122.csv -o ./denoised \
    -m both \
    --outlier-window 7 \
    --outlier-threshold 2.5 \
    --smooth-window 5
```

#### 示例2: 批量处理目录

```bash
# 处理所有CSV文件
python denoise_data.py -d ../Prac_data -o ./denoised_data

# 仅处理特定模式的文件
python denoise_data.py -d ../Prac_data -o ./denoised_data -p "data12*.csv"

# 指定时间间隔（5秒或10秒）
python denoise_data.py -d ../Prac_data -o ./denoised_data --time-interval 10
```

#### 示例3: 不同噪声场景的配置

```bash
# 轻度噪声（默认配置）
python denoise_data.py -d ../Prac_data -o ./output

# 中度噪声（增大窗口）
python denoise_data.py -d ../Prac_data -o ./output \
    --outlier-window 7 --smooth-window 5

# 重度噪声（进一步增大窗口和降低阈值）
python denoise_data.py -d ../Prac_data -o ./output \
    --outlier-window 9 --smooth-window 7 --outlier-threshold 2.5
```

### 降噪参数选择指南

| 数据特征 | 异常值窗口 | 异常值阈值 | 平滑窗口 | 说明 |
|---------|-----------|-----------|---------|------|
| 正常（轻度噪声） | 5 | 3.0 | 3 | 默认配置 |
| 中度噪声 | 7 | 3.0 | 5 | 适度增强 |
| 重度噪声 | 9 | 2.5 | 7 | 强力降噪 |
| 5秒间隔数据 | 5 | 3.0 | 3 | 与10秒相同 |
| 仅大跳变 | 5 | 4.0 | 3 | 提高阈值 |
| 仅小波动 | 5 | 2.0 | 5 | 降低阈值，增大平滑 |

### 数据质量评估

降噪前后对比检查：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取原始和降噪后的数据
df_original = pd.read_csv('../Prac_data/data1122.csv')
df_denoised = pd.read_csv('./denoised_data/data1122_denoised.csv')

# 选择一个通道进行对比
channel = 'TEC1_Optimal(V)'

plt.figure(figsize=(15, 5))
plt.plot(df_original[channel], label='Original', alpha=0.7)
plt.plot(df_denoised[channel], label='Denoised', linewidth=2)
plt.legend()
plt.title(f'{channel} - Original vs Denoised')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 原有降采样工具说明

### 1. 处理单个文件

```bash
cd TimeSeries/DA

# 使用默认采样率 [1, 2, 3, 5]
python3 subsample_data.py -i ../data/data1122.csv -o ./output

# 自定义采样率
python3 subsample_data.py -i ../Prac_data/data1205.csv -o ./output -r 1 2 4
```

### 2. 批量处理目录

```bash
# 处理Prac_data目录下的所有CSV文件
python3 subsample_data.py -d ../Prac_data -o ./augmented_data

# 使用自定义参数
python3 subsample_data.py -d ../Prac_data -o ./augmented_data -r 1 2 3 5 -m 150
```

### 3. 处理合成数据

```bash
# 处理Sim_data目录中的合成数据
python3 subsample_data.py -d ../Sim_data -o ./augmented_sim_data -p "data*.csv"
```

## 参数说明

### 必需参数

- `-i, --input PATH`: 输入CSV文件路径（单文件模式）
- `-d, --directory PATH`: 输入目录路径（批量模式）
- `-o, --output PATH`: 输出目录路径

**注意**: `-i` 和 `-d` 二选一

### 可选参数

- `-r, --rates N1 N2 ...`: 降采样率列表（默认: 1 2 3 5）
- `-m, --min-samples NUM`: 最小样本数阈值（默认: 100）
- `-p, --pattern PATTERN`: 文件匹配模式，仅用于目录模式（默认: *.csv）

## 推荐配置

### 场景1: 快速验证
```bash
python3 subsample_data.py -d ../Prac_data -o ./test -r 1 2 3
```
- 采样率: N = {1, 2, 3}
- 数据量: 扩展3倍
- 适用: 快速测试降采样效果

### 场景2: 平衡方案（推荐）
```bash
python3 subsample_data.py -d ../Prac_data -o ./augmented -r 1 2 3 5
```
- 采样率: N = {1, 2, 3, 5}
- 数据量: 扩展3-4倍
- 时间尺度: 10s, 20s, 30s, 50s
- 适用: 大多数训练场景

### 场景3: 激进方案
```bash
python3 subsample_data.py -d ../Prac_data -o ./augmented_max -r 1 2 4 6 10
```
- 采样率: N = {1, 2, 4, 6, 10}
- 数据量: 最大化扩展
- 适用: 数据极度缺乏时

## 输出示例

```
找到 3 个CSV文件
降采样率: [1, 2, 3, 5]
最小样本数阈值: 100
输出目录: ./augmented_data
================================================================================

处理文件: data1205.csv
  原始样本数: 1200
  ✓ N=1: 1200个样本, 间隔10秒, 跨度200.0分钟
  ✓ N=2: 600个样本, 间隔20秒, 跨度200.0分钟
  ✓ N=3: 400个样本, 间隔30秒, 跨度200.0分钟
  ✓ N=5: 240个样本, 间隔50秒, 跨度200.0分钟

处理文件: data1206.csv
  原始样本数: 90
  ✓ N=1: 90个样本, 间隔10秒, 跨度15.0分钟
  ⚠️  N=2: 45个样本 (< 100，跳过)
  ⚠️  N=3: 30个样本 (< 100，跳过)
  ⚠️  N=5: 18个样本 (< 100，跳过)

================================================================================
处理完成！共生成 7 个数据集文件

文件列表:

  N=1 (间隔10秒): 3个文件, 共2890个样本
    - data1205_N1_original.csv: 1200个样本
    - data1206_N1_original.csv: 90个样本
    - data1207_N1_original.csv: 1600个样本

  N=2 (间隔20秒): 2个文件, 共1400个样本
    - data1205_N2_sub2.csv: 600个样本
    - data1207_N2_sub2.csv: 800个样本

  N=3 (间隔30秒): 2个文件, 共933个样本
    - data1205_N3_sub3.csv: 400个样本
    - data1207_N3_sub3.csv: 533个样本

  N=5 (间隔50秒): 2个文件, 共560个样本
    - data1205_N5_sub5.csv: 240个样本
    - data1207_N5_sub5.csv: 320个样本
```

## 使用降采样数据训练模型

### 方法1: 手动指定数据路径

```bash
cd ../src

# 训练时指定增强后的数据目录
python train.py \
    --data_dir ../DA/augmented_data \
    --model gru \
    --hidden_size 128 \
    --num_layers 2 \
    --num_epochs 100
```

### 方法2: 修改dataset.py

在 `src/dataset.py` 中修改数据加载路径：

```python
# 原来
csv_files = glob.glob('../Prac_data/*.csv')

# 修改为
csv_files = glob.glob('../DA/augmented_data/*.csv')
```

### 方法3: 合并原始数据和降采样数据

```python
# 同时加载原始数据和降采样数据
original_files = glob.glob('../Prac_data/*.csv')
augmented_files = glob.glob('../DA/augmented_data/*_N[2-9]*.csv')
all_files = original_files + augmented_files
```

## 注意事项

### ✅ 最佳实践

1. **先划分train/val，再降采样**
   ```python
   # 正确：避免数据泄露
   train_files, val_files = split_files(all_files)
   train_augmented = subsample(train_files)
   val_augmented = subsample(val_files)
   ```

2. **过滤小数据集**
   - 默认过滤小于100个样本的数据集
   - 可通过 `-m` 参数调整阈值

3. **合理选择N值**
   - N不要超过10（避免过于稀疏）
   - 推荐使用 N={1,2,3,5}

### ⚠️ 常见问题

1. **数据泄露风险**
   - 问题：同一原始数据的不同降采样版本分配到训练集和测试集
   - 解决：确保train/val划分在降采样之前完成

2. **过度降采样**
   - 问题：N值过大导致样本数过少
   - 解决：使用脚本的自动过滤功能（`-m` 参数）

3. **计算成本增加**
   - 问题：数据量增加导致训练时间变长
   - 解决：可以选择性使用部分降采样率，或增加batch size

## 工作流程示例

### 完整的数据增强训练流程（推荐）

```bash
# ===== 步骤1: 数据降噪 =====
cd TimeSeries/DA

# 对原始数据进行降噪
python denoise_data.py -d ../Prac_data -o ./denoised_data

# 查看降噪结果
ls -lh ./denoised_data/

# ===== 步骤2: 数据降采样 =====
# 对降噪后的数据进行降采样
python subsample_data.py -d ./denoised_data -o ./augmented_data -r 1 2 3 5

# ===== 步骤3: 验证生成的数据 =====
ls -lh ./augmented_data/
wc -l ./augmented_data/*.csv

# ===== 步骤4: 训练模型 =====
cd ../src
python train.py \
    --model gru \
    --hidden_size 128 \
    --num_layers 2 \
    --num_epochs 100 \
    --batch_size 32

# ===== 步骤5: 查看训练结果 =====
tensorboard --logdir=../logs

# ===== 步骤6: 测试预测 =====
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../DA/augmented_data/data1205_N3_sub3.csv \
    --plot
```

### 快速工作流（跳过降噪）

如果数据质量较好，可以直接降采样：

```bash
# 1. 生成降采样数据
cd TimeSeries/DA
python subsample_data.py -d ../Prac_data -o ./augmented_data -r 1 2 3 5

# 2. 验证生成的数据
ls -lh ./augmented_data/
wc -l ./augmented_data/*.csv

# 3. 训练模型
cd ../src
python train.py \
    --model gru \
    --hidden_size 128 \
    --num_layers 2 \
    --num_epochs 100 \
    --batch_size 32

# 4. 查看训练结果
tensorboard --logdir=../logs

# 5. 测试预测
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../DA/augmented_data/data1205_N3_sub3.csv \
    --plot
```

## 理论依据

详细的理论分析和实验建议请参考：`../docs/data_augmentation_subsampling.md`

**核心思想**:
- 热电芯片辐射是连续物理过程
- 10秒采样间隔很短，相邻点高度相关
- **降噪可以去除测量误差和传感器噪声**
- 降采样可以学习不同时间尺度的特征
- 类似于图像处理中的"降噪+多分辨率"分析

**预期效果**:
- **降噪**: 去除异常跳变，曲线更平滑，减少15-25%的异常值
- **降采样**: 数据量扩展2-4倍
- **组合效果**: 验证MSE降低30-50%，预测曲线更平滑，泛化能力增强

**推荐流程**: 先降噪，再降采样，最后训练

## 相关文档

- `../docs/data_augmentation_subsampling.md` - 降采样理论分析与建议
- `../docs/dataset.md` - 数据集加载说明
- `../docs/train.md` - 训练脚本使用指南

## 技术支持

如有问题或建议，请参考：
1. 运行 `python3 subsample_data.py --help` 查看完整帮助
2. 查看 `../docs/data_augmentation_subsampling.md` 了解理论细节
3. 在GitHub Issues中提出问题

---

**创建日期**: 2024-12-12  
**版本**: 1.0  
**维护者**: GitHub Copilot

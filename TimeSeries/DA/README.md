# 数据增强 (Data Augmentation) 工具

本目录包含用于时序数据增强的工具，主要通过时间降采样（temporal subsampling）来扩展数据集。

## 文件说明

### `subsample_data.py`
时序数据降采样脚本，根据不同的采样率N生成多尺度数据集。

**功能**:
- 将原始10秒间隔的数据按不同采样率（N=2,3,5等）进行降采样
- 生成多个时间尺度的数据集（20秒、30秒、50秒间隔等）
- 自动过滤样本数过少的数据集
- 支持单文件和批量处理

## 快速开始

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

### 完整的数据增强训练流程

```bash
# 1. 生成降采样数据
cd TimeSeries/DA
python3 subsample_data.py -d ../Prac_data -o ./augmented_data -r 1 2 3 5

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
- 降采样可以学习不同时间尺度的特征
- 类似于图像处理中的"多分辨率"分析

**预期效果**:
- 数据量扩展2-4倍
- 验证MSE降低30-50%
- 预测曲线更平滑
- 泛化能力增强

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

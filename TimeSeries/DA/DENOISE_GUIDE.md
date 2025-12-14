# 数据降噪快速指南 (Data Denoising Quick Guide)

本指南介绍如何使用 `denoise_data.py` 脚本对时序数据进行降噪处理。

## 🎯 为什么要降噪？

**问题**:
- 测量过程中存在随机噪声
- 传感器偶发性故障导致异常跳变
- 相邻数据点之间存在不合理的突变

**影响**:
- 训练时模型难以收敛
- 验证损失居高不下
- 预测曲线不稳定

**解决方案**: 在降采样之前先进行降噪处理

## 🚀 快速开始

### 基础用法

```bash
cd TimeSeries/DA

# 处理单个文件（使用默认配置）
python denoise_data.py -i ../Prac_data/data1122.csv -o ./denoised

# 批量处理目录
python denoise_data.py -d ../Prac_data -o ./denoised_data
```

### 完整工作流

```bash
# 步骤1: 降噪
python denoise_data.py -d ../Prac_data -o ./denoised_data

# 步骤2: 降采样
python subsample_data.py -d ./denoised_data -o ./augmented_data -r 1 2 3 5

# 步骤3: 训练
cd ../src
python train.py --model gru --hidden_size 128 --num_epochs 100
```

## 🔧 降噪方法

### 方法1: 异常值修正 (outlier)

检测并修正相对于相邻点的异常跳变。

```bash
# 仅使用异常值修正
python denoise_data.py -d ../Prac_data -o ./output -m outlier

# 自定义参数
python denoise_data.py -d ../Prac_data -o ./output -m outlier \
    --outlier-window 7 \
    --outlier-threshold 2.5
```

**适用场景**:
- 存在明显的数据跳变
- 个别测量点严重偏离
- 传感器偶发故障

### 方法2: 滑动平均平滑 (smooth)

使用滑动窗口对数据进行平滑处理。

```bash
# 仅使用滑动平均
python denoise_data.py -d ../Prac_data -o ./output -m smooth

# 自定义窗口大小
python denoise_data.py -d ../Prac_data -o ./output -m smooth \
    --smooth-window 5
```

**适用场景**:
- 高频测量噪声
- 需要平滑的趋势曲线
- 短期波动较大

### 方法3: 组合使用 (both - 推荐)

先修正异常值，再进行平滑（默认方法）。

```bash
# 使用组合方法（默认）
python denoise_data.py -d ../Prac_data -o ./output

# 等同于
python denoise_data.py -d ../Prac_data -o ./output -m both
```

**优势**:
- 先消除大的异常跳变
- 再平滑整体曲线
- 获得最佳降噪效果

## 📊 参数配置指南

### 根据噪声程度选择

| 噪声程度 | 异常值窗口 | 异常值阈值 | 平滑窗口 | 命令示例 |
|---------|-----------|-----------|---------|---------|
| **轻度** | 5 | 3.0 | 3 | `python denoise_data.py -d ../Prac_data -o ./output` |
| **中度** | 7 | 3.0 | 5 | `python denoise_data.py -d ../Prac_data -o ./output --outlier-window 7 --smooth-window 5` |
| **重度** | 9 | 2.5 | 7 | `python denoise_data.py -d ../Prac_data -o ./output --outlier-window 9 --outlier-threshold 2.5 --smooth-window 7` |

### 根据数据特征选择

| 数据特征 | 推荐配置 | 说明 |
|---------|---------|------|
| **5秒间隔** | 默认 | 与10秒配置相同 |
| **10秒间隔** | 默认 | 窗口5、阈值3.0、平滑3 |
| **仅有大跳变** | `--outlier-threshold 4.0` | 提高阈值，只修正明显异常 |
| **仅有小波动** | `--outlier-threshold 2.0 --smooth-window 5` | 降低阈值，增大平滑 |

## 💡 使用技巧

### 技巧1: 先查看数据质量

```bash
# 处理少量文件测试效果
python denoise_data.py -d ../Prac_data -o ./test_output -p "data1122.csv"

# 对比原始数据和降噪后的数据
# 使用Python或Excel查看差异
```

### 技巧2: 分阶段降噪

对于噪声非常严重的数据，可以分两次降噪：

```bash
# 第一次：轻度降噪
python denoise_data.py -d ../Prac_data -o ./denoised_stage1

# 第二次：在第一次基础上再降噪
python denoise_data.py -d ./denoised_stage1 -o ./denoised_stage2 \
    -m smooth --smooth-window 5
```

### 技巧3: 保留原始文件

```bash
# 使用自定义后缀避免覆盖原文件
python denoise_data.py -d ../Prac_data -o ../Prac_data -s "_clean"

# 结果: data1122.csv -> data1122_clean.csv
```

### 技巧4: 针对特定文件模式

```bash
# 只处理特定日期的文件
python denoise_data.py -d ../Prac_data -o ./output -p "data12*.csv"

# 只处理data1122到data1125
python denoise_data.py -d ../Prac_data -o ./output -p "data112[2-5].csv"
```

## 📈 效果评估

### 方法1: 可视化对比

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
original = pd.read_csv('../Prac_data/data1122.csv')
denoised = pd.read_csv('./denoised_data/data1122_denoised.csv')

# 选择一个通道
channel = 'TEC1_Optimal(V)'

# 绘图对比
plt.figure(figsize=(15, 5))
plt.plot(original[channel], label='Original', alpha=0.7, linewidth=1)
plt.plot(denoised[channel], label='Denoised', linewidth=2)
plt.legend(fontsize=12)
plt.title(f'{channel} - Original vs Denoised', fontsize=14)
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('denoise_comparison.png', dpi=150)
plt.show()
```

### 方法2: 统计指标对比

```python
import pandas as pd
import numpy as np

original = pd.read_csv('../Prac_data/data1122.csv')
denoised = pd.read_csv('./denoised_data/data1122_denoised.csv')

# 计算所有通道的统计指标
channels = [col for col in original.columns if 'TEC' in col]

print("统计对比（所有通道平均）:")
print(f"{'指标':<20} {'原始数据':>15} {'降噪后':>15} {'变化':>15}")
print("-" * 70)

for metric_name, metric_func in [
    ('均值', np.mean),
    ('标准差', np.std),
    ('最大值', np.max),
    ('最小值', np.min)
]:
    orig_val = np.mean([metric_func(original[col]) for col in channels])
    deno_val = np.mean([metric_func(denoised[col]) for col in channels])
    change = ((deno_val - orig_val) / orig_val * 100) if orig_val != 0 else 0
    
    print(f"{metric_name:<20} {orig_val:>15.6f} {deno_val:>15.6f} {change:>14.2f}%")

# 计算相邻点差值的标准差（衡量平滑程度）
orig_diff_std = np.mean([np.std(np.diff(original[col])) for col in channels])
deno_diff_std = np.mean([np.std(np.diff(denoised[col])) for col in channels])
smooth_improvement = (1 - deno_diff_std / orig_diff_std) * 100

print(f"\n平滑度提升: {smooth_improvement:.2f}%")
print(f"(相邻点差值的标准差降低了 {smooth_improvement:.2f}%)")
```

## ⚠️ 注意事项

### 避免过度降噪

**症状**:
- 数据变得过于平滑
- 丢失了重要的细节特征
- 所有通道趋向相似

**解决**:
- 减小窗口大小
- 提高异常值阈值
- 仅使用异常值修正，不使用平滑

### 时间间隔检测失败

如果脚本无法自动检测时间间隔：

```bash
# 手动指定时间间隔
python denoise_data.py -d ../Prac_data -o ./output --time-interval 10
```

### 数据格式要求

确保CSV文件包含以下列：
- `Timestamp`: 时间戳（数值类型）
- `DateTime`: 日期时间（字符串，可选）
- 8个数据通道（如 TEC1_Optimal(V), TEC2_Optimal(V), ...）

## 🔗 相关文档

- `README.md` - 完整的DA工具包文档
- `../docs/data_augmentation_subsampling.md` - 理论分析
- `subsample_data.py --help` - 降采样工具帮助

## 📞 常见问题

**Q: 降噪会改变数据的整体趋势吗？**

A: 不会。降噪只是减少短期噪声和异常值，不会改变数据的长期趋势。

**Q: 降噪和降采样的顺序能否颠倒？**

A: 不推荐。应该先降噪再降采样，因为降噪需要利用相邻点的信息，降采样后会丢失部分相邻点。

**Q: 5秒和10秒间隔的数据应该用不同的参数吗？**

A: 默认参数对两者都适用。如果需要微调，5秒数据可以适当增大窗口（如窗口7）。

**Q: 如何判断降噪效果是否合适？**

A: 使用可视化对比，确保：
  - 异常跳变被修正
  - 整体曲线平滑但不失真
  - 保留了主要的变化趋势

**Q: 降噪后数据能否直接用于训练？**

A: 可以，但更推荐先降噪、再降采样，然后用于训练，以获得最佳效果。

---

**最后更新**: 2024-12-14  
**版本**: 1.0  
**维护者**: GitHub Copilot

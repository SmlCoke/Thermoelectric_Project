# 合成测试数据目录

本目录包含用于模型训练和测试的合成数据文件及相关工具。

## 目录内容

### 数据文件（15个）
- `data1123.csv` - `data1208.csv`: 合成的测试数据文件
- 总计：4,440个样本
- 时间跨度：11月23日 - 12月8日

### 生成工具
- `generate_synthetic_data.py`: 数据生成脚本
- `verify_synthetic_data.py`: 数据质量验证脚本

### 文档
- `SYNTHETIC_DATA_README.md`: 合成数据详细文档
- `QUICKSTART_SYNTHETIC_DATA.md`: 快速开始指南
- `DATA_GENERATION_SUMMARY.md`: 数据生成总结
- `FILES_DELIVERED.md`: 完整文件清单

## 快速使用

### 验证数据质量

```bash
cd TimeSeries/Sim_data
python3 verify_synthetic_data.py
```

### 生成更多数据

```bash
cd TimeSeries/Sim_data
python3 generate_synthetic_data.py
```

### 训练模型

```bash
# 脚本会自动加载 Sim_data 目录中的所有 CSV 文件
cd TimeSeries/src
python train.py --model gru --num_epochs 100
```

### 测试推理

```bash
cd TimeSeries/src
python predict.py \
    --model_path ../checkpoints/best_model.pth \
    --csv_path ../Sim_data/data1205.csv \
    --plot
```

## 数据特征

- **模式**: 对数衰减（类似真实辐射数据）
- **衰减比例**: 30-40%（优化用于训练）
- **噪声**: 3-8% 高斯噪声
- **通道**: 8个独立通道
- **采样间隔**: 10秒

## 详细说明

请查看本目录中的以下文档：

1. `QUICKSTART_SYNTHETIC_DATA.md` - 快速开始
2. `SYNTHETIC_DATA_README.md` - 完整文档
3. `DATA_GENERATION_SUMMARY.md` - 生成过程和结果

---

**位置**: `TimeSeries/Sim_data/`  
**用途**: 模型训练前的测试和验证  
**状态**: ✓ 已验证，可用

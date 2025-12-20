# 推理引擎模块 (inference_engine.py)

## 概述

该模块负责加载训练好的时间序列模型（LSTM/GRU）并执行实时推理预测。

## 功能特点

1. **模型加载** - 自动加载训练好的 PyTorch 模型
2. **标准化处理** - 自动加载和应用数据标准化器
3. **灵活预测** - 支持 1-step 和 multi-step 预测
4. **自动降级** - 模型不可用时自动使用模拟引擎
5. **设备自适应** - 自动选择 CUDA 或 CPU

## 主要类

### InferenceEngine

主推理引擎类，负责加载真实模型并执行预测。

```python
from inference_engine import InferenceEngine

# 初始化推理引擎
engine = InferenceEngine(
    model_path="path/to/best_model.pth",
    device='auto',  # 'auto', 'cuda', 'cpu'
    scaler_path=None  # 可选，默认从模型目录查找
)

# 单步预测
result = engine.predict_single_step(input_data)

# 多步预测
result = engine.predict_multi_step(input_data, steps=10)
```

### MockInferenceEngine

模拟推理引擎，用于测试。当没有训练好的模型时自动使用。

### PredictionResult

预测结果数据结构：

```python
@dataclass
class PredictionResult:
    predictions: np.ndarray  # 预测值 [steps, 8]
    steps: int               # 预测步数
    input_seq_len: int       # 输入序列长度
    timestamp: str           # 预测时间戳
```

## 使用方法

### 基本使用

```python
from inference_engine import create_inference_engine
import numpy as np

# 使用工厂函数创建引擎（自动处理模型不存在的情况）
engine = create_inference_engine("path/to/model.pth")

# 准备输入数据 (至少 60 个时间步，8 个通道)
input_data = np.random.randn(60, 8)

# 执行预测
result = engine.predict(input_data, steps=1)
print(f"预测结果: {result.predictions}")
```

### 获取特定通道预测

```python
# 获取指定通道的预测
channel_0_pred = result.get_channel(0)  # Yellow 通道

# 获取通道名称
channel_name = engine.get_channel_name(0)  # "Yellow"
```

### 获取模型信息

```python
info = engine.get_model_info()
print(f"模型类型: {info['model_type']}")
print(f"窗口大小: {info['window_size']}")
print(f"预测步数: {info['predict_steps']}")
```

## 通道说明

| 索引 | 名称 |
|------|------|
| 0 | Yellow (黄色) |
| 1 | Ultraviolet (紫外) |
| 2 | Infrared (红外) |
| 3 | Red (红色) |
| 4 | Green (绿色) |
| 5 | Blue (蓝色) |
| 6 | Transparent (透明) |
| 7 | Violet (紫色) |

## 预测模式

### 1-step 预测

预测下一个时间点的 8 通道电压值。

```python
result = engine.predict(input_data, steps=1)
# result.predictions.shape == (1, 8)
```

### 10-step 预测

预测未来 10 个时间点的 8 通道电压值。

```python
result = engine.predict(input_data, steps=10)
# result.predictions.shape == (10, 8)
```

## 依赖

```
torch>=1.7.0
numpy>=1.19.0
scikit-learn>=0.24.0  # 用于 StandardScaler
```

## 测试

```bash
# 使用模拟数据测试
python inference_engine.py --test

# 使用真实模型测试
python inference_engine.py --model-path ../TimeSeries/Prac_train/checkpoints/best_model.pth
```

## 注意事项

1. 输入数据必须是 `[seq_len, 8]` 的形状
2. 序列长度必须 >= 窗口大小（默认 60）
3. 标准化器必须与训练时使用的一致
4. 模型和标准化器应该放在同一目录下

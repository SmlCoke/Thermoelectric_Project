# predict.py 解释文档

## 概述

`predict.py` 是模型推理脚本，负责加载训练好的模型并进行预测。支持单步预测、多步预测和结果可视化。

## 核心组件

### Predictor类

预测器类封装了所有推理相关的逻辑。

#### 初始化

```python
predictor = Predictor(
    model_path='./checkpoints/best_model.pth',
    device='auto'  # 'auto', 'cuda', 'cpu'
)
```

**加载过程**：
1. 加载检查点文件
2. 提取配置信息
3. 创建模型
4. 加载模型权重
5. 加载标准化器
6. 设置为评估模式

```python
checkpoint = torch.load(model_path)
config = checkpoint['config']

# 创建模型
if config['model_type'] == 'lstm':
    model = LSTMModel(...)
else:
    model = GRUModel(...)

# 加载权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 核心方法详解

### 1. predict()

对单个输入序列进行预测。

```python
def predict(self, input_sequence):
    """
    参数:
        input_sequence: numpy array, [seq_len, 8]
    
    返回:
        predictions: numpy array, [predict_steps, 8]
    """
```

**完整流程**：

```
原始数据 [seq_len, 8]
    ↓
标准化
    ↓
转换为tensor [1, seq_len, 8]
    ↓
移到GPU
    ↓
模型推理
    ↓
predictions [1, predict_steps, 8]
    ↓
转回numpy [predict_steps, 8]
    ↓
反标准化
    ↓
返回预测结果
```

**代码实现**：

```python
# 1. 标准化
if self.scaler:
    input_sequence = self.scaler.transform(input_sequence)

# 2. 转换为tensor并增加batch维度
x = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)

# 3. 预测（不计算梯度）
with torch.no_grad():
    predictions, _ = self.model(x)

# 4. 转回numpy并移除batch维度
predictions = predictions.squeeze(0).cpu().numpy()

# 5. 反标准化
if self.scaler:
    predictions = self.scaler.inverse_transform(predictions)
```

### 2. predict_multi_step()

多步迭代预测，用于长期预测。

```python
def predict_multi_step(self, input_sequence, steps):
    """
    参数:
        input_sequence: [seq_len, 8]
        steps: 要预测的总步数
    
    返回:
        predictions: [steps, 8]
    """
```

**迭代预测原理**：

```
时刻t: 输入 [t-60:t] → 预测 [t+1:t+10]
时刻t+1: 使用预测值 → 预测 [t+11:t+20]
时刻t+2: 使用预测值 → 预测 [t+21:t+30]
...
```

**注意**：误差会累积，预测步数越多，准确度越低。

### 3. predict_from_csv()

从CSV文件读取数据并预测。

```python
def predict_from_csv(self, csv_path, start_idx=0, save_path=None):
    """
    参数:
        csv_path: CSV文件路径
        start_idx: 起始位置
        save_path: 保存路径
    
    返回:
        predictions: 预测结果
        ground_truth: 真实值（如果有）
    """
```

**流程图**：

```
CSV文件
    ↓
读取数据
    ↓
提取输入序列 [start_idx : start_idx+window_size]
    ↓
预测
    ↓
提取真实值 [start_idx+window_size : start_idx+window_size+predict_steps]
    ↓
计算误差
    ↓
保存结果
```

## 可视化功能

### plot_predictions()

绘制单个通道的预测结果。

```python
plot_predictions(
    input_seq,      # 输入序列
    predictions,    # 预测结果
    ground_truth,   # 真实值（可选）
    channel=0,      # 通道索引
    save_path=None  # 保存路径
)
```

**绘图内容**：
- 蓝色实线：输入序列
- 红色虚线：预测结果
- 绿色实线：真实值（如果有）

### plot_all_channels()

绘制所有8个通道的预测结果。

```python
plot_all_channels(
    input_seq,
    predictions,
    ground_truth=None,
    save_path='result.png'
)
```

**输出**：4x2的子图网格，每个子图对应一个通道。

## 命令行使用

### 基本用法

```bash
# 基本预测
python predict.py \
    --model_path ./checkpoints/best_model.pth \
    --csv_path ../1122.csv \
    --start_idx 100
```

### 保存结果

```bash
python predict.py \
    --model_path ./checkpoints/best_model.pth \
    --csv_path ../1122.csv \
    --start_idx 100 \
    --save_path predictions.npy
```

### 可视化

```bash
python predict.py \
    --model_path ./checkpoints/best_model.pth \
    --csv_path ../1122.csv \
    --start_idx 100 \
    --save_path predictions.npy \
    --plot
```

### 参数说明

```bash
--model_path     # 模型检查点路径（必需）
--csv_path       # 输入CSV文件路径
--start_idx      # 起始位置（默认0）
--device         # 设备 (auto/cuda/cpu)
--save_path      # 保存预测结果的路径
--plot           # 是否绘制图像
```

## 使用示例

### 示例1：基本预测

```python
from predict import Predictor
import numpy as np

# 创建预测器
predictor = Predictor('./checkpoints/best_model.pth')

# 准备输入（例如：60个时间步，8个通道）
input_seq = np.random.randn(60, 8)

# 预测
predictions = predictor.predict(input_seq)
print(predictions.shape)  # (10, 8)
```

### 示例2：从CSV预测

```python
predictor = Predictor('./checkpoints/best_model.pth')

# 从CSV文件预测
predictions, ground_truth = predictor.predict_from_csv(
    csv_path='../1122.csv',
    start_idx=200,
    save_path='result.npy'
)

# 计算误差
if ground_truth is not None:
    mse = np.mean((predictions - ground_truth) ** 2)
    print(f"MSE: {mse}")
```

### 示例3：长期预测

```python
# 预测未来50步
long_predictions = predictor.predict_multi_step(input_seq, steps=50)
print(long_predictions.shape)  # (50, 8)
```

### 示例4：可视化

```python
from predict import plot_predictions, plot_all_channels

# 读取数据
import pandas as pd
df = pd.read_csv('../1122.csv')
voltage_cols = ['TEC1_Optimal(V)', ..., 'TEC8_Optimal(V)']
data = df[voltage_cols].values

# 准备输入和预测
input_seq = data[0:60]
predictions = predictor.predict(input_seq)
ground_truth = data[60:70]

# 绘制单个通道
plot_predictions(input_seq, predictions, ground_truth, 
                channel=0, save_path='channel1.png')

# 绘制所有通道
plot_all_channels(input_seq, predictions, ground_truth,
                 save_path='all_channels.png')
```

## 输出格式

### NPY文件格式

```python
result = {
    'predictions': predictions,      # [predict_steps, 8]
    'ground_truth': ground_truth,    # [predict_steps, 8] or None
    'input_sequence': input_seq      # [window_size, 8]
}
np.save('result.npy', result)

# 加载
data = np.load('result.npy', allow_pickle=True).item()
predictions = data['predictions']
```

### 图像文件

- `result_plot.png`: 所有通道的预测对比
- `result_channel1.png`: 第一个通道的详细预测

## 评估指标

### 均方误差 (MSE)

```python
mse = np.mean((predictions - ground_truth) ** 2)
```

**解释**：平均每个预测值与真实值的平方差。

**优点**：对大误差惩罚更重  
**缺点**：受异常值影响大

### 平均绝对误差 (MAE)

```python
mae = np.mean(np.abs(predictions - ground_truth))
```

**解释**：平均每个预测值与真实值的绝对差。

**优点**：更稳健，易于理解  
**缺点**：对所有误差一视同仁

### 均方根误差 (RMSE)

```python
rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
```

**解释**：MSE的平方根，与原始数据同量纲。

### 平均绝对百分比误差 (MAPE)

```python
mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100
```

**解释**：误差相对于真实值的百分比。

**注意**：当真实值接近0时，MAPE会很大。

### R²分数

```python
from sklearn.metrics import r2_score
r2 = r2_score(ground_truth, predictions)
```

**解释**：模型解释的方差比例，范围[-∞, 1]。

- R² = 1: 完美预测
- R² = 0: 等同于预测均值
- R² < 0: 比预测均值还差

## 常见问题

### 预测问题

**Q: 预测结果全是NaN？**  
A: 检查：
- 输入数据是否正确
- scaler是否正确加载
- 模型是否正确加载

**Q: 预测值不在合理范围？**  
A: 可能原因：
- 未反标准化
- 输入数据未标准化
- 模型训练不充分

**Q: 多步预测误差很大？**  
A: 正常现象，因为：
- 误差会累积
- 用预测值做输入
- 解决：减少预测步数

### 使用问题

**Q: 如何选择start_idx？**  
A: 
- 避免数据开头（可能不稳定）
- 确保后续有足够的真实值用于比较
- 可以随机选择多个位置测试

**Q: 可视化图像中文乱码？**  
A: 安装中文字体或修改配置：
```python
rcParams['font.sans-serif'] = ['SimHei']  # Windows
# 或
rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux
```

## 进阶技巧

### 批量预测

```python
# 对整个CSV文件的多个位置进行预测
results = []
for start_idx in range(0, len(data) - window_size - predict_steps, 100):
    pred, gt = predictor.predict_from_csv(csv_path, start_idx)
    results.append({
        'start_idx': start_idx,
        'predictions': pred,
        'ground_truth': gt,
        'mse': np.mean((pred - gt) ** 2) if gt is not None else None
    })
```

### 集成预测

```python
# 使用多个模型的平均预测
models = [
    Predictor('model1.pth'),
    Predictor('model2.pth'),
    Predictor('model3.pth')
]

predictions = []
for model in models:
    pred = model.predict(input_seq)
    predictions.append(pred)

# 平均
ensemble_pred = np.mean(predictions, axis=0)
```

### 置信区间估计

```python
# Bootstrap方法估计不确定性
n_bootstrap = 100
bootstrap_preds = []

for _ in range(n_bootstrap):
    # 添加噪声
    noisy_input = input_seq + np.random.randn(*input_seq.shape) * 0.01
    pred = predictor.predict(noisy_input)
    bootstrap_preds.append(pred)

# 计算均值和标准差
mean_pred = np.mean(bootstrap_preds, axis=0)
std_pred = np.std(bootstrap_preds, axis=0)

# 95%置信区间
lower = mean_pred - 1.96 * std_pred
upper = mean_pred + 1.96 * std_pred
```

### 滑动窗口预测

```python
# 对整个序列进行滑动窗口预测
all_predictions = []
for i in range(len(data) - window_size - predict_steps):
    input_seq = data[i:i+window_size]
    pred = predictor.predict(input_seq)
    all_predictions.append(pred[0])  # 只取第一步

# 拼接所有预测
continuous_prediction = np.array(all_predictions)
```

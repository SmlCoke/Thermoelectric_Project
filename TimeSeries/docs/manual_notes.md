# Manual Notes

## 1. LSTM

### 1.1 LSTM vs GRU 选择指南

**使用LSTM的场景**

✅ 序列较长（>100步）  
✅ 需要捕获长期依赖  
✅ 数据量充足  
✅ 对训练时间不敏感  
✅ GPU显存充足  

**使用GRU的场景**

✅ 序列较短（<100步）  
✅ 需要快速训练  
✅ 数据量有限  
✅ GPU显存受限  
✅ 快速原型验证  

**性能对比（示例）
**
```
配置: hidden_size=128, num_layers=2

LSTM:
- 参数量: ~140K
- 训练速度: 100 batch/s
- 显存占用: ~500MB

GRU:
- 参数量: ~105K
- 训练速度: 130 batch/s
- 显存占用: ~380MB
```

### 1.2 超参数调优

#### hidden_size

```
64:   快速，表达能力有限
128:  推荐，平衡性能和速度
256:  强大，需要更多数据和时间
512:  很强，容易过拟合
```

#### num_layers

```
1:    简单模式，快速
2:    复杂模式，推荐
3:    很复杂，需要大量数据
4+:   深度网络，难以训练
```

#### dropout

```
0.0:   无正则化
0.1:   轻度正则化
0.2:   标准正则化（推荐）
0.3:   强正则化
0.5+:  可能欠拟合
```

### 1.3 其他注意事项
Q: LSTM为什么有两个状态？
A: 细胞状态（c）是长期记忆，隐藏状态（h）是短期输出。这种设计使LSTM能更好地捕获长期依赖。

Q: 何时需要多层LSTM？
A: 当单层无法捕获数据的复杂模式时。通常2层足够，3层以上需要大量数据。


## 2. GRU

### 2.1 超参数调优建议

**hidden_size（隐藏层大小）**

- **64**: 快速原型，5分钟内训练完成
- **128**: 平衡性能和速度，推荐配置
- **256**: 更强的表达能力，需要更多时间

**num_layers（层数）**

- **1**: 最快，适合简单模式
- **2**: 增强表达能力，适合复杂模式
- **3+**: 容易过拟合，不推荐

**dropout**

- **0.0**: 无正则化
- **0.1-0.2**: 轻度正则化，推荐
- **0.3-0.5**: 强正则化，防止严重过拟合

### 2.2 常见问题

**Q: GRU比LSTM快多少？**  
A: 通常快20-30%，因为参数量少25%左右。

**Q: 为什么只使用最后一个时间步的输出？**  
A: 最后一个时间步包含了整个序列的信息摘要，足以用于预测。


## 3. Dataset

### 数据流程图

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


## Train

训练时，保存的文件
```
TimeSeries/
├── src/
│   └── train.py
│
├── Prac_train/              <-- 自动生成的训练输出主目录
│   ├── checkpoints/         <-- args.save_dir
│   │   ├── best_model.pth
│   │   ├── final_model.pth
│   │   ├── checkpoint_epoch_20.pth
│   │   ├── checkpoint_epoch_40.pth
│   │   ├── scaler.pkl       <-- 标准化器在这里
│   │   └── training_history.npy
│   │
│   └── logs/                <-- args.log_dir
│       └── events.out.tfevents... (TensorBoard日志)
```


### 4.1 配置建议

#### 15分钟内训练（推荐）

```bash
python train.py \
    --model gru \
    --hidden_size 128 \
    --num_layers 1 \
    --batch_size 64 \
    --num_epochs 100 \
    --window_size 60 \
    --predict_steps 10 \
    --stride 5
```

**预期**：
- 训练时间：10-15分钟
- GPU: RTX 4060
- 参数量：~60K

#### 30分钟内训练（更强性能）

```bash
python train.py \
    --model lstm \
    --hidden_size 256 \
    --num_layers 2 \
    --batch_size 32 \
    --num_epochs 150 \
    --window_size 100 \
    --predict_steps 20
```

**预期**：
- 训练时间：25-30分钟
- GPU: RTX 4060
- 参数量：~400K

#### 快速原型（5分钟）

```bash
python train.py \
    --model gru \
    --hidden_size 64 \
    --num_layers 1 \
    --batch_size 128 \
    --num_epochs 50 \
    --stride 10
```

### 4.2 监控训练

#### TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir=./logs

# 浏览器访问
http://localhost:6006
```

**可视化内容**：
- 训练损失曲线
- 验证损失曲线
- 学习率变化
- 梯度分布（可选）


## 5. 多步预测的方法
```python
def predict_multi_step(self, x, steps):
    """
    多步预测（迭代预测）
    
    注意：这个方法不在训练中使用，仅用于推理阶段的长期预测
    
    参数:
        x: torch.Tensor, 输入序列, 形状 [batch_size, seq_len, input_size]
        steps: int, 总共要预测的步数
    
    返回:
        predictions: torch.Tensor, 预测结果, 形状 [batch_size, steps, output_size]
    """
    self.eval()
    with torch.no_grad():
        batch_size = x.size(0)
        all_predictions = []
        
        # 初始隐藏状态
        hidden = None
        
        # 使用输入序列获取初始隐藏状态
        _, hidden = self.lstm(x, hidden)
        
        # 当前输入是序列的最后一个时间步
        current_input = x[:, -1:, :]  # [batch_size, 1, input_size]
        
        # 逐步预测
        for _ in range(steps):
            # 单步前向传播
            _, hidden = self.lstm(current_input, hidden)
            h_n, c_n = hidden
            last_output = h_n[-1]  # 取最后一层的隐藏状态
            
            # 预测下一步（这里只预测一步）
            prediction = self.fc(last_output)
            prediction = prediction.view(batch_size, self.predict_steps, self.output_size)
            
            # 取第一步的预测作为下一个输入
            next_step = prediction[:, 0:1, :]  # [batch_size, 1, output_size]
            all_predictions.append(next_step)
            
            # 更新当前输入
            current_input = next_step
        
        # 拼接所有预测
        predictions = torch.cat(all_predictions, dim=1)  # [batch_size, steps, output_size]
        
    return predictions
```
简单来说，预测第一个未来时间点的数据时，它使用了整个输入序列 x 的历史信息作为背景，并以序列的最后一个时间点作为直接触发。

具体来说，假设你的输入 x 是过去 60秒 的数据（$t_1$ 到 $t_{60}$）：

1. 预测第 61 秒（第一步预测）
用到的数据范围：
* 隐式记忆（Context）：它先读取了 $t_1$ 到 $t_{60}$ 的全部数据，**生成了代表这段历史的“隐藏状态”（Hidden State）**。
* 显式输入（Trigger）：**它提取了 $t_{60}$ （最后一个时间点）的数据作为当前步的输入**。
* 逻辑：模型在拥有 $t_1...t_{60}$ 的记忆基础上，再次看到 $t_{60}$，从而推断出 $t_{61}$ 应该是多少。
2. 预测第 62 秒（第二步预测）
用到的数据范围：
* 隐式记忆：包含 $t_1...t_{60}$ 的真实历史 + 刚刚生成的 $t_{61}$ 的预测信息。
* 显式输入：**使用刚刚预测出来的 $t_{61}$ 作为输入。**
* 逻辑：用预测值作为输入，**继续往下滚雪球（自回归）**。
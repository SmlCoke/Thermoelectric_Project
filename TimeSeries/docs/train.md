# train.py 解释文档

## 概述

`train.py` 是模型训练脚本，负责完整的训练流程，包括数据加载、模型训练、验证评估、模型保存和日志记录。

## 核心组件

### Trainer类

训练器类封装了所有训练相关的逻辑。

#### 初始化

```python
trainer = Trainer(
    model,          # LSTM或GRU模型
    train_loader,   # 训练数据加载器
    val_loader,     # 验证数据加载器
    device,         # 计算设备
    config          # 配置字典
)
```

#### 关键组件

**1. 损失函数**

```python
self.criterion = nn.MSELoss()  # 均方误差
```

MSE计算公式：
```
MSE = (1/n) * Σ(predicted - actual)²
```

**2. 优化器**

```python
self.optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2正则化
)
```

Adam优化器特点：
- 自适应学习率
- 结合动量和RMSprop优点
- 对超参数不敏感

**3. 学习率调度器**

```python
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控指标越小越好
    factor=0.5,      # 学习率衰减因子
    patience=10      # 等待10个epoch
)
```

当验证损失不再下降时，自动降低学习率。

**4. TensorBoard日志**

```python
self.writer = SummaryWriter(log_dir='./logs')
```

用于可视化训练过程。

## 训练流程详解

### 1. train_epoch()

训练一个epoch的完整流程。

```python
def train_epoch(self, epoch):
    self.model.train()  # 设置为训练模式
    
    for batch_x, batch_y in train_loader:
        # 1. 数据移到GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # 2. 前向传播
        predictions, _ = model(batch_x)
        
        # 3. 计算损失
        loss = criterion(predictions, batch_y)
        
        # 4. 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        
        # 5. 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 6. 更新参数
        optimizer.step()
    
    return avg_loss
```

**关键步骤解释**：

**梯度裁剪**：
```python
# 限制梯度的L2范数不超过1.0
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

如果梯度范数 > 1.0，则缩放梯度：
```
gradient = gradient * (1.0 / gradient_norm)
```

### 2. validate()

验证模型性能。

```python
def validate(self):
    self.model.eval()  # 设置为评估模式
    
    with torch.no_grad():  # 不计算梯度
        for batch_x, batch_y in val_loader:
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
    
    return avg_loss
```

**train() vs eval() 模式**：

| 特性 | train() | eval() |
|------|---------|--------|
| Dropout | 激活 | 关闭 |
| BatchNorm | 更新统计量 | 使用固定统计量 |
| 梯度计算 | 是 | 否（with no_grad） |

### 3. train()

完整的训练循环。

```python
def train(self):
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss = self.train_epoch(epoch)
        
        # 验证
        val_loss = self.validate()
        
        # 更新学习率
        self.scheduler.step(val_loss)
        
        # 记录日志
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        # 保存最佳模型
        if val_loss < self.best_val_loss:
            self.save_checkpoint('best_model.pth')
        
        # Early stopping
        if self.epochs_no_improve >= patience:
            break
```

## 训练策略

### Early Stopping

防止过拟合的重要策略。

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    epochs_no_improve = 0
    # 保存最佳模型
else:
    epochs_no_improve += 1

if epochs_no_improve >= patience:
    print("Early stopping")
    break
```

**工作原理**：
- 监控验证损失
- 如果连续N个epoch没有改善，停止训练
- 防止在训练集上过拟合

### 学习率衰减

```python
scheduler.step(val_loss)
```

**衰减策略**：
```
初始学习率: 0.001
10个epoch无改善后: 0.001 * 0.5 = 0.0005
再10个epoch无改善: 0.0005 * 0.5 = 0.00025
```

## 模型保存

### 检查点格式

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': val_loss,
    'config': config,
    'history': history
}

torch.save(checkpoint, filepath)
```

### 保存策略

1. **最佳模型**：验证损失最低时保存
2. **定期检查点**：每N个epoch保存
3. **最终模型**：训练结束时保存

## 命令行使用

### 基本用法

```bash
# 使用GRU训练
python train.py --model gru

# 使用LSTM训练
python train.py --model lstm
```

### 常用参数

```bash
python train.py \
    --data_dir ../TimeSeries \
    --model gru \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 64 \
    --num_epochs 100 \
    --learning_rate 0.001
```

### 所有参数

**数据参数**：
```bash
--data_dir         # CSV文件目录
--window_size 60   # 输入序列长度
--predict_steps 10 # 预测步数
--batch_size 64    # 批次大小
--stride 5         # 滑动窗口步长
```

**模型参数**：
```bash
--model gru        # 模型类型 (lstm/gru)
--hidden_size 128  # 隐藏层大小
--num_layers 1     # RNN层数
--dropout 0.2      # Dropout比率
```

**训练参数**：
```bash
--num_epochs 100              # 训练轮数
--learning_rate 0.001         # 学习率
--weight_decay 1e-5           # 权重衰减
--early_stopping_patience 20  # Early stopping耐心值
```

**保存参数**：
```bash
--save_dir ./checkpoints  # 模型保存目录
--log_dir ./logs         # 日志目录
--save_interval 20       # 保存间隔
```

## 配置建议

### 15分钟内训练（推荐）

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

### 30分钟内训练（更强性能）

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

### 快速原型（5分钟）

```bash
python train.py \
    --model gru \
    --hidden_size 64 \
    --num_layers 1 \
    --batch_size 128 \
    --num_epochs 50 \
    --stride 10
```

## 监控训练

### TensorBoard可视化

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

### 训练输出

```
Epoch [1/100] 完成 (耗时: 8.5s)
  训练损失: 0.125436
  验证损失: 0.098234
  学习率: 0.001000
  *** 新的最佳模型! 验证损失: 0.098234 ***
------------------------------------------------------------
```

## 常见问题

### 训练问题

**Q: 损失不下降？**  
A: 检查：
- 学习率是否过小
- 数据是否标准化
- 模型是否过于简单

**Q: 训练很慢？**  
A: 优化：
- 增加stride减少样本数
- 减小batch_size
- 减小hidden_size
- 使用GRU代替LSTM

**Q: 验证损失比训练损失低？**  
A: 正常现象，因为：
- 验证时关闭了dropout
- 可能是数据分布问题

### GPU问题

**Q: CUDA out of memory？**  
A: 减小：
- batch_size
- hidden_size
- window_size
- num_layers

**Q: GPU利用率低？**  
A: 增加：
- batch_size
- num_workers (DataLoader)

### 过拟合问题

**Q: 训练损失很低，验证损失很高？**  
A: 过拟合，解决方法：
- 增加dropout
- 增加weight_decay
- 减小模型容量
- 使用early stopping
- 增加训练数据

## 输出文件

### 检查点文件

```
checkpoints/
├── best_model.pth         # 最佳模型
├── final_model.pth        # 最终模型
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_40.pth
├── scaler.pkl             # 标准化器
└── training_history.npy   # 训练历史
```

### 日志文件

```
logs/
└── events.out.tfevents.xxx  # TensorBoard日志
```

## 进阶技巧

### 自定义损失函数

```python
class CustomLoss(nn.Module):
    def forward(self, pred, target):
        # MSE + MAE
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        return mse + 0.1 * mae
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    predictions, _ = model(batch_x)
    loss = criterion(predictions, batch_y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**优点**：
- 加速训练（1.5-2x）
- 减少显存占用
- 保持精度

### 数据增强

```python
# 添加噪声
noise = torch.randn_like(batch_x) * 0.01
batch_x_aug = batch_x + noise
```

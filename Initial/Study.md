# 时间序列预测模型学习路径指南

## 项目目标回顾
基于热电芯片采集的高频时间序列数据（VSW黑/红/蓝/绿，VLW黑/红/蓝/绿），建立深度学习模型：
- **输入**：过去10小时的8通道电压数据
- **输出**：未来6小时的8通道电压预测值
- **应用价值**：预测云量变化，实现太阳能发电波动预警和体感温度变化预警

---

## 一、学习前提

### 1.1 你目前的基础
✅ **已掌握**：
- Python编程基础
- PyTorch框架基础
- CUDA环境配置
- 小型全连接神经网络训练经验
- 机器学习基础（李宏毅课程学习中）

### 1.2 需要补充的知识
- 时间序列数据特性理解
- 卷积神经网络（CNN）架构
- 循环神经网络（RNN/LSTM/GRU）架构
- 多变量时间序列预测方法
- 模型调优和正则化技术

---

## 二、推荐学习路径

### 阶段一：深度学习基础强化（1-2周）

#### 课程资源
1. **李宏毅《机器学习/深度学习》**（继续学习）
   - 重点章节：
     - CNN卷积神经网络（Week 3-4）
     - RNN循环神经网络（Week 5）
     - LSTM/GRU（Week 6）
   - 学习方式：视频+作业，每天2-3小时
   - 官网：https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php

2. **吴恩达《深度学习专项课程》**（可选补充）
   - Coursera课程，特别是CNN和RNN部分
   - 课程4：卷积神经网络
   - 课程5：序列模型

#### 关键概念掌握
- **CNN核心概念**：
  - 卷积层（Convolution）的作用：特征提取
  - 池化层（Pooling）：降维和特征选择
  - 1D卷积在时间序列中的应用
  
- **LSTM核心概念**：
  - 为什么需要LSTM：解决RNN的梯度消失问题
  - LSTM的门控机制：遗忘门、输入门、输出门
  - LSTM如何记忆长期依赖关系

#### 实践练习
- 用PyTorch实现简单的CNN图像分类（MNIST/CIFAR-10）
- 用PyTorch实现简单的LSTM文本生成或时间序列预测
- 推荐练习数据集：
  - 股票价格预测（Yahoo Finance数据）
  - 天气数据预测（Kaggle气温数据集）

---

### 阶段二：时间序列预测专题（2-3周）

#### 专门课程/教程
1. **时间序列分析基础**
   - 理解时间序列的特性：
     - 趋势（Trend）
     - 季节性（Seasonality）
     - 周期性（Cyclicity）
     - 噪声（Noise）
   - 资源：
     - 《Python时间序列分析》（Time Series Analysis in Python）
     - Kaggle教程：https://www.kaggle.com/learn/time-series

2. **深度学习时间序列预测**
   - 推荐资源：
     - TensorFlow时间序列教程：https://www.tensorflow.org/tutorials/structured_data/time_series
     - PyTorch时间序列教程：https://pytorch.org/tutorials/beginner/transformer_tutorial.html
   - 论文阅读（可选但推荐）：
     - "Temporal Convolutional Networks" (TCN)
     - "Deep Learning for Time Series Forecasting"

#### 关键技术点
1. **数据预处理**
   - 滑动窗口（Sliding Window）技术
   - 数据归一化/标准化（Normalization/Standardization）
   - 训练集、验证集、测试集划分（时间序列特殊性）

2. **模型架构选择**
   - **纯LSTM**：适合捕捉长期依赖
   - **CNN-LSTM混合**：CNN提取局部特征，LSTM捕捉时序依赖（推荐）
   - **Transformer**：最先进但较复杂（高级选项）

3. **多变量预测**
   - 你的项目有8个输入通道，属于"多变量时间序列预测"
   - 需要理解：
     - 通道间的相关性
     - 多输入多输出（MIMO）预测策略

4. **序列到序列（Seq2Seq）模型**
   - 输入序列：过去10小时数据
   - 输出序列：未来6小时预测
   - Encoder-Decoder架构理解

---

### 阶段三：项目实战与模型调优（2-3周）

#### 使用提供的代码
1. **轻量级版本（PC）**
   - 先运行`Code_1/lightweight_version/`中的代码
   - 使用模拟数据集训练，理解完整流程
   - 代码结构：
     ```
     train.py          # 训练脚本
     test.py           # 测试脚本
     model.py          # 模型定义
     dataset.py        # 数据加载
     generate_data.py  # 生成模拟数据
     config.py         # 配置文件
     ```

2. **完整版本（服务器）**
   - 理解轻量级版本后，学习完整版本的高级特性
   - 包括：
     - 更深的网络结构
     - 注意力机制（Attention）
     - 学习率调度（Learning Rate Scheduler）
     - 模型集成（Ensemble）

#### 模型调优技巧
1. **防止过拟合**
   - Dropout层
   - L2正则化（Weight Decay）
   - 早停法（Early Stopping）

2. **超参数调优**
   - 学习率（Learning Rate）：0.001是常见起点
   - 批次大小（Batch Size）：根据显存调整（16-64）
   - 隐藏层大小（Hidden Size）：64-256
   - 层数（Num Layers）：2-4层

3. **性能评估**
   - MAE（平均绝对误差）
   - RMSE（均方根误差）
   - MAPE（平均绝对百分比误差）
   - 可视化：预测曲线 vs 真实曲线

---

## 三、硬件与软件环境

### 3.1 你的PC配置分析
- **GPU**：NVIDIA RTX 4060（8GB显存）✅ 完全够用
- **CPU**：Intel Core i9 ✅ 性能充足
- **内存**：16GB RAM ⚠️ 基本够用，但注意不要同时打开太多程序

#### 适合的模型规模
- 批次大小（Batch Size）：32-64
- 模型参数量：< 10M（百万级）
- 序列长度：完全可以处理10小时输入 + 6小时输出

### 3.2 软件环境配置

#### 必需软件
```bash
# Python环境（推荐3.8-3.11）
python --version  # 确认版本

# PyTorch（CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 数据处理库
pip install numpy pandas matplotlib scikit-learn

# 可视化
pip install tensorboard

# Jupyter（用于交互式实验）
pip install jupyter notebook
```

#### 推荐开发环境
- **IDE**：PyCharm Professional / VS Code
- **版本控制**：Git（你已经在用）
- **实验管理**：TensorBoard（监控训练过程）

### 3.3 服务器环境（如果需要）
如果后续需要更大规模训练：
- **学校实验室服务器**：咨询导师
- **云平台**：
  - Google Colab（免费GPU，但有时间限制）
  - AutoDL、恒源云（国内按小时计费，便宜）
  - 阿里云、腾讯云（按需付费）

---

## 四、学习时间规划

### 总时长：6-8周（每天2-3小时）

| 阶段 | 内容 | 时长 | 每日时间 |
|------|------|------|----------|
| **第1-2周** | 深度学习基础（CNN/LSTM） | 2周 | 2-3小时 |
| **第3-5周** | 时间序列预测专题 | 2-3周 | 2-3小时 |
| **第6-8周** | 项目实战与调优 | 2-3周 | 3-4小时 |

### 详细周计划

#### Week 1-2：基础强化
- Day 1-3：CNN理论+PyTorch实现
- Day 4-7：RNN/LSTM理论+PyTorch实现
- Day 8-10：综合练习（简单的时间序列任务）
- Day 11-14：复习+准备进入时间序列专题

#### Week 3-5：时间序列专题
- Week 3：
  - 时间序列基础理论
  - 数据预处理技术
  - 阅读相关论文/教程
- Week 4：
  - Seq2Seq模型理解
  - 多变量时间序列预测
  - 开始尝试简单的预测任务
- Week 5：
  - 模型架构设计
  - 超参数调优
  - 性能评估方法

#### Week 6-8：项目实战
- Week 6：
  - 运行轻量级版本代码
  - 理解每个模块的功能
  - 在模拟数据上训练
- Week 7：
  - 尝试修改模型架构
  - 调整超参数
  - 可视化结果
- Week 8：
  - 运行完整版本（如有服务器）
  - 准备接入真实数据
  - 撰写技术文档

---

## 五、关键代码概念理解

### 5.1 滑动窗口示例
```python
# 将时间序列数据转换为训练样本
def create_sequences(data, input_length, output_length):
    """
    data: shape (总时间步, 特征数) 如(10000, 8)
    input_length: 输入序列长度（过去10小时）
    output_length: 输出序列长度（未来6小时）
    """
    X, y = [], []
    for i in range(len(data) - input_length - output_length):
        X.append(data[i:i+input_length])        # 过去的数据
        y.append(data[i+input_length:i+input_length+output_length])  # 未来的数据
    return np.array(X), np.array(y)
```

### 5.2 CNN-LSTM混合模型概念
```python
class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN部分：提取局部时序特征
        self.conv1d = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3)
        
        # LSTM部分：捕捉长期依赖
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2)
        
        # 全连接层：生成预测
        self.fc = nn.Linear(128, 8)  # 输出8个通道的预测
    
    def forward(self, x):
        # x shape: (batch, sequence, channels)
        x = self.conv1d(x.transpose(1, 2))  # CNN需要(batch, channels, sequence)
        x = self.lstm(x.transpose(1, 2))     # LSTM需要(sequence, batch, features)
        x = self.fc(x[0])                    # 取LSTM输出
        return x
```

### 5.3 训练循环框架
```python
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        # 前向传播
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    
    print(f'Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}')
```

---

## 六、常见问题与解决

### Q1：我的16GB内存够用吗？
**A**：够用的，但需要注意：
- 训练时关闭其他大型程序（浏览器少开标签页）
- 如果出现内存不足，减小batch_size
- 数据加载时使用`num_workers=2`（不要太大）

### Q2：训练很慢怎么办？
**A**：检查以下几点：
- 确认使用了GPU：`print(torch.cuda.is_available())`
- 数据已经放到GPU上：`data = data.cuda()`
- 使用混合精度训练（可选）：`torch.cuda.amp`

### Q3：模型不收敛/过拟合怎么办？
**A**：
- **不收敛**：降低学习率、检查数据归一化、增加batch size
- **过拟合**：增加Dropout、使用L2正则化、获取更多数据

### Q4：如何判断模型性能好坏？
**A**：
- 对比baseline（简单模型如线性回归）
- 看验证集loss是否持续下降
- 可视化预测结果与真实值
- 计算MAE/RMSE指标

---

## 七、学习资源汇总

### 在线课程
1. 李宏毅深度学习：https://speech.ee.ntu.edu.tw/~hylee/ml/
2. 吴恩达深度学习：https://www.coursera.org/specializations/deep-learning
3. Fast.ai深度学习：https://www.fast.ai/

### 书籍推荐
1. 《深度学习》（花书）- Ian Goodfellow（理论深入）
2. 《动手学深度学习》- 李沐（PyTorch版，实战性强）⭐推荐
3. 《Python深度学习》- François Chollet

### 实战教程
1. PyTorch官方教程：https://pytorch.org/tutorials/
2. Kaggle时间序列竞赛：https://www.kaggle.com/competitions
3. Papers with Code（时间序列预测）：https://paperswithcode.com/task/time-series-forecasting

### 论文阅读（进阶）
1. "Attention Is All You Need" - Transformer原理
2. "Temporal Convolutional Networks" - TCN架构
3. "DeepAR: Probabilistic Forecasting" - 概率预测

---

## 八、项目成功标准

### 最小可行产品（MVP）
- [ ] 能够运行轻量级版本代码
- [ ] 在模拟数据上完成训练
- [ ] 理解代码每个部分的作用
- [ ] 能够调整基本超参数

### 进阶目标
- [ ] 在真实数据上训练模型
- [ ] 模型预测误差MAE < 合理阈值
- [ ] 能够识别"云来了"的特征（V_SW↓, V_LW↑）
- [ ] 实现实时预测功能

### 高级目标
- [ ] 尝试多种模型架构对比
- [ ] 实现注意力机制
- [ ] 部署到树莓派/服务器进行实时预测
- [ ] 撰写技术报告/PPT展示

---

## 九、下一步行动

### 立即开始（今天）
1. ✅ 阅读本文档，理解整体学习路径
2. 📺 继续学习李宏毅课程的CNN章节
3. 📁 查看`Code_1`文件夹，运行`generate_data.py`生成模拟数据

### 本周目标
1. 完成CNN和LSTM的理论学习
2. 运行轻量级版本的代码，理解数据流
3. 在模拟数据上完成一次完整训练

### 本月目标
1. 完成时间序列预测专题学习
2. 能够独立修改模型架构
3. 开始准备接入真实数据

---

## 十、激励与建议

### 你的优势
- ✅ 有PyTorch基础，上手快
- ✅ 有实际硬件环境（PC + GPU）
- ✅ 项目场景明确，有实际应用价值
- ✅ 已经在学习李宏毅课程，基础扎实

### 学习建议
1. **循序渐进**：不要跳步，先把基础打牢
2. **多动手**：理论要结合代码实践
3. **记录问题**：遇到bug/疑惑记录下来，逐个解决
4. **可视化**：多画图，理解数据和模型行为
5. **寻求帮助**：学校老师、同学、在线社区（Stack Overflow、知乎）

### 预期成果
完成这个学习路径后，你将能够：
- ✅ 理解并实现CNN-LSTM混合模型
- ✅ 处理多变量时间序列数据
- ✅ 训练和评估深度学习模型
- ✅ 为课程项目贡献核心技术模块
- ✅ 为未来的机器学习项目打下坚实基础

---

**最后提醒**：机器学习是一门实践性很强的学科，不要害怕犯错，多尝试、多实验。当你训练出第一个能预测云量的模型时，那种成就感会让这一切努力都值得！💪

祝学习顺利！如有问题，随时查阅代码中的README和注释。

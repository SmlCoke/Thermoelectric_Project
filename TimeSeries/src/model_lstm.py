"""
LSTM模型实现

该模块实现了基于LSTM的时间序列预测模型
"""

import torch
import torch.nn as nn
# 确保输出编码为UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')

class LSTMModel(nn.Module):
    """
    LSTM时间序列预测模型
    
    架构:
        Input -> LSTM -> Dropout -> Fully Connected -> Output
    
    输入: [batch_size, seq_len, input_size]
    输出: [batch_size, predict_steps, output_size]
    """
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=1, 
                 output_size=8, predict_steps=10, dropout=0.2):
        """
        初始化LSTM模型
        
        参数:
            input_size: int, 输入特征维度（8个电压通道）
            hidden_size: int, LSTM隐藏层大小
            num_layers: int, LSTM层数
            output_size: int, 输出特征维度（8个电压通道）
            predict_steps: int, 预测的未来时间步数
            dropout: float, Dropout比率（防止过拟合）
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.predict_steps = predict_steps
        self.dropout = dropout
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状: [batch, seq, feature]
            dropout=dropout if num_layers > 1 else 0  # 只在多层时使用dropout
        )
        
        # Dropout层（用于全连接层前）
        self.dropout_layer = nn.Dropout(dropout)
        
        # 全连接层：将LSTM输出映射到预测结果
        # 输入: [batch_size, hidden_size]
        # 输出: [batch_size, predict_steps * output_size]
        self.fc = nn.Linear(hidden_size, predict_steps * output_size)
    
    def forward(self, x, hidden=None):
        """
        前向传播
        
        参数:
            x: torch.Tensor, 输入序列, 形状 [batch_size, seq_len, input_size]
            hidden: tuple of torch.Tensor or None, 初始隐藏状态和细胞状态
                   (h_0, c_0) 其中:
                   h_0: [num_layers, batch_size, hidden_size]
                   c_0: [num_layers, batch_size, hidden_size]
                   如果为None，则自动初始化为零
        
        返回:
            output: torch.Tensor, 预测结果, 形状 [batch_size, predict_steps, output_size]
            hidden: tuple of torch.Tensor, 最终隐藏状态和细胞状态
                   (h_n, c_n) 其中:
                   h_n: [num_layers, batch_size, hidden_size]
                   c_n: [num_layers, batch_size, hidden_size]
        """
        batch_size = x.size(0)
        
        # 如果未提供初始隐藏状态，初始化为零
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # LSTM前向传播
        # lstm_out: [batch_size, seq_len, hidden_size]
        # hidden: (h_n, c_n)
        #   h_n: [num_layers, batch_size, hidden_size]
        #   c_n: [num_layers, batch_size, hidden_size]
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步的输出
        # last_output: [batch_size, hidden_size]
        last_output = lstm_out[:, -1, :]
        # fj: lstm_out维度: [batch_size, seq_len, hidden_size]
        # fj: 这个张量包含了模型在阅读输入序列时，每一个时间步（第1秒、第2秒...直到第60秒）产生的理解（隐藏状态）。
        # fj: 目的是 “获取对整个历史片段的最终总结”。
        # fj: 信息累积（Memory Accumulation）： LSTM 的核心机制是“记忆”。当它处理到第 60 个时间步时，它的隐藏状态不仅仅包含第 60 秒的信息，还通过内部的门控机制（遗忘门、输入门）保留了第 1 秒到第 59 秒中所有重要的历史信息。因此，最后一个时间步的状态，就是整个输入窗口（过去3小时/60个点）的浓缩精华。

        # Dropout
        last_output = self.dropout_layer(last_output)
        
        # 全连接层
        # fc_out: [batch_size, predict_steps * output_size]
        fc_out = self.fc(last_output)
        
        # 重塑为 [batch_size, predict_steps, output_size]
        output = fc_out.view(batch_size, self.predict_steps, self.output_size)
        
        return output, hidden
    
    def _init_hidden(self, batch_size, device):
        """
        初始化隐藏状态和细胞状态为零
        
        参数:
            batch_size: int, 批次大小
            device: torch.device, 设备（CPU或GPU）
        
        返回:
            hidden: tuple of torch.Tensor, (h_0, c_0)
                   h_0: [num_layers, batch_size, hidden_size]
                   c_0: [num_layers, batch_size, hidden_size]
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)
    
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


class LSTMEncoder(nn.Module):
    """
    LSTM编码器（可选的高级架构）
    
    这是一个更复杂的架构，使用编码器-解码器结构
    适合更长期的预测任务
    """
    
    def __init__(self, input_size=8, hidden_size=128, num_layers=1, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x):
        """
        编码输入序列
        
        参数:
            x: [batch_size, seq_len, input_size]
        
        返回:
            outputs: [batch_size, seq_len, hidden_size]
            hidden: tuple (h_n, c_n)
                   h_n: [num_layers, batch_size, hidden_size]
                   c_n: [num_layers, batch_size, hidden_size]
        """
        outputs, hidden = self.lstm(x)
        return outputs, hidden


class LSTMDecoder(nn.Module):
    """
    LSTM解码器（可选的高级架构）
    """
    
    def __init__(self, hidden_size=128, output_size=8, num_layers=1, dropout=0.2):
        super(LSTMDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        """
        解码生成预测
        
        参数:
            x: [batch_size, 1, output_size] - 上一步的输出或初始输入
            hidden: tuple (h, c)
                   h: [num_layers, batch_size, hidden_size]
                   c: [num_layers, batch_size, hidden_size]
        
        返回:
            output: [batch_size, output_size]
            hidden: tuple (h, c)
        """
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("测试LSTM模型")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    model = LSTMModel(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        output_size=8,
        predict_steps=10,
        dropout=0.2
    ).to(device)
    
    print(f"\n模型架构:")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    batch_size = 32
    seq_len = 60
    input_size = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output, hidden = model(x)
    h_n, c_n = hidden
    print(f"输出形状: {output.shape}")
    print(f"隐藏状态形状: {h_n.shape}")
    print(f"细胞状态形状: {c_n.shape}")
    
    # 测试多步预测
    predictions = model.predict_multi_step(x, steps=20)
    print(f"\n多步预测形状: {predictions.shape}")
    
    print("\nLSTM模型测试成功!")

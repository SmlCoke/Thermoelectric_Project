"""
高级CNN-LSTM模型 - 完整版本
包含注意力机制和更深的网络结构
"""

import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.query(x)  # (batch, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final linear projection
        out = self.fc_out(out)
        
        return out


class AdvancedCNN_LSTM_Forecaster(nn.Module):
    """
    高级CNN-LSTM时间序列预测模型
    包含更深的网络、注意力机制、残差连接等
    """
    def __init__(self, config):
        super(AdvancedCNN_LSTM_Forecaster, self).__init__()
        
        self.config = config
        self.num_channels = config.NUM_CHANNELS
        self.input_length = config.INPUT_LENGTH
        self.output_length = config.OUTPUT_LENGTH
        
        # === 1. 更深的CNN层 ===
        self.conv1 = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=config.CNN_CHANNELS // 2,
            kernel_size=7,
            padding=3
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.CNN_CHANNELS // 2,
            out_channels=config.CNN_CHANNELS,
            kernel_size=5,
            padding=2
        )
        self.conv3 = nn.Conv1d(
            in_channels=config.CNN_CHANNELS,
            out_channels=config.CNN_CHANNELS,
            kernel_size=3,
            padding=1
        )
        
        self.bn1 = nn.BatchNorm1d(config.CNN_CHANNELS // 2)
        self.bn2 = nn.BatchNorm1d(config.CNN_CHANNELS)
        self.bn3 = nn.BatchNorm1d(config.CNN_CHANNELS)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(config.DROPOUT_RATE)
        
        # === 2. 更深的LSTM层 ===
        self.lstm = nn.LSTM(
            input_size=config.CNN_CHANNELS,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT_RATE if config.LSTM_NUM_LAYERS > 1 else 0,
            bidirectional=False
        )
        
        # === 3. 注意力机制（可选）===
        self.use_attention = config.USE_ATTENTION
        if self.use_attention:
            self.attention = MultiHeadAttention(
                hidden_size=config.LSTM_HIDDEN_SIZE,
                num_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT_RATE
            )
            self.attention_norm = nn.LayerNorm(config.LSTM_HIDDEN_SIZE)
        
        # === 4. 更深的全连接层 ===
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        self.fc1 = nn.Linear(config.LSTM_HIDDEN_SIZE, config.LSTM_HIDDEN_SIZE * 2)
        self.fc2 = nn.Linear(config.LSTM_HIDDEN_SIZE * 2, config.LSTM_HIDDEN_SIZE)
        self.fc3 = nn.Linear(config.LSTM_HIDDEN_SIZE, config.OUTPUT_LENGTH * self.num_channels)
        
        self.bn_fc1 = nn.BatchNorm1d(config.LSTM_HIDDEN_SIZE * 2)
        self.bn_fc2 = nn.BatchNorm1d(config.LSTM_HIDDEN_SIZE)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，shape = (batch_size, input_length, num_channels)
        Returns:
            output: 预测张量，shape = (batch_size, output_length, num_channels)
        """
        batch_size = x.size(0)
        
        # === CNN特征提取 ===
        # 转换为CNN所需的格式: (batch, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)
        
        # 第三个卷积块（残差连接）
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity  # 残差连接
        x = self.relu(x)
        
        # 池化降维
        x = self.pool(x)
        
        # === LSTM时序建模 ===
        # 转换回LSTM所需格式: (batch, sequence_length, features)
        x = x.transpose(1, 2)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)
        
        # === 注意力机制（可选）===
        if self.use_attention:
            # 应用多头注意力
            attn_out = self.attention(lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)  # 残差连接 + LayerNorm
        
        # 取LSTM最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # shape: (batch, hidden_size)
        
        # === 全连接层生成预测 ===
        x = self.dropout(last_output)
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # 重塑为 (batch, output_length, num_channels)
        output = x.view(batch_size, self.output_length, self.num_channels)
        
        return output
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    from config import Config
    
    print("测试高级模型架构...")
    config = Config()
    model = AdvancedCNN_LSTM_Forecaster(config)
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, config.INPUT_LENGTH, config.NUM_CHANNELS)
    
    # 前向传播
    output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {model.count_parameters():,}")
    print(f"使用注意力机制: {config.USE_ATTENTION}")
    print("模型测试通过！")

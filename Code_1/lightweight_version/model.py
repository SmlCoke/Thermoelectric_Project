"""
CNN-LSTM混合模型 - 轻量级版本
模型架构：
1. 1D CNN层：提取局部时序特征
2. LSTM层：捕捉长期时间依赖
3. 全连接层：生成多步预测
"""

import torch
import torch.nn as nn

class CNN_LSTM_Forecaster(nn.Module):
    """
    CNN-LSTM时间序列预测模型
    """
    def __init__(self, config):
        super(CNN_LSTM_Forecaster, self).__init__()
        
        self.config = config
        self.num_channels = config.NUM_CHANNELS
        self.input_length = config.INPUT_LENGTH
        self.output_length = config.OUTPUT_LENGTH
        
        # 1D卷积层（提取局部特征）
        self.conv1 = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=config.CNN_CHANNELS,
            kernel_size=7,
            padding=3
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.CNN_CHANNELS,
            out_channels=config.CNN_CHANNELS,
            kernel_size=5,
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(config.CNN_CHANNELS)
        self.bn2 = nn.BatchNorm1d(config.CNN_CHANNELS)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM层（捕捉时序依赖）
        # 注意：池化后序列长度会变为 input_length // 2
        self.lstm = nn.LSTM(
            input_size=config.CNN_CHANNELS,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT_RATE if config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # 全连接层（生成预测）
        self.fc1 = nn.Linear(config.LSTM_HIDDEN_SIZE, config.LSTM_HIDDEN_SIZE)
        self.fc2 = nn.Linear(config.LSTM_HIDDEN_SIZE, config.OUTPUT_LENGTH * self.num_channels)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，shape = (batch_size, input_length, num_channels)
        Returns:
            output: 预测张量，shape = (batch_size, output_length, num_channels)
        """
        batch_size = x.size(0)
        
        # 转换为CNN所需的格式: (batch, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 池化降维
        x = self.pool(x)
        
        # 转换回LSTM所需格式: (batch, sequence_length, features)
        x = x.transpose(1, 2)
        
        # LSTM时序建模
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取LSTM最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # shape: (batch, hidden_size)
        
        # 全连接层生成预测
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 重塑为 (batch, output_length, num_channels)
        output = x.view(batch_size, self.output_length, self.num_channels)
        
        return output
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    from config import Config
    
    print("测试模型架构...")
    config = Config()
    model = CNN_LSTM_Forecaster(config)
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, config.INPUT_LENGTH, config.NUM_CHANNELS)
    
    # 前向传播
    output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {model.count_parameters():,}")
    print("模型测试通过！")

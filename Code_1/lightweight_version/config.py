"""
配置文件 - 轻量级版本
适用于个人PC（NVIDIA 4060 + 16GB RAM）
"""

import torch

class Config:
    # 数据参数
    NUM_CHANNELS = 8  # 8个通道：VSW黑/红/蓝/绿, VLW黑/红/蓝/绿
    INPUT_HOURS = 10   # 输入：过去10小时
    OUTPUT_HOURS = 6   # 输出：未来6小时
    SAMPLE_INTERVAL = 10  # 采样间隔（秒）
    
    # 计算时间步数（每10秒采样一次）
    INPUT_LENGTH = INPUT_HOURS * 3600 // SAMPLE_INTERVAL  # 3600步
    OUTPUT_LENGTH = OUTPUT_HOURS * 3600 // SAMPLE_INTERVAL  # 2160步
    
    # 模型参数（轻量级）
    CNN_CHANNELS = 32      # CNN输出通道数（较小）
    LSTM_HIDDEN_SIZE = 64  # LSTM隐藏层大小（较小）
    LSTM_NUM_LAYERS = 2    # LSTM层数
    DROPOUT_RATE = 0.2     # Dropout比率
    
    # 训练参数
    BATCH_SIZE = 16        # 批次大小（适合16GB内存）
    NUM_EPOCHS = 50        # 训练轮数
    LEARNING_RATE = 0.001  # 初始学习率
    WEIGHT_DECAY = 1e-5    # L2正则化系数
    
    # 数据划分
    TRAIN_RATIO = 0.7      # 训练集比例
    VAL_RATIO = 0.15       # 验证集比例
    TEST_RATIO = 0.15      # 测试集比例
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 文件路径
    DATA_PATH = 'data/simulated_data.npz'
    MODEL_SAVE_PATH = 'checkpoints/best_model.pth'
    LOG_DIR = 'logs/'
    
    # 其他
    SEED = 42              # 随机种子
    PRINT_FREQ = 10        # 打印频率（每多少批次打印一次）
    
    @staticmethod
    def print_config():
        """打印配置信息"""
        print("=" * 50)
        print("轻量级版本配置")
        print("=" * 50)
        print(f"数据参数:")
        print(f"  - 输入长度: {Config.INPUT_LENGTH} 步 ({Config.INPUT_HOURS} 小时)")
        print(f"  - 输出长度: {Config.OUTPUT_LENGTH} 步 ({Config.OUTPUT_HOURS} 小时)")
        print(f"  - 通道数: {Config.NUM_CHANNELS}")
        print(f"\n模型参数:")
        print(f"  - CNN通道: {Config.CNN_CHANNELS}")
        print(f"  - LSTM隐藏层: {Config.LSTM_HIDDEN_SIZE}")
        print(f"  - LSTM层数: {Config.LSTM_NUM_LAYERS}")
        print(f"\n训练参数:")
        print(f"  - 批次大小: {Config.BATCH_SIZE}")
        print(f"  - 训练轮数: {Config.NUM_EPOCHS}")
        print(f"  - 学习率: {Config.LEARNING_RATE}")
        print(f"\n设备: {Config.DEVICE}")
        print("=" * 50)

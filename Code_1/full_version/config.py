"""
配置文件 - 完整版本
适用于服务器（大显存GPU，例如V100, A100, RTX 3090等）
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
    
    # 模型参数（完整版 - 更大更深）
    CNN_CHANNELS = 128     # CNN输出通道数（更大）
    LSTM_HIDDEN_SIZE = 256  # LSTM隐藏层大小（更大）
    LSTM_NUM_LAYERS = 4     # LSTM层数（更多）
    DROPOUT_RATE = 0.3      # Dropout比率（更高）
    USE_ATTENTION = True    # 是否使用注意力机制
    ATTENTION_HEADS = 8     # 注意力头数
    
    # 训练参数
    BATCH_SIZE = 64         # 批次大小（更大）
    NUM_EPOCHS = 100        # 训练轮数（更多）
    LEARNING_RATE = 0.0005  # 初始学习率
    WEIGHT_DECAY = 1e-4     # L2正则化系数
    GRADIENT_CLIP = 1.0     # 梯度裁剪阈值
    
    # 学习率调度
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'  # 'reduce_on_plateau', 'cosine', 'step'
    WARMUP_EPOCHS = 5          # 预热轮数
    MIN_LR = 1e-6              # 最小学习率
    
    # 数据增强
    USE_DATA_AUGMENTATION = True
    NOISE_LEVEL = 0.01         # 添加的噪声水平
    
    # 混合精度训练
    USE_MIXED_PRECISION = True  # 使用FP16加速训练
    
    # 数据划分
    TRAIN_RATIO = 0.7      # 训练集比例
    VAL_RATIO = 0.15       # 验证集比例
    TEST_RATIO = 0.15      # 测试集比例
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MULTI_GPU = torch.cuda.device_count() > 1  # 是否使用多GPU
    
    # 文件路径
    DATA_PATH = 'data/simulated_data_extended.npz'
    MODEL_SAVE_PATH = 'checkpoints/best_model.pth'
    LOG_DIR = 'logs/'
    
    # 模型集成
    USE_ENSEMBLE = False    # 是否使用模型集成
    NUM_ENSEMBLE_MODELS = 3  # 集成模型数量
    
    # 其他
    SEED = 42              # 随机种子
    PRINT_FREQ = 5         # 打印频率（每多少批次打印一次）
    SAVE_FREQ = 10         # 保存检查点频率（每多少个epoch）
    
    @staticmethod
    def print_config():
        """打印配置信息"""
        print("=" * 60)
        print("完整版本配置（服务器级）")
        print("=" * 60)
        print(f"数据参数:")
        print(f"  - 输入长度: {Config.INPUT_LENGTH} 步 ({Config.INPUT_HOURS} 小时)")
        print(f"  - 输出长度: {Config.OUTPUT_LENGTH} 步 ({Config.OUTPUT_HOURS} 小时)")
        print(f"  - 通道数: {Config.NUM_CHANNELS}")
        print(f"\n模型参数:")
        print(f"  - CNN通道: {Config.CNN_CHANNELS}")
        print(f"  - LSTM隐藏层: {Config.LSTM_HIDDEN_SIZE}")
        print(f"  - LSTM层数: {Config.LSTM_NUM_LAYERS}")
        print(f"  - 使用注意力: {Config.USE_ATTENTION}")
        if Config.USE_ATTENTION:
            print(f"  - 注意力头数: {Config.ATTENTION_HEADS}")
        print(f"\n训练参数:")
        print(f"  - 批次大小: {Config.BATCH_SIZE}")
        print(f"  - 训练轮数: {Config.NUM_EPOCHS}")
        print(f"  - 学习率: {Config.LEARNING_RATE}")
        print(f"  - 混合精度: {Config.USE_MIXED_PRECISION}")
        print(f"\n设备:")
        print(f"  - 主设备: {Config.DEVICE}")
        if Config.MULTI_GPU:
            print(f"  - GPU数量: {torch.cuda.device_count()}")
        print("=" * 60)

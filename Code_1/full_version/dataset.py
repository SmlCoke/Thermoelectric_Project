"""
数据集加载和预处理 - 完整版本
包含数据增强等高级功能
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类（带数据增强）
    """
    def __init__(self, data, input_length, output_length, augment=False, noise_level=0.01):
        """
        Args:
            data: numpy数组，shape = (total_timesteps, num_channels)
            input_length: 输入序列长度
            output_length: 输出序列长度
            augment: 是否进行数据增强
            noise_level: 噪声水平
        """
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.augment = augment
        self.noise_level = noise_level
        
        # 计算可以生成多少个样本
        self.num_samples = len(data) - input_length - output_length + 1
        
        if self.num_samples <= 0:
            raise ValueError(f"数据太短！需要至少 {input_length + output_length} 个时间步")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        获取一个训练样本
        Returns:
            x: 输入序列，shape = (input_length, num_channels)
            y: 目标序列，shape = (output_length, num_channels)
        """
        # 输入：从idx开始的input_length个时间步
        x = self.data[idx : idx + self.input_length].copy()
        
        # 输出：紧接着的output_length个时间步
        y = self.data[idx + self.input_length : idx + self.input_length + self.output_length].copy()
        
        # 数据增强（仅在训练时）
        if self.augment:
            # 添加随机噪声
            x += np.random.normal(0, self.noise_level, x.shape)
            
            # 随机缩放（模拟不同的辐射强度）
            scale = np.random.uniform(0.95, 1.05)
            x *= scale
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


def load_and_preprocess_data(config, data_path=None):
    """
    加载并预处理数据
    
    Args:
        config: 配置对象
        data_path: 数据文件路径
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    print(f"正在加载数据：{data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}\n"
                              f"请先运行 generate_data.py 生成模拟数据")
    
    # 加载数据
    data_dict = np.load(data_path)
    raw_data = data_dict['data']  # shape: (total_timesteps, num_channels)
    
    print(f"数据形状: {raw_data.shape}")
    print(f"时间步数: {raw_data.shape[0]}, 通道数: {raw_data.shape[1]}")
    
    # 数据归一化（每个通道独立归一化）
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(raw_data)
    
    print("数据归一化完成")
    
    # 划分训练集、验证集、测试集
    total_samples = len(normalized_data) - config.INPUT_LENGTH - config.OUTPUT_LENGTH + 1
    
    train_size = int(total_samples * config.TRAIN_RATIO)
    val_size = int(total_samples * config.VAL_RATIO)
    
    # 注意：时间序列数据不能随机打乱，要按时间顺序划分
    train_data = normalized_data[:train_size + config.INPUT_LENGTH + config.OUTPUT_LENGTH - 1]
    val_data = normalized_data[train_size:train_size + val_size + config.INPUT_LENGTH + config.OUTPUT_LENGTH - 1]
    test_data = normalized_data[train_size + val_size:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_data)} 个时间步")
    print(f"  验证集: {len(val_data)} 个时间步")
    print(f"  测试集: {len(test_data)} 个时间步")
    
    # 创建数据集（训练集启用数据增强）
    train_dataset = TimeSeriesDataset(
        train_data, 
        config.INPUT_LENGTH, 
        config.OUTPUT_LENGTH,
        augment=config.USE_DATA_AUGMENTATION,
        noise_level=config.NOISE_LEVEL
    )
    val_dataset = TimeSeriesDataset(
        val_data, 
        config.INPUT_LENGTH, 
        config.OUTPUT_LENGTH,
        augment=False
    )
    test_dataset = TimeSeriesDataset(
        test_data, 
        config.INPUT_LENGTH, 
        config.OUTPUT_LENGTH,
        augment=False
    )
    
    print(f"\n样本数量:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  测试样本: {len(test_dataset)}")
    
    # 创建数据加载器
    # 服务器版本可以使用更多的workers
    num_workers = 4 if config.DEVICE.type == 'cuda' else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    # 测试数据加载
    from config import Config
    
    config = Config()
    
    print("测试数据加载器...")
    try:
        train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(config)
        
        # 测试一个批次
        for batch_x, batch_y in train_loader:
            print(f"\n批次数据形状:")
            print(f"  输入X: {batch_x.shape}")
            print(f"  目标Y: {batch_y.shape}")
            break
        
        print("\n数据加载测试通过！")
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先运行 'python generate_data.py' 生成数据")

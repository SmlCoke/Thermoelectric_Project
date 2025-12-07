"""
时间序列数据加载模块

该模块负责：
1. 读取多个CSV文件（每个文件代表一天的测量数据）
2. 将数据按日期划分为独立片段
3. 使用滑动窗口从每个片段提取训练样本
4. 数据标准化处理
5. 构建PyTorch Dataset和DataLoader
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle


class ThermoelectricDataset(Dataset):
    """
    热电芯片时间序列数据集
    
    特点：
    - 支持多个独立的时间序列片段（每天一个片段）
    - 使用滑动窗口提取固定长度的序列样本
    - 不进行跨日期的数据连接或插值
    """
    
    def __init__(self, data_dir, window_size=60, predict_steps=10, 
                 stride=1, normalize=True, train_ratio=0.8):
        """
        初始化数据集
        
        参数:
            data_dir: str, CSV文件所在目录
            window_size: int, 输入序列的长度（时间步数）
            predict_steps: int, 预测的未来时间步数
            stride: int, 滑动窗口的步长
            normalize: bool, 是否进行数据标准化
            train_ratio: float, 训练集比例（用于划分训练/验证集）
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.predict_steps = predict_steps
        self.stride = stride
        self.normalize = normalize
        self.train_ratio = train_ratio
        
        # 数据列名（8个热电芯片通道）
        self.voltage_columns = [
            'TEC1_Optimal(V)', 'TEC2_Optimal(V)', 'TEC3_Optimal(V)', 'TEC4_Optimal(V)',
            'TEC5_Optimal(V)', 'TEC6_Optimal(V)', 'TEC7_Optimal(V)', 'TEC8_Optimal(V)'
        ]
        
        # 加载所有数据片段
        self.segments = self._load_segments()
        
        # 数据标准化
        self.scaler = None
        if self.normalize:
            self._fit_scaler()
            self._normalize_segments()
        
        # 从片段中提取样本
        self.samples = self._extract_samples()
        
        print(f"数据集初始化完成:")
        print(f"  - 片段数量: {len(self.segments)}")
        print(f"  - 样本总数: {len(self.samples)}")
        print(f"  - 窗口大小: {self.window_size}")
        print(f"  - 预测步数: {self.predict_steps}")
    
    def _load_segments(self):
        """
        加载所有CSV文件，每个文件代表一个独立的时间序列片段
        
        返回:
            segments: list of numpy arrays, 每个数组形状为 [seq_len, 8]
        """
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, '*.csv')))
        
        if len(csv_files) == 0:
            raise ValueError(f"在目录 {self.data_dir} 中未找到CSV文件")
        
        segments = []
        for csv_file in csv_files:
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                
                # 提取电压数据（8个通道）
                voltage_data = df[self.voltage_columns].values
                
                # 检查数据有效性
                if len(voltage_data) < self.window_size + self.predict_steps:
                    print(f"警告: {os.path.basename(csv_file)} 数据点过少，已跳过")
                    continue
                
                segments.append(voltage_data)
                print(f"加载 {os.path.basename(csv_file)}: {voltage_data.shape}")
                
            except Exception as e:
                print(f"警告: 无法加载 {csv_file}: {e}")
                continue
        
        if len(segments) == 0:
            raise ValueError("没有成功加载任何数据片段")
        
        return segments
    
    def _fit_scaler(self):
        """
        在所有数据上拟合标准化器
        
        注意：虽然片段之间不连续，但我们仍然希望所有数据使用相同的标准化参数
        """
        # 将所有片段的数据合并（仅用于计算统计量）
        all_data = np.vstack(self.segments)
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_data)
        
        print(f"标准化器已拟合:")
        print(f"  - 均值: {self.scaler.mean_}")
        print(f"  - 标准差: {self.scaler.scale_}")
    
    def _normalize_segments(self):
        """
        对所有片段进行标准化
        """
        self.segments = [self.scaler.transform(seg) for seg in self.segments]
    
    def _extract_samples(self):
        """
        使用滑动窗口从每个片段中提取训练样本
        
        对于每个片段，我们提取多个 (输入序列, 目标序列) 对：
        - 输入序列: [window_size, 8]
        - 目标序列: [predict_steps, 8]
        
        返回:
            samples: list of tuples (x, y)
        """
        samples = []
        
        for segment_idx, segment in enumerate(self.segments):
            seq_len = len(segment)
            
            # 滑动窗口提取样本
            for i in range(0, seq_len - self.window_size - self.predict_steps + 1, self.stride):
                # 输入: 当前窗口
                x = segment[i:i + self.window_size]
                
                # 目标: 紧接着的未来步
                y = segment[i + self.window_size:i + self.window_size + self.predict_steps]
                
                samples.append((x, y, segment_idx))
        
        return samples
    
    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        返回:
            x: torch.Tensor, 形状 [window_size, 8]
            y: torch.Tensor, 形状 [predict_steps, 8]
        """
        x, y, segment_idx = self.samples[idx]
        
        # 转换为PyTorch张量
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y
    
    def split_train_val(self):
        """
        划分训练集和验证集
        
        策略：按照片段划分，保证验证集的片段完全独立
        
        返回:
            train_dataset: ThermoelectricDataset
            val_dataset: ThermoelectricDataset
        """
        num_segments = len(self.segments)
        num_train_segments = int(num_segments * self.train_ratio)
        
        # 随机打乱片段索引（使用固定种子以保证可重复性）
        rng = np.random.RandomState(42)
        segment_indices = rng.permutation(num_segments)
        train_indices = set(segment_indices[:num_train_segments])
        val_indices = set(segment_indices[num_train_segments:])
        
        # 分别提取训练和验证样本
        train_samples = [s for s in self.samples if s[2] in train_indices]
        val_samples = [s for s in self.samples if s[2] in val_indices]
        
        # 创建新的数据集对象
        train_dataset = ThermoelectricDataset.__new__(ThermoelectricDataset)
        train_dataset.samples = train_samples
        train_dataset.scaler = self.scaler
        train_dataset.normalize = self.normalize
        
        val_dataset = ThermoelectricDataset.__new__(ThermoelectricDataset)
        val_dataset.samples = val_samples
        val_dataset.scaler = self.scaler
        val_dataset.normalize = self.normalize
        
        print(f"\n数据集划分:")
        print(f"  - 训练集: {len(train_samples)} 样本")
        print(f"  - 验证集: {len(val_samples)} 样本")
        
        return train_dataset, val_dataset
    
    def inverse_transform(self, data):
        """
        反标准化：将标准化后的数据转换回原始尺度
        
        参数:
            data: numpy array or torch.Tensor
            
        返回:
            原始尺度的数据
        """
        if self.scaler is None:
            return data
        
        # 如果是PyTorch张量，先转换为numpy
        is_tensor = torch.is_tensor(data)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
        
        # 反标准化
        original_shape = data.shape
        data_2d = data.reshape(-1, 8)
        data_2d = self.scaler.inverse_transform(data_2d)
        data = data_2d.reshape(original_shape)
        
        # 如果原来是张量，转换回去
        if is_tensor:
            data = torch.FloatTensor(data).to(device)
        
        return data
    
    def save_scaler(self, path):
        """保存标准化器"""
        if self.scaler is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"标准化器已保存到: {path}")
    
    def load_scaler(self, path):
        """加载标准化器"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"标准化器已从 {path} 加载")


def create_dataloaders(data_dir, batch_size=32, window_size=60, predict_steps=10,
                       stride=1, normalize=True, train_ratio=0.8, num_workers=0):
    """
    创建训练和验证的DataLoader
    
    参数:
        data_dir: str, CSV文件所在目录
        batch_size: int, 批次大小
        window_size: int, 输入序列长度
        predict_steps: int, 预测步数
        stride: int, 滑动窗口步长
        normalize: bool, 是否标准化
        train_ratio: float, 训练集比例
        num_workers: int, 数据加载的并行工作进程数
    
    返回:
        train_loader: DataLoader
        val_loader: DataLoader
        dataset: ThermoelectricDataset (原始数据集，包含scaler等信息)
    """
    # 创建数据集
    dataset = ThermoelectricDataset(
        data_dir=data_dir,
        window_size=window_size,
        predict_steps=predict_steps,
        stride=stride,
        normalize=normalize,
        train_ratio=train_ratio
    )
    
    # 划分训练集和验证集
    train_dataset, val_dataset = dataset.split_train_val()
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱顺序
        num_workers=num_workers,
        pin_memory=True  # 加速GPU数据传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset


# 测试代码
if __name__ == '__main__':
    # 测试数据加载
    print("=" * 60)
    print("测试数据集加载")
    print("=" * 60)
    
    # 假设CSV文件在上级目录
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    
    try:
        # 创建数据集
        dataset = ThermoelectricDataset(
            data_dir=data_dir,
            window_size=60,
            predict_steps=10,
            stride=5,
            normalize=True,
            train_ratio=0.8
        )
        
        # 测试获取样本
        print("\n测试样本:")
        x, y = dataset[0]
        print(f"输入形状: {x.shape}")  # [60, 8]
        print(f"目标形状: {y.shape}")  # [10, 8]
        print(f"输入数据范围: [{x.min():.4f}, {x.max():.4f}]")
        
        # 测试DataLoader
        print("\n测试DataLoader:")
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=data_dir,
            batch_size=32,
            window_size=60,
            predict_steps=10
        )
        
        # 获取一个批次
        for batch_x, batch_y in train_loader:
            print(f"批次输入形状: {batch_x.shape}")  # [32, 60, 8]
            print(f"批次目标形状: {batch_y.shape}")  # [32, 10, 8]
            break
        
        print("\n数据集测试成功!")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

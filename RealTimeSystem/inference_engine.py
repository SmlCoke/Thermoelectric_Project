"""
推理引擎模块

该模块负责：
1. 加载训练好的时间序列模型
2. 对滑动窗口数据进行推理
3. 支持 1-step 和 10-step 预测模式

使用方式：
    from inference_engine import InferenceEngine
    
    engine = InferenceEngine(model_path="path/to/model.pth")
    predictions = engine.predict(input_data, steps=10)
"""

import os
import sys
import logging
import pickle
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch

# [修复] 限制 PyTorch 线程数
# 在多线程环境（Flask + GUI）中，PyTorch 的 OpenMP 线程可能会导致死锁或 CPU 饥饿
# 设置为 1 可以避免这些问题，且对于这种小规模数据的推理，性能影响微乎其微
torch.set_num_threads(1)

# 添加 TimeSeries/src 到路径（如果存在）
_timeseries_src_path = os.path.join(os.path.dirname(__file__), '..', 'TimeSeries', 'src')
if os.path.exists(_timeseries_src_path):
    sys.path.insert(0, _timeseries_src_path)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据结构"""
    predictions: np.ndarray  # 预测值 [steps, 8]
    steps: int               # 预测步数
    input_seq_len: int       # 输入序列长度
    timestamp: str           # 预测时间戳
    
    def get_channel(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的预测值"""
        return self.predictions[:, channel_idx]
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'predictions': self.predictions.tolist(),
            'steps': self.steps,
            'input_seq_len': self.input_seq_len,
            'timestamp': self.timestamp
        }


class InferenceEngine:
    """
    推理引擎类
    
    负责加载模型并执行时间序列预测
    """
    
    # 通道名称
    CHANNEL_NAMES = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',  # [修改] 默认改为 'cpu'，避免多线程 CUDA 冲突
        scaler_path: Optional[str] = None
    ):
        """
        初始化推理引擎
        
        参数:
            model_path: str, 模型检查点路径
            device: str, 计算设备 ('cpu' 推荐用于多线程环境)
            scaler_path: str, 标准化器路径 (可选，默认从模型目录查找)
        """
        # [修改] 强制使用 CPU，除非显式指定其他
        # 在 Flask + PyQt5 多线程环境中，CUDA 容易导致死锁
        self.device = torch.device(device)
        
        logger.info(f"推理引擎初始化")
        logger.info(f"  设备: {self.device}")
        
        # 加载模型
        self.model_path = model_path
        self.model, self.config = self._load_model(model_path)
        
        # 加载标准化器
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        
        self.scaler = self._load_scaler(scaler_path)
        
        # 模型信息
        self.window_size = self.config.get('window_size', 60)
        self.predict_steps = self.config.get('predict_steps', 10)
        self.model_type = self.config.get('model_type', 'unknown')
        
        logger.info(f"  模型类型: {self.model_type.upper()}")
        logger.info(f"  窗口大小: {self.window_size}")
        logger.info(f"  默认预测步数: {self.predict_steps}")
        logger.info("推理引擎初始化完成")
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, dict]:
        """
        加载模型
        
        参数:
            model_path: str, 模型检查点路径
        
        返回:
            model: torch.nn.Module, 加载的模型
            config: dict, 模型配置
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint.get('config', {})
        model_type = config.get('model_type', 'lstm')
        
        # 创建模型
        if model_type == 'lstm':
            try:
                from model_lstm import LSTMModel
            except ImportError as e:
                raise ImportError(f"无法导入 LSTM 模型模块。请确保 TimeSeries/src 目录存在: {e}")
            model = LSTMModel(
                input_size=8,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 1),
                output_size=8,
                predict_steps=config.get('predict_steps', 10),
                dropout=config.get('dropout', 0.2)
            )
        elif model_type == 'gru':
            try:
                from model_gru import GRUModel
            except ImportError as e:
                raise ImportError(f"无法导入 GRU 模型模块。请确保 TimeSeries/src 目录存在: {e}")
            model = GRUModel(
                input_size=8,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 1),
                output_size=8,
                predict_steps=config.get('predict_steps', 10),
                dropout=config.get('dropout', 0.2)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"模型加载成功")
        logger.info(f"  训练 epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  验证损失: {checkpoint.get('val_loss', 'N/A'):.6f}")
        
        return model, config
    
    def _load_scaler(self, scaler_path: str) -> Optional[object]:
        """
        加载标准化器
        
        参数:
            scaler_path: str, 标准化器路径
        
        返回:
            scaler: StandardScaler or None
        """
        # 1. 尝试直接加载指定路径
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"标准化器已加载: {scaler_path}")
                return scaler
            except Exception as e:
                logger.error(f"加载标准化器失败: {e}")
        
        # 2. [新增] 尝试在模型路径的父目录查找 (常见情况)
        # 如果 model_path 是 .../checkpoints/best_model.pth
        # 我们希望查找 .../scaler.pkl
        model_dir = os.path.dirname(self.model_path)
        parent_dir = os.path.dirname(model_dir)
        parent_scaler_path = os.path.join(parent_dir, 'scaler.pkl')
        
        if os.path.exists(parent_scaler_path):
            try:
                with open(parent_scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"在父目录找到并加载标准化器: {parent_scaler_path}")
                return scaler
            except Exception as e:
                logger.error(f"加载父目录标准化器失败: {e}")

        logger.warning(f"未找到标准化器 (尝试路径: {scaler_path}, {parent_scaler_path})")
        logger.warning("!!! 警告: 预测结果将是原始模型输出（未反标准化），数值可能接近 0 !!!")
        return None
    
    def predict(
        self,
        input_data: np.ndarray,
        steps: int = 1,
        timestamp: str = ""
    ) -> PredictionResult:
        """
        执行预测
        
        参数:
            input_data: np.ndarray, 输入数据 [seq_len, 8]
            steps: int, 预测步数 (1 或 10)
            timestamp: str, 时间戳标记
        
        返回:
            PredictionResult, 预测结果
        """
        # 验证输入
        if input_data.ndim != 2 or input_data.shape[1] != 8:
            raise ValueError(f"输入数据形状错误: 期望 [seq_len, 8], 实际 {input_data.shape}")
        
        if input_data.shape[0] < self.window_size:
            raise ValueError(f"输入序列过短: 期望 >= {self.window_size}, 实际 {input_data.shape[0]}")
        
        # 截取最后 window_size 个时间步
        input_seq = input_data[-self.window_size:].copy()
        
        # 标准化
        if self.scaler is not None:
            input_seq = self.scaler.transform(input_seq)
        
        # 转换为 tensor
        x = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)  # [1, seq_len, 8]
        
        # 执行预测
        with torch.no_grad():
            if steps == 1:
                # 单步预测：取模型输出的第一步
                predictions, _ = self.model(x)
                predictions = predictions[:, 0:1, :]  # [1, 1, 8]
            elif steps <= self.predict_steps:
                # 多步预测（在模型训练范围内）
                predictions, _ = self.model(x)
                predictions = predictions[:, :steps, :]  # [1, steps, 8]
            else:
                # 超过模型训练步数的预测：使用迭代方法
                predictions = self.model.predict_multi_step(x, steps)
        
        # 转换回 numpy
        predictions = predictions.squeeze(0).cpu().numpy()  # [steps, 8]
        
        # 反标准化
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        return PredictionResult(
            predictions=predictions,
            steps=steps,
            input_seq_len=input_data.shape[0],
            timestamp=timestamp
        )
    
    def predict_single_step(
        self,
        input_data: np.ndarray,
        timestamp: str = ""
    ) -> PredictionResult:
        """
        单步预测快捷方法
        
        参数:
            input_data: np.ndarray, 输入数据 [seq_len, 8]
            timestamp: str, 时间戳标记
        
        返回:
            PredictionResult, 预测结果
        """
        return self.predict(input_data, steps=1, timestamp=timestamp)
    
    def predict_multi_step(
        self,
        input_data: np.ndarray,
        steps: int = 10,
        timestamp: str = ""
    ) -> PredictionResult:
        """
        多步预测快捷方法
        
        参数:
            input_data: np.ndarray, 输入数据 [seq_len, 8]
            steps: int, 预测步数
            timestamp: str, 时间戳标记
        
        返回:
            PredictionResult, 预测结果
        """
        return self.predict(input_data, steps=steps, timestamp=timestamp)
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        返回:
            dict, 模型信息
        """
        return {
            'model_type': self.model_type,
            'window_size': self.window_size,
            'predict_steps': self.predict_steps,
            'device': str(self.device),
            'model_path': self.model_path,
            'has_scaler': self.scaler is not None,
            'config': self.config
        }
    
    def get_channel_name(self, channel_idx: int) -> str:
        """
        获取通道名称
        
        参数:
            channel_idx: int, 通道索引 (0-7)
        
        返回:
            str, 通道名称
        """
        if 0 <= channel_idx < len(self.CHANNEL_NAMES):
            return self.CHANNEL_NAMES[channel_idx]
        return f"Channel_{channel_idx}"


class MockInferenceEngine:
    """
    模拟推理引擎 (用于测试)
    
    当没有训练好的模型时使用
    """
    
    CHANNEL_NAMES = InferenceEngine.CHANNEL_NAMES
    
    def __init__(self):
        """初始化模拟推理引擎"""
        self.window_size = 60
        self.predict_steps = 10
        self.model_type = 'mock'
        logger.info("使用模拟推理引擎 (仅用于测试)")
    
    def predict(
        self,
        input_data: np.ndarray,
        steps: int = 1,
        timestamp: str = ""
    ) -> PredictionResult:
        """
        执行模拟预测
        
        基于输入数据的最后几个值，添加少量随机噪声生成预测
        """
        # 获取最后的值作为基准
        last_values = input_data[-1]
        
        # 生成模拟预测（带有少量随机波动）
        predictions = np.zeros((steps, 8))
        for i in range(steps):
            noise = np.random.randn(8) * 0.02
            trend = (np.random.randn(8) * 0.005) * (i + 1)  # 轻微趋势
            predictions[i] = last_values + noise + trend
        
        return PredictionResult(
            predictions=predictions,
            steps=steps,
            input_seq_len=input_data.shape[0],
            timestamp=timestamp
        )
    
    def predict_single_step(self, input_data: np.ndarray, timestamp: str = "") -> PredictionResult:
        return self.predict(input_data, steps=1, timestamp=timestamp)
    
    def predict_multi_step(self, input_data: np.ndarray, steps: int = 10, timestamp: str = "") -> PredictionResult:
        return self.predict(input_data, steps=steps, timestamp=timestamp)
    
    def get_model_info(self) -> dict:
        return {
            'model_type': 'mock',
            'window_size': self.window_size,
            'predict_steps': self.predict_steps,
            'device': 'cpu',
            'model_path': 'N/A',
            'has_scaler': False
        }
    
    def get_channel_name(self, channel_idx: int) -> str:
        if 0 <= channel_idx < len(self.CHANNEL_NAMES):
            return self.CHANNEL_NAMES[channel_idx]
        return f"Channel_{channel_idx}"


def create_inference_engine(model_path: Optional[str] = None, **kwargs) -> Union[InferenceEngine, MockInferenceEngine]:
    """
    创建推理引擎的工厂函数
    
    参数:
        model_path: str or None, 模型路径 (None 时使用模拟引擎)
        **kwargs: 传递给 InferenceEngine 的其他参数
    
    返回:
        InferenceEngine or MockInferenceEngine
    """
    if model_path is None or not os.path.exists(model_path):
        logger.warning("未找到模型文件，使用模拟推理引擎")
        return MockInferenceEngine()
    
    try:
        return InferenceEngine(model_path, **kwargs)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.warning("降级使用模拟推理引擎")
        return MockInferenceEngine()


# 测试代码
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='推理引擎测试')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--test', action='store_true',
                       help='使用模拟数据进行测试')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("推理引擎测试")
    print("=" * 60)
    
    # 创建推理引擎
    engine = create_inference_engine(args.model_path)
    
    # 打印模型信息
    print("\n模型信息:")
    for key, value in engine.get_model_info().items():
        print(f"  {key}: {value}")
    
    # 创建测试数据
    print("\n创建测试数据...")
    test_data = np.random.randn(60, 8) * 0.1 + 0.5
    print(f"输入数据形状: {test_data.shape}")
    
    # 单步预测
    print("\n执行单步预测...")
    result1 = engine.predict_single_step(test_data, timestamp="test")
    print(f"预测结果形状: {result1.predictions.shape}")
    print(f"预测值:\n{result1.predictions}")
    
    # 多步预测
    print("\n执行10步预测...")
    result10 = engine.predict_multi_step(test_data, steps=10, timestamp="test")
    print(f"预测结果形状: {result10.predictions.shape}")
    print(f"预测值 (前3步):\n{result10.predictions[:3]}")
    
    print("\n测试完成!")

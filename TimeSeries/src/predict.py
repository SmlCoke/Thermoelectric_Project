"""
预测脚本

该模块负责：
1. 加载训练好的模型
2. 加载测试数据
3. 生成预测结果
4. 可视化预测结果
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
T_16 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 16)    # Times New Roman
T_14 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 14)    # Times New Roman
T_12 = FontProperties(fname= r'C:\\Windows\\Fonts\\times.ttf', size = 12)    # Times New Roman

# 导入自定义模块
from model_lstm import LSTMModel
from model_gru import GRUModel
from dataset import ThermoelectricDataset


# 设置中文字体（如果可用）
try:
    rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    rcParams['axes.unicode_minus'] = False
except (KeyError, ValueError) as e:
    # 如果字体设置失败，使用默认字体
    import warnings
    warnings.warn(f"Could not set font configuration: {e}. Using default fonts.")

# 确保输出编码为UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')

class Predictor:
    """预测器类，封装预测逻辑"""
    
    def __init__(self, model_path, device='auto'):
        """
        初始化预测器
        
        参数:
            model_path: str, 模型检查点路径
            device: str, 设备 ('auto', 'cuda', 'cpu')
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载检查点
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        # fj: 保持预测时的配置参数与训练时一致。
        self.config = checkpoint['config']
        
        # 创建模型
        if self.config['model_type'] == 'lstm':
            self.model = LSTMModel(
                input_size=8,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=8,
                predict_steps=self.config['predict_steps'],
                dropout=self.config['dropout']
            )
        elif self.config['model_type'] == 'gru':
            self.model = GRUModel(
                input_size=8,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=8,
                predict_steps=self.config['predict_steps'],
                dropout=self.config['dropout']
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.config['model_type']}")
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功 ({self.config['model_type'].upper()})")
        print(f"训练epoch: {checkpoint['epoch']}")
        print(f"验证损失: {checkpoint['val_loss']:.6f}")
        
        # 加载标准化器
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"标准化器已加载")
        else:
            self.scaler = None
            print("警告: 未找到标准化器，预测结果将是标准化后的值")
    
    def predict(self, input_sequence):
        """
        对单个输入序列进行预测
        
        参数:
            input_sequence: numpy array, 形状 [seq_len, 8]
        
        返回:
            predictions: numpy array, 形状 [predict_steps, 8]
        """
        # 标准化输入（如果有scaler）
        # fj: 这里用的mean和std参数都是训练集的整体参数
        if self.scaler is not None:
            input_sequence = self.scaler.transform(input_sequence)
        
        # 转换为tensor
        x = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)  # [1, seq_len, 8]
        
        # 预测
        with torch.no_grad():
            predictions, _ = self.model(x)
        # fj: 固定预测当初训练时的设置好的预测步数，并非1步
        
        # 转换回numpy
        predictions = predictions.squeeze(0).cpu().numpy()  # [predict_steps, 8]
        
        # 反标准化（如果有scaler）
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def predict_multi_step(self, input_sequence, steps):
        """
        多步迭代预测
        
        参数:
            input_sequence: numpy array, 形状 [seq_len, 8]
            steps: int, 要预测的总步数
        
        返回:
            predictions: numpy array, 形状 [steps, 8]
        """
        # 标准化输入
        if self.scaler is not None:
            input_sequence = self.scaler.transform(input_sequence)
        
        # 转换为tensor
        x = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)  # [1, seq_len, 8]
        
        # 使用模型的多步预测方法
        with torch.no_grad():
            predictions = self.model.predict_multi_step(x, steps)
        
        # 转换回numpy
        predictions = predictions.squeeze(0).cpu().numpy()  # [steps, 8]
        
        # 反标准化
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def predict_from_csv(self, csv_path, start_idx=0, save_path=None):
        """
        从CSV文件中读取数据并预测
        
        参数:
            csv_path: str, CSV文件路径
            start_idx: int, 从哪个位置开始取输入序列
            save_path: str or None, 保存预测结果的路径
        
        返回:
            predictions: numpy array
            ground_truth: numpy array (如果有的话)
        """
        # fj: 预测的步数与predict函数相同，固定为self.config['predict_steps']
        # 读取数据
        df = pd.read_csv(csv_path)
        voltage_columns = [
            "Yellow", "Ultraviolet", "Infrared", "Red",
            "Green", "Blue", "Transparent", "Violet"
        ]
        data = df[voltage_columns].values
        
        # 提取输入序列
        window_size = self.config['window_size']
        predict_steps = self.config['predict_steps']
        
        if start_idx + window_size > len(data):
            raise ValueError("start_idx太大，超出数据范围")
        
        input_seq = data[start_idx:start_idx + window_size]
        
        # 预测
        predictions = self.predict(input_seq)
        
        # 提取真实值（如果有）
        ground_truth = None
        if start_idx + window_size + predict_steps <= len(data):
            ground_truth = data[start_idx + window_size:start_idx + window_size + predict_steps]
        
        # 保存结果
        if save_path is not None:
            result = {
                'predictions': predictions,
                'ground_truth': ground_truth,
                'input_sequence': data[start_idx:start_idx + window_size]
            }
            np.save(save_path, result)
            print(f"预测结果已保存到: {save_path}")
        
        return predictions, ground_truth


def plot_predictions(input_seq, predictions, ground_truth=None, channel=0, save_path=None):
    """
    可视化预测结果
    
    参数:
        input_seq: numpy array, 输入序列 [seq_len, 8]
        predictions: numpy array, 预测结果 [predict_steps, 8]
        ground_truth: numpy array or None, 真实值 [predict_steps, 8]
        channel: int, 要可视化的通道 (0-7)
        save_path: str or None, 保存图像的路径
    """
    plt.figure(figsize=(12, 6))
    voltage_columns = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    # 时间轴
    input_time = np.arange(len(input_seq))
    pred_time = np.arange(len(input_seq), len(input_seq) + len(predictions))
    
    # 绘制输入序列
    plt.plot(input_time, input_seq[:, channel], 'b-', label='Input Sequence', linewidth=2)
    
    # 绘制预测
    plt.plot(pred_time, predictions[:, channel], 'r--', label='Prediction', linewidth=2)
    
    # 绘制真实值（如果有）
    if ground_truth is not None:
        plt.plot(pred_time, ground_truth[:, channel], 'g-', label='Ground Truth', linewidth=2)
    
    plt.xlabel('Time Step', fontproperties=T_14)
    plt.ylabel(f'Voltage (mV)', fontproperties=T_14)
    plt.title(f'Time Series Prediction - {voltage_columns[channel]}', fontproperties=T_14)
    plt.legend(prop = T_12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_channels(input_seq, predictions, ground_truth=None, save_path=None):
    """
    可视化所有8个通道的预测结果
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    input_time = np.arange(len(input_seq))
    pred_time = np.arange(len(input_seq), len(input_seq) + len(predictions))
    voltage_columns = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    for channel in range(8):
        ax = axes[channel]
        
        # 输入序列
        ax.plot(input_time, input_seq[:, channel], 'b-', label='Input', linewidth=1.5)
        
        # 预测
        ax.plot(pred_time, predictions[:, channel], 'r--', label='Prediction', linewidth=1.5)
        
        # 真实值
        if ground_truth is not None:
            ax.plot(pred_time, ground_truth[:, channel], 'g-', label='Ground Truth', linewidth=1.5)
        
        ax.set_xlabel('Time Step', fontproperties=T_14)
        ax.set_ylabel(f'Voltage (mV)', fontproperties=T_14)
        ax.set_title(f'{voltage_columns[channel]}', fontproperties=T_14)
        ax.legend(prop = T_12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    """主函数"""
    
    # 创建预测器
    predictor = Predictor(args.model_path, device=args.device)
    
    if args.csv_path:
        # 从CSV文件预测
        print(f"\n从CSV文件预测: {args.csv_path}")
        print(f"起始位置: {args.start_idx}")
        
        predictions, ground_truth = predictor.predict_from_csv(
            args.csv_path,
            start_idx=args.start_idx,
            save_path=args.save_path
        )
        
        print(f"\n预测结果形状: {predictions.shape}")
        print(f"预测结果:\n{predictions}")
        
        if ground_truth is not None:
            print(f"\n真实值:\n{ground_truth}")
            
            # 计算误差
            mse = np.mean((predictions - ground_truth) ** 2)
            mae = np.mean(np.abs(predictions - ground_truth))
            print(f"\n均方误差 (MSE): {mse:.6f}")
            print(f"平均绝对误差 (MAE): {mae:.6f}")
        
        # 可视化
        if args.plot:
            # 读取完整数据用于绘图
            df = pd.read_csv(args.csv_path)
            voltage_columns = [
            "Yellow", "Ultraviolet", "Infrared", "Red",
            "Green", "Blue", "Transparent", "Violet"
            ]
            data = df[voltage_columns].values
            window_size = predictor.config['window_size']
            input_seq = data[args.start_idx:args.start_idx + window_size]
            
            # 绘制所有通道
            plot_save_path = args.save_path.replace('.npy', '_all_channels.svg') if args.save_path else None
            plot_all_channels(input_seq, predictions, ground_truth, save_path=plot_save_path)
            
            # 绘制单个通道
            for channel_index in range(8):
                single_plot_path = args.save_path.replace('.npy', f'_{voltage_columns[channel_index]}.svg') if args.save_path else None
                plot_predictions(input_seq, predictions, ground_truth, channel=channel_index, save_path=single_plot_path)

    else:
        print("请提供CSV文件路径 (--csv_path)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    
    parser.add_argument('--model_path', type=str, required=True,  
                       help='模型检查点路径 (例如: ./checkpoints/best_model.pth)')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='输入CSV文件路径')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='从CSV文件的哪个位置开始取输入序列')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--save_path', type=str, default=None,
                       help='保存预测结果的路径 (.npy)')
    parser.add_argument('--plot', action='store_true',
                       help='是否绘制预测结果图')
    
    args = parser.parse_args()
    main(args)

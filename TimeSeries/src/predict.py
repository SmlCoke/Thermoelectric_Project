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
from datetime import datetime, timedelta
import matplotlib.dates as mdates
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


def build_time_axis(indices, start_time_str, interval_seconds):
    """Convert index array to datetime list when both time args are provided."""
    if start_time_str is None and interval_seconds is None:
        return indices, False
    if not start_time_str or interval_seconds is None:
        print("警告：仅提供了部分时间参数，已回退到索引轴。")
        return indices, False
    try:
        base_time = datetime.strptime(start_time_str, "%H:%M:%S")
    except ValueError:
        print("警告：--start_time 格式需为 时:分:秒，例如 12:36:26，已回退到索引轴。")
        return indices, False
    if interval_seconds <= 0:
        print("警告：--time_interval 必须大于 0，已回退到索引轴。")
        return indices, False
    step = timedelta(seconds=interval_seconds)
    times = [base_time + int(idx) * step for idx in indices]
    return times, True


def format_time_axis(ax, x_is_time):
    if x_is_time:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

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


def plot_predictions(input_seq, predictions, ground_truth=None, channel=0, save_path=None,
                     start_idx=0, start_time=None, time_interval=None):
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
    # 时间轴（默认用步索引，可选用实际时间）
    input_indices = np.arange(start_idx, start_idx + len(input_seq))
    pred_indices = np.arange(start_idx + len(input_seq), start_idx + len(input_seq) + len(predictions))
    input_time, x_is_time = build_time_axis(input_indices, start_time, time_interval)
    pred_time, _ = build_time_axis(pred_indices, start_time, time_interval)
    
    # 绘制输入序列
    plt.plot(input_time, input_seq[:, channel], 'b-', label='Input Sequence', linewidth=2)
    
    # 绘制预测
    plt.plot(pred_time, predictions[:, channel], 'r--', label='Prediction', linewidth=2)
    
    # 绘制真实值（如果有）
    if ground_truth is not None:
        plt.plot(pred_time, ground_truth[:, channel], 'g-', label='Ground Truth', linewidth=2)
    
    plt.xlabel('Time', fontproperties=T_14)
    plt.ylabel(f'Voltage (mV)', fontproperties=T_14)
    plt.title(f'Time Series Prediction - {voltage_columns[channel]}', fontproperties=T_14)
    plt.legend(prop = T_12)
    plt.grid(True, alpha=0.3)
    format_time_axis(plt.gca(), x_is_time)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_channels(input_seq, predictions, ground_truth=None, save_path=None,
                      start_idx=0, start_time=None, time_interval=None):
    """
    可视化所有8个通道的预测结果
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    input_indices = np.arange(start_idx, start_idx + len(input_seq))
    pred_indices = np.arange(start_idx + len(input_seq), start_idx + len(input_seq) + len(predictions))
    input_time, x_is_time = build_time_axis(input_indices, start_time, time_interval)
    pred_time, _ = build_time_axis(pred_indices, start_time, time_interval)
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
        
        ax.set_xlabel('Time', fontproperties=T_14)
        ax.set_ylabel(f'Voltage (mV)', fontproperties=T_14)
        ax.set_title(f'{voltage_columns[channel]}', fontproperties=T_14)
        ax.legend(prop = T_12)
        ax.grid(True, alpha=0.3)
        format_time_axis(ax, x_is_time)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def predict_custom_points(predictor: Predictor, csv_path: str,
                          input_indices: list[int],
                          target_start_indices: list[int],
                          channel_name: str = "Violet",
                          save_npy: str | None = None,
                          save_fig: str | None = None,
                          start_time: str | None = None,
                          time_interval: float | None = None):
    """
    使用原始（未降采样）模型，按给定索引做定制预测，并拼接可视化。

    input_indices: 用作输入序列的原始数据索引列表（len=window_size）
    target_start_indices: 每个单步预测窗口的起始索引（len = 需要的预测点数）
                          对于每个起点，取该窗口的首步预测作为目标点。
    """
    df = pd.read_csv(csv_path)
    voltage_columns = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    if channel_name not in voltage_columns:
        raise ValueError(f"通道 {channel_name} 不存在，可选：{voltage_columns}")
    ch_idx = voltage_columns.index(channel_name)

    data = df[voltage_columns].values
    ws = predictor.config['window_size']  # 期望 60
    ps = predictor.config['predict_steps']  # 期望 10

    # 取 60 个输入点
    if len(input_indices) != ws:
        raise ValueError(f"input_indices 数量应为 {ws}，当前 {len(input_indices)}")
    input_seq = data[input_indices]
    print("input_seq", input_seq)
    # 逐窗单步预测（取 predict 返回的第一步）
    preds_list = []
    target_indices = []
    for s in target_start_indices:
        window = data[s:s+ws]
        if len(window) != ws:
            raise ValueError(f"起点 {s} 的窗口长度不足 {ws}")
        pred_10 = predictor.predict(window)  # [predict_steps, 8]
        preds_list.append(pred_10[0])        # 仅首步
        target_indices.append(s + ws)        # 对应时间点

    preds = np.stack(preds_list, axis=0)  # [num_targets, 8]
    gt_targets = data[target_indices, ch_idx]  # Ground Truth for目标点

    # 新增：计算并打印误差
    mse = float(np.mean((preds[:, ch_idx] - gt_targets) ** 2))
    mae = float(np.mean(np.abs(preds[:, ch_idx] - gt_targets)))
    print(f"[Custom] {channel_name} MSE: {mse:.6f}, MAE: {mae:.6f}")

    # 保存中间数据
    if save_npy:
        np.save(save_npy, {
            "input_indices": np.array(input_indices),
            "input_seq": input_seq,
            "target_start_indices": np.array(target_start_indices),
            "target_indices": np.array(target_indices),
            "predictions": preds,
            "gt_targets": gt_targets,
            "mse": mse,
            "mae": mae,
        })
        print(f"中间数据已保存: {save_npy}")

    # 拼接并绘图（单通道）
    # 构建时间轴（若提供时间参数，则转换为实际时间）
    x_input, x_is_time = build_time_axis(np.array(input_indices), start_time, time_interval)
    x_target, _ = build_time_axis(np.array(target_indices), start_time, time_interval)

    plt.figure(figsize=(12, 5))
    plt.plot(x_input, input_seq[:, ch_idx], 'b-', label='Input (GT)', linewidth=2)
    plt.plot(x_target, preds[:, ch_idx], 'r--', label='Predicted', linewidth=2)
    plt.plot(x_target, gt_targets, 'g-', label='Ground Truth', linewidth=1.5)
    if len(x_input) > 0:
        plt.axvline(x_input[-1], color='gray', linestyle=':', alpha=0.8, label='Boundary')
    plt.xlabel('Time', fontproperties=T_14)
    plt.ylabel('Voltage (mV)', fontproperties=T_14)
    plt.title(f'Custom Prediction - {channel_name}', fontproperties=T_16)
    plt.legend(prop=T_12)
    plt.grid(True, alpha=0.3)
    format_time_axis(plt.gca(), x_is_time)
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight', dpi=150)
        print(f"图像已保存: {save_fig}")
    else:
        plt.show()
    plt.close()

    return {
        "input_indices": input_indices,
        "target_indices": target_indices,
        "predictions": preds,
        "mse": mse,
        "mae": mae,
    }


def main(args):
    """主函数"""
    predictor = Predictor(args.model_path, device=args.device)

    # ---- 新增：定制索引预测模式 ----
    if args.custom:
        input_indices = list(range(458-1, 813-1, 6))           # 458..812, 共60
        target_start_indices = list(range(758-1, 813-1, 6))    # 758..812, 共10 -> 预测 818..872
        ret = predict_custom_points(
            predictor,
            csv_path=args.csv_path,
            input_indices=input_indices,
            target_start_indices=target_start_indices,
            channel_name=args.channel,
            save_npy=args.save_path,
            save_fig=args.plot_path,
            start_time=args.start_time,
            time_interval=args.time_interval
        )
        # 追加打印，便于日志查看
        print(f"[Custom] {args.channel} MSE: {ret['mse']:.6f}, MAE: {ret['mae']:.6f}")
        return
    # ---- 原有逻辑 ----
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
            print("input_seq: ", input_seq)
            # 绘制所有通道
            plot_save_path = args.save_path.replace('.npy', '_all_channels.svg') if args.save_path else None
            plot_all_channels(
                input_seq,
                predictions,
                ground_truth,
                save_path=plot_save_path,
                start_idx=args.start_idx,
                start_time=args.start_time,
                time_interval=args.time_interval,
            )
            
            # 绘制单个通道
            for channel_index in range(8):
                single_plot_path = args.save_path.replace('.npy', f'_{voltage_columns[channel_index]}.svg') if args.save_path else None
                plot_predictions(
                    input_seq,
                    predictions,
                    ground_truth,
                    channel=channel_index,
                    save_path=single_plot_path,
                    start_idx=args.start_idx,
                    start_time=args.start_time,
                    time_interval=args.time_interval,
                )

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
                       help='是否绘制预测结果图（时间轴可由 --start_time / --time_interval 控制）')
    parser.add_argument('--start_time', type=str, default=None,
                       help='起始时间点，格式 时:分:秒，例如 12:36:26')
    parser.add_argument('--time_interval', type=float, default=None,
                       help='相邻两个数据点的时间间隔（秒）')
    # 新增参数
    parser.add_argument('--custom', action='store_true',
                        help='使用预定义索引（458..812 输入，预测 818..872）并绘图')
    parser.add_argument('--channel', type=str, default='Violet',
                        help='定制模式下要绘制的通道名称')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='定制模式下保存绘图的路径 (.svg/.png 等)')

    args = parser.parse_args()
    main(args)

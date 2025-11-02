"""
测试脚本 - 完整版本
加载训练好的模型，在测试集上进行预测并可视化
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import Config
from model import AdvancedCNN_LSTM_Forecaster
from dataset import load_and_preprocess_data

def calculate_metrics(predictions, targets):
    """
    计算评估指标
    """
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    # 避免除零错误，只计算目标值不为0的位置
    mask = targets != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def test_model(config, num_samples=5):
    """
    测试模型并可视化结果
    
    Args:
        config: 配置对象
        num_samples: 展示多少个测试样本
    """
    print("=" * 60)
    print("加载模型和数据...")
    print("=" * 60)
    
    # 检查模型是否存在
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"错误：模型文件不存在: {config.MODEL_SAVE_PATH}")
        print("请先运行 'python train.py' 训练模型")
        return
    
    # 加载数据
    _, _, test_loader, scaler = load_and_preprocess_data(config)
    
    # 加载模型
    model = AdvancedCNN_LSTM_Forecaster(config)
    
    # 处理可能的多GPU训练模型
    if config.MULTI_GPU:
        model = nn.DataParallel(model)
    
    model = model.to(config.DEVICE)
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    
    # 处理state_dict的键名（可能有module.前缀）
    state_dict = checkpoint['model_state_dict']
    if config.MULTI_GPU and not isinstance(model, nn.DataParallel):
        # 移除module.前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    actual_model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ 模型加载成功 (训练epoch: {checkpoint['epoch']})")
    print(f"✓ 验证损失: {checkpoint['val_loss']:.6f}")
    
    # 收集所有预测和真实值
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    print("\n正在进行预测...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            
            # 移回CPU并转为numpy
            all_inputs.append(inputs.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    # 合并所有批次
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"✓ 预测完成")
    print(f"  测试样本数: {len(all_predictions)}")
    
    # 计算整体指标
    print("\n" + "=" * 60)
    print("整体性能指标")
    print("=" * 60)
    
    # 将数据展平以计算总体指标
    pred_flat = all_predictions.reshape(-1)
    target_flat = all_targets.reshape(-1)
    
    metrics = calculate_metrics(pred_flat, target_flat)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # 为每个通道单独计算指标
    channel_names = ['VSW黑', 'VSW红', 'VSW蓝', 'VSW绿', 
                     'VLW黑', 'VLW红', 'VLW蓝', 'VLW绿']
    
    print("\n各通道MAE:")
    for i, channel_name in enumerate(channel_names):
        channel_pred = all_predictions[:, :, i].reshape(-1)
        channel_target = all_targets[:, :, i].reshape(-1)
        mae = mean_absolute_error(channel_target, channel_pred)
        print(f"  {channel_name}: {mae:.6f}")
    
    # 可视化
    print("\n" + "=" * 60)
    print("生成可视化图...")
    print("=" * 60)
    
    visualize_predictions(all_inputs, all_predictions, all_targets, 
                         channel_names, config, num_samples)

def visualize_predictions(inputs, predictions, targets, channel_names, config, num_samples=5):
    """
    可视化预测结果
    """
    os.makedirs('results', exist_ok=True)
    
    # 随机选择几个样本进行可视化
    num_samples = min(num_samples, len(predictions))
    sample_indices = np.random.choice(len(predictions), num_samples, replace=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(f'测试样本 {sample_idx + 1} - 预测结果', fontsize=16)
        
        input_seq = inputs[sample_idx]
        pred_seq = predictions[sample_idx]
        target_seq = targets[sample_idx]
        
        # 时间轴
        input_hours = np.arange(config.INPUT_LENGTH) * config.SAMPLE_INTERVAL / 3600
        output_hours = (config.INPUT_LENGTH + np.arange(config.OUTPUT_LENGTH)) * config.SAMPLE_INTERVAL / 3600
        
        # 为每个通道绘图
        for ch_idx in range(8):
            row = ch_idx % 4
            col = ch_idx // 4
            ax = axes[row, col]
            
            # 输入序列（历史数据）
            ax.plot(input_hours, input_seq[:, ch_idx], 'b-', 
                   label='历史数据', linewidth=1, alpha=0.7)
            
            # 真实值（绿色）
            ax.plot(output_hours, target_seq[:, ch_idx], 'g-', 
                   label='真实值', linewidth=2)
            
            # 预测值（红色虚线）
            ax.plot(output_hours, pred_seq[:, ch_idx], 'r--', 
                   label='预测值', linewidth=2)
            
            # 添加垂直线分隔输入和输出
            ax.axvline(x=config.INPUT_LENGTH * config.SAMPLE_INTERVAL / 3600, 
                      color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('时间 (小时)')
            ax.set_ylabel('归一化电压')
            ax.set_title(channel_names[ch_idx])
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f'results/prediction_sample_{idx + 1}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存图像: {save_path}")
    
    # 绘制误差分布图
    print("\n生成误差分析图...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('各通道预测误差分布', fontsize=16)
    
    for ch_idx in range(8):
        row = ch_idx // 4
        col = ch_idx % 4
        ax = axes[row, col]
        
        # 计算误差
        errors = predictions[:, :, ch_idx].reshape(-1) - targets[:, :, ch_idx].reshape(-1)
        
        # 绘制误差直方图
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('预测误差')
        ax.set_ylabel('频数')
        ax.set_title(f'{channel_names[ch_idx]}\nMAE={np.abs(errors).mean():.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/error_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存误差分析图: {save_path}")
    
    # 绘制云遮挡事件识别能力
    print("\n生成云遮挡事件识别示例...")
    
    # 选择一个样本
    sample_idx = 0
    pred_seq = predictions[sample_idx]
    target_seq = targets[sample_idx]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    output_hours = np.arange(config.OUTPUT_LENGTH) * config.SAMPLE_INTERVAL / 3600
    
    # 绘制VSW黑和VLW黑（最能体现云遮挡）
    ax.plot(output_hours, target_seq[:, 0], 'b-', label='VSW黑 (真实)', linewidth=2)
    ax.plot(output_hours, pred_seq[:, 0], 'b--', label='VSW黑 (预测)', linewidth=2)
    ax.plot(output_hours, target_seq[:, 4], 'r-', label='VLW黑 (真实)', linewidth=2)
    ax.plot(output_hours, pred_seq[:, 4], 'r--', label='VLW黑 (预测)', linewidth=2)
    
    ax.set_xlabel('预测时间 (小时)', fontsize=12)
    ax.set_ylabel('归一化电压', fontsize=12)
    ax.set_title('云遮挡事件预测能力示例\n(VSW↓ + VLW↑ = 云来了)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/cloud_event_prediction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 保存云遮挡示例图: {save_path}")
    
    print("\n" + "=" * 60)
    print("测试完成！所有结果已保存到 'results/' 目录")
    print("=" * 60)

if __name__ == "__main__":
    config = Config()
    test_model(config, num_samples=3)

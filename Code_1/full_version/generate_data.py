"""
生成模拟的时间序列数据 - 完整版本
生成更多天数的数据，用于服务器训练
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config

def generate_simulated_data(config, num_days=30, save_path=None):
    """
    生成模拟的多通道时间序列数据（更多天数）
    
    模拟策略：
    1. 白天太阳辐射增强 -> VSW增大
    2. 夜间地面辐射为主 -> VLW相对增大
    3. 云遮挡时：VSW骤降，VLW骤升
    4. 不同颜色的芯片对不同波长有不同响应
    5. 添加长期趋势和季节性变化
    
    Args:
        config: 配置对象
        num_days: 生成多少天的数据（默认30天，用于服务器训练）
        save_path: 保存路径
    
    Returns:
        data: numpy数组，shape = (total_timesteps, 8)
    """
    
    if save_path is None:
        save_path = config.DATA_PATH
    
    print("=" * 60)
    print("开始生成扩展模拟数据（完整版）...")
    print("=" * 60)
    
    # 时间参数
    seconds_per_day = 24 * 3600
    total_seconds = num_days * seconds_per_day
    total_timesteps = total_seconds // config.SAMPLE_INTERVAL
    
    print(f"生成参数:")
    print(f"  天数: {num_days}")
    print(f"  采样间隔: {config.SAMPLE_INTERVAL} 秒")
    print(f"  总时间步: {total_timesteps}")
    print(f"  预计文件大小: ~{total_timesteps * 8 * 8 / 1024 / 1024:.2f} MB")
    
    # 初始化数据数组
    data = np.zeros((total_timesteps, config.NUM_CHANNELS))
    
    # 生成时间轴（小时为单位，方便计算日周期）
    time_hours = np.arange(total_timesteps) * config.SAMPLE_INTERVAL / 3600
    
    print("\n正在生成各通道数据...")
    
    # === 1. 生成日周期基础信号 ===
    # 太阳辐射（白天强，夜间弱）
    solar_radiation = np.maximum(0, np.sin(2 * np.pi * time_hours / 24 - np.pi/2))
    
    # 长波辐射（夜间相对增强）
    longwave_radiation = 0.3 + 0.2 * np.sin(2 * np.pi * time_hours / 24 + np.pi/2)
    
    # === 2. 添加月级别的季节性变化 ===
    seasonal_trend = 0.1 * np.sin(2 * np.pi * time_hours / (24 * 30))  # 月周期
    solar_radiation = solar_radiation * (1 + seasonal_trend)
    
    # === 3. 添加长期趋势（模拟季节变化）===
    long_term_trend = 0.05 * (time_hours / (24 * num_days))  # 线性增长
    solar_radiation = solar_radiation * (1 + long_term_trend)
    
    # === 4. 添加随机云遮挡事件（更多事件）===
    num_cloud_events = num_days * 5  # 平均每天5次云遮挡
    cloud_mask = np.zeros(total_timesteps)
    
    print(f"  生成 {num_cloud_events} 个云遮挡事件...")
    
    for _ in range(num_cloud_events):
        # 随机选择云遮挡的开始时间和持续时间
        start_idx = np.random.randint(0, total_timesteps - 720)
        duration = np.random.randint(60, 720)  # 10分钟-2小时
        
        # 云遮挡期间：逐渐遮挡，再逐渐消散
        cloud_profile = np.sin(np.linspace(0, np.pi, duration))
        
        # 云的强度也随机
        intensity = np.random.uniform(0.3, 1.0)
        cloud_mask[start_idx:start_idx + duration] = np.maximum(
            cloud_mask[start_idx:start_idx + duration],
            cloud_profile * intensity
        )
    
    # === 5. 生成各通道数据 ===
    
    # 短波通道（VSW）：受太阳辐射影响，受云遮挡影响大
    vsw_black = 1.0 * solar_radiation * (1 - 0.8 * cloud_mask)
    vsw_red = 0.8 * solar_radiation * (1 - 0.7 * cloud_mask)
    vsw_blue = 0.6 * solar_radiation * (1 - 0.9 * cloud_mask)
    vsw_green = 0.7 * solar_radiation * (1 - 0.8 * cloud_mask)
    
    # 长波通道（VLW）：受地面辐射影响，云来时增强
    vlw_black = longwave_radiation * (1 + 0.3 * cloud_mask)
    vlw_red = 0.9 * longwave_radiation * (1 + 0.4 * cloud_mask)
    vlw_blue = 0.7 * longwave_radiation * (1 + 0.2 * cloud_mask)
    vlw_green = 0.8 * longwave_radiation * (1 + 0.3 * cloud_mask)
    
    # === 6. 添加多种噪声 ===
    white_noise_level = 0.02
    colored_noise_level = 0.01
    
    for i, channel_data in enumerate([vsw_black, vsw_red, vsw_blue, vsw_green,
                                       vlw_black, vlw_red, vlw_blue, vlw_green]):
        # 白噪声
        white_noise = np.random.normal(0, white_noise_level, total_timesteps)
        
        # 有色噪声（低频漂移）
        colored_noise = np.zeros(total_timesteps)
        for freq_factor in [1, 2, 3, 5]:
            colored_noise += colored_noise_level * np.sin(
                2 * np.pi * time_hours / (24 * freq_factor) + 
                np.random.uniform(0, 2 * np.pi)
            )
        
        # 周级别漂移
        weekly_drift = 0.01 * np.sin(2 * np.pi * time_hours / (24 * 7))
        
        # 随机脉冲噪声（模拟偶发干扰）
        num_spikes = num_days * 2
        for _ in range(num_spikes):
            spike_idx = np.random.randint(0, total_timesteps)
            spike_duration = np.random.randint(1, 10)
            spike_amplitude = np.random.uniform(-0.05, 0.05)
            if spike_idx + spike_duration < total_timesteps:
                channel_data[spike_idx:spike_idx + spike_duration] += spike_amplitude
        
        data[:, i] = channel_data + white_noise + colored_noise + weekly_drift
    
    # === 7. 确保数据非负且在合理范围内 ===
    data = np.clip(data, 0, None)
    
    # === 8. 保存数据 ===
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    np.savez_compressed(save_path, 
                       data=data,
                       time_hours=time_hours,
                       config={
                           'num_channels': config.NUM_CHANNELS,
                           'sample_interval': config.SAMPLE_INTERVAL,
                           'num_days': num_days
                       })
    
    print(f"\n✓ 数据已保存到: {save_path}")
    print(f"  数据形状: {data.shape}")
    print(f"  文件大小: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    
    # === 9. 生成可视化 ===
    print("\n正在生成可视化图...")
    visualize_data(data, time_hours, save_dir=os.path.dirname(save_path) or '.')
    
    return data


def visualize_data(data, time_hours, save_dir='.'):
    """
    可视化生成的数据
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # 只显示前3天的数据，避免图太挤
    display_hours = min(72, len(time_hours))
    time_display = time_hours[:display_hours]
    
    # === 短波通道 ===
    ax = axes[0]
    ax.plot(time_display, data[:display_hours, 0], label='VSW黑', linewidth=1)
    ax.plot(time_display, data[:display_hours, 1], label='VSW红', linewidth=1)
    ax.plot(time_display, data[:display_hours, 2], label='VSW蓝', linewidth=1)
    ax.plot(time_display, data[:display_hours, 3], label='VSW绿', linewidth=1)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('电压 (归一化)')
    ax.set_title('短波通道 (VSW) - 前3天数据')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # === 长波通道 ===
    ax = axes[1]
    ax.plot(time_display, data[:display_hours, 4], label='VLW黑', linewidth=1)
    ax.plot(time_display, data[:display_hours, 5], label='VLW红', linewidth=1)
    ax.plot(time_display, data[:display_hours, 6], label='VLW蓝', linewidth=1)
    ax.plot(time_display, data[:display_hours, 7], label='VLW绿', linewidth=1)
    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('电压 (归一化)')
    ax.set_title('长波通道 (VLW) - 前3天数据')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'data_visualization.png')
    plt.savefig(save_path, dpi=150)
    print(f"✓ 可视化图已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    config = Config()
    
    # 生成30天的模拟数据（服务器版本）
    data = generate_simulated_data(config, num_days=30)
    
    print("\n" + "=" * 60)
    print("扩展数据生成完成！")
    print("=" * 60)
    print("\n接下来可以运行:")
    print("  python train.py  # 开始训练模型（服务器版本）")

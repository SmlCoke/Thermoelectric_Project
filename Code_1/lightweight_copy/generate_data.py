"""
生成模拟的时间序列数据
模拟热电芯片的8通道电压数据（VSW黑/红/蓝/绿, VLW黑/红/蓝/绿）
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config
import sys; 

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def generate_simulated_data(config, num_days=7, save_path=None):
    """
    生成模拟的多通道时间序列数据
    
    模拟策略：
    1. 白天太阳辐射增强 -> VSW增大
    2. 夜间地面辐射为主 -> VLW相对增大
    3. 云遮挡时：VSW骤降，VLW骤升
    4. 不同颜色的芯片对不同波长有不同响应
    
    Args:
        config: 配置对象
        num_days: 生成多少天的数据
        save_path: 保存路径
    
    Returns:
        data: numpy数组，shape = (total_timesteps, 8)
    """
    
    if save_path is None:
        save_path = config.DATA_PATH
    
    print("=" * 60)
    print("开始生成模拟数据...")
    print("=" * 60)
    
    # 时间参数
    seconds_per_day = 24 * 3600
    total_seconds = num_days * seconds_per_day
    print("total_seconds:", total_seconds)
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
    
    # 打印完整numpy数组时不省略
    np.set_printoptions(threshold=np.inf, suppress=True)
    print("\n时间轴示例 (小时):")
    print(time_hours[:24 * 3 * 360])  # 打印前3天的时间

    # === 通道定义 ===
    # 0-3: VSW黑, VSW红, VSW蓝, VSW绿
    # 4-7: VLW黑, VLW红, VLW蓝, VLW绿
    
    print("\n正在生成各通道数据...")
    
    # === 1. 生成日周期基础信号 ===
    # 太阳辐射（白天强，夜间弱）
    solar_radiation = np.maximum(0, np.sin(2 * np.pi * time_hours / 24 - np.pi/2))
    
    # 长波辐射（夜间相对增强）
    longwave_radiation = 0.3 + 0.2 * np.sin(2 * np.pi * time_hours / 24 + np.pi/2)
    
    # === 2. 添加随机云遮挡事件 ===
    num_cloud_events = num_days * 3  # 平均每天3次云遮挡
    cloud_mask = np.zeros(total_timesteps)
    
    for _ in range(num_cloud_events):
        # 随机选择云遮挡的开始时间和持续时间
        start_idx = np.random.randint(0, total_timesteps - 360)
        duration = np.random.randint(60, 360)  # 10-60分钟
        
        # 云遮挡期间：逐渐遮挡，再逐渐消散
        cloud_profile = np.sin(np.linspace(0, np.pi, duration))
        cloud_mask[start_idx:start_idx + duration] = np.maximum(
            cloud_mask[start_idx:start_idx + duration],
            cloud_profile
        )
    
    # === 3. 生成各通道数据 ===
    
    # 短波通道（VSW）：受太阳辐射影响，受云遮挡影响大
    # 不同颜色对不同波长的响应不同
    vsw_black = 1.0 * solar_radiation * (1 - 0.8 * cloud_mask)  # 黑色吸收全波段
    vsw_red = 0.8 * solar_radiation * (1 - 0.7 * cloud_mask)    # 红色对长波敏感
    vsw_blue = 0.6 * solar_radiation * (1 - 0.9 * cloud_mask)   # 蓝色对短波敏感
    vsw_green = 0.7 * solar_radiation * (1 - 0.8 * cloud_mask)  # 绿色居中
    
    # 长波通道（VLW）：受地面辐射影响，云来时增强
    vlw_black = longwave_radiation * (1 + 0.3 * cloud_mask)
    vlw_red = 0.9 * longwave_radiation * (1 + 0.4 * cloud_mask)
    vlw_blue = 0.7 * longwave_radiation * (1 + 0.2 * cloud_mask)
    vlw_green = 0.8 * longwave_radiation * (1 + 0.3 * cloud_mask)
    
    # === 4. 添加噪声 ===
    noise_level = 0.02
    for i, channel_data in enumerate([vsw_black, vsw_red, vsw_blue, vsw_green,
                                       vlw_black, vlw_red, vlw_blue, vlw_green]):
        # 白噪声
        white_noise = np.random.normal(0, noise_level, total_timesteps)
        
        # 低频漂移
        drift = 0.01 * np.sin(2 * np.pi * time_hours / (24 * 7))  # 周级别漂移
        
        data[:, i] = channel_data + white_noise + drift
    
    # === 5. 确保数据非负且在合理范围内 ===
    data = np.clip(data, 0, None)
    
    # === 6. 保存数据 ===
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    np.savez_compressed(save_path, 
                       data=data,
                       time_hours=time_hours,
                       config={
                           'num_channels': config.NUM_CHANNELS,
                           'sample_interval': config.SAMPLE_INTERVAL,
                           'num_days': num_days
                       })
    
    print(f"\n 数据已保存到: {save_path}")
    print(f"  数据形状: {data.shape}")
    
    # === 7. 生成可视化 ===
    print("\n正在生成可视化图...")
    visualize_data(data, time_hours, save_dir=os.path.dirname(save_path) or '.')
    
    return data


def visualize_data(data, time_hours, save_dir='.'):
    """
    可视化生成的数据
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # 只显示前3天的数据，避免图太挤
    display_hours = min(72*360, len(time_hours))
    time_display = time_hours[:display_hours]
    
    
    # === 短波通道 ===
    ax = axes[0]
    ax.plot(time_display, data[:display_hours, 0], label='VSW_black', linewidth=1)
    ax.plot(time_display, data[:display_hours, 1], label='VSW_red', linewidth=1)
    ax.plot(time_display, data[:display_hours, 2], label='VSW_blue', linewidth=1)
    ax.plot(time_display, data[:display_hours, 3], label='VSW_green', linewidth=1)
    ax.set_xlabel('Time(Hour)')
    ax.set_ylabel('Votage (Normalized)')
    ax.set_title('VSW Channels - First 3 Days')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # === 长波通道 ===
    ax = axes[1]
    ax.plot(time_display, data[:display_hours, 4], label='VLW_black', linewidth=1)
    ax.plot(time_display, data[:display_hours, 5], label='VLW_red', linewidth=1)
    ax.plot(time_display, data[:display_hours, 6], label='VLW_blue', linewidth=1)
    ax.plot(time_display, data[:display_hours, 7], label='VLW_green', linewidth=1)
    ax.set_xlabel('Time(Hour)')
    ax.set_ylabel('Voltage (Normalized)')
    ax.set_title('VLW Channels - First 3 Days')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'data_visualization.svg')
    plt.savefig(save_path, dpi=150)
    print(f" 可视化图已保存: {save_path}")
    plt.close()
    
    # === 显示云遮挡事件的影响 ===
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 选择一个有云遮挡的时间段
    start_hour = 24  # 从第2天开始
    end_hour = start_hour + 6  # 显示6小时
    start_idx = int(start_hour * 3600 / 10)  # 转换为索引
    end_idx = int(end_hour * 3600 / 10)
    
    time_slice = time_hours[start_idx:end_idx]
    ax.plot(time_slice, data[start_idx:end_idx, 0], label='VSW_black', linewidth=2)
    ax.plot(time_slice, data[start_idx:end_idx, 4], label='VLW_black', linewidth=2)
    ax.set_xlabel('Time(Hour)')
    ax.set_ylabel('Voltage (Normalized)')
    ax.set_title('Cloud Cover Event Example: VSW Decrease, VLW Increase')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cloud_event_example.svg')
    plt.savefig(save_path, dpi=150)
    print(f" 云遮挡示例图已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    config = Config()
    
    # 生成7天的模拟数据
    data = generate_simulated_data(config, num_days=7)
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print("\n接下来可以运行:")
    print("  python train.py  # 开始训练模型")

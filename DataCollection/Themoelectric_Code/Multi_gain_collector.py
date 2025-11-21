#!/usr/bin/env python3
# Multi_gain_collector.py
# 多增益4个ADS1115数据采集脚本（兼容 adafruit-circuitpython-ads1x15 v3.x）
# 采集8个TEC1-04906热电芯片的电压数据，差分模式 (A0-A1, A2-A3)
# 特性：对每个通道采集多个增益下的电压值，并自动选择最优值

import time
import csv
from datetime import datetime
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# 配置参数
SAMPLE_INTERVAL = 10  # 采集间隔（秒）
# 4个ADS1115的I2C地址（按你的硬件接法）
ADS1115_ADDRESSES = {
    'ADS1': 0x48,  # ADDR -> GND
    'ADS2': 0x49,  # ADDR -> VDD
    'ADS3': 0x4A,  # ADDR -> SDA
    'ADS4': 0x4B,  # ADDR -> SCL
}

# 增益配置及对应的电压范围
# ADS1115增益设置：gain值 -> (±电压范围V, 描述)
GAIN_SETTINGS = {
    16: (0.256, "±0.256V"),
    8:  (0.512, "±0.512V"),
    4:  (1.024, "±1.024V"),
    2:  (2.048, "±2.048V"),
    1:  (4.096, "±4.096V"),
}

# 需要测试的增益列表（从高精度到低精度）
GAINS_TO_TEST = [16, 8, 4, 2, 1]

# 每个TEC在CSV中的列数（各增益值 + 最优值 + 最优增益）
COLUMNS_PER_TEC = len(GAINS_TO_TEST) + 2


def setup_all_ads1115():
    """初始化所有4个ADS1115"""
    ads_devices = {}
    try:
        # 创建I2C总线
        i2c = busio.I2C(board.SCL, board.SDA)
        # 小等待以稳定 I2C 总线
        time.sleep(0.2)

        # 初始化每个ADS1115
        for name, address in ADS1115_ADDRESSES.items():
            try:
                ads = ADS.ADS1115(i2c, address=address)
                # 初始增益设置为16（后续会动态改变）
                ads.gain = 16
                ads_devices[name] = ads
                print(f"{name} 初始化成功，地址: 0x{address:02X}")
            except Exception as e:
                print(f"{name} (0x{address:02X}) 初始化失败: {e}")

        if len(ads_devices) == 0:
            print("错误：没有成功初始化任何ADS1115设备")
            return None
        elif len(ads_devices) < len(ADS1115_ADDRESSES):
            print(f"警告：只成功初始化了 {len(ads_devices)}/{len(ADS1115_ADDRESSES)} 个ADS1115设备")

        return ads_devices

    except Exception as e:
        print(f"I2C总线初始化失败: {e}")
        return None


def read_differential_voltage_with_gain(ads, channel_pair, gain):
    """
    使用指定增益读取差分电压（兼容新版 API：使用整数通道号）
    
    设计说明：
    - 每次读取前设置增益，确保使用正确的量程
    - 10ms延迟确保ADS1115内部放大器稳定
    - 单线程顺序访问，无并发问题
    
    channel_pair: '0-1' 或 '2-3'
    gain: 增益值 (16, 8, 4, 2, 1)
    返回电压（V）或 None
    """
    try:
        # 设置增益
        ads.gain = gain
        # 短暂延迟以确保增益设置生效
        time.sleep(0.01)
        
        if channel_pair == '0-1':
            chan = AnalogIn(ads, 0, 1)
        elif channel_pair == '2-3':
            chan = AnalogIn(ads, 2, 3)
        else:
            raise ValueError("无效的通道对，应为 '0-1' 或 '2-3'")

        # 注意：AnalogIn.voltage 返回以 V 为单位的浮点数
        return chan.voltage
    except Exception as e:
        print(f"读取通道{channel_pair}(gain={gain})失败: {e}")
        return None


def read_multi_gain_voltage(ads, channel_pair):
    """
    对指定通道采集多个增益下的电压值，并选择最优值
    
    设计说明：
    - 采集所有增益下的数据以提供完整记录（满足需求：记录所有增益值到CSV）
    - 不采用早停优化，因为需要完整的多增益数据用于后续分析
    - 单线程顺序访问ADC，无并发风险
    
    返回字典：{
        'gain_16': voltage, 'gain_8': voltage, ... ,
        'optimal_voltage': best_voltage, 'optimal_gain': best_gain
    }
    """
    result = {}
    
    # 采集所有增益下的电压（注意：需要全部采集以记录到CSV）
    for gain in GAINS_TO_TEST:
        voltage = read_differential_voltage_with_gain(ads, channel_pair, gain)
        result[f'gain_{gain}'] = voltage
    
    # 选择最优电压值：
    # 规则：选择电压值在有效范围内（不超过满量程的95%）且增益最高（精度最高）的测量值
    optimal_voltage = None
    optimal_gain = None
    
    for gain in GAINS_TO_TEST:  # 已按精度从高到低排序
        voltage = result[f'gain_{gain}']
        if voltage is None:
            continue
            
        max_range = GAIN_SETTINGS[gain][0]
        # 检查电压是否在有效范围内（不超过满量程的95%以确保精度）
        if abs(voltage) <= max_range * 0.95:
            optimal_voltage = voltage
            optimal_gain = gain
            break  # 找到第一个（精度最高的）有效值即可
    
    result['optimal_voltage'] = optimal_voltage
    result['optimal_gain'] = optimal_gain
    
    return result


def read_all_voltages_multi_gain(ads_devices):
    """
    读取所有8个TEC芯片在多个增益下的电压，按 ADS1..ADS4 顺序返回列表（长度8）
    每个元素是一个字典，包含各增益下的电压值和最优值
    """
    voltages = []

    # ADS1 - TEC1 和 TEC2
    if 'ADS1' in ads_devices:
        voltages.append(read_multi_gain_voltage(ads_devices['ADS1'], '0-1'))  # TEC1
        voltages.append(read_multi_gain_voltage(ads_devices['ADS1'], '2-3'))  # TEC2
    else:
        voltages.extend([None, None])

    # ADS2 - TEC3 和 TEC4
    if 'ADS2' in ads_devices:
        voltages.append(read_multi_gain_voltage(ads_devices['ADS2'], '0-1'))  # TEC3
        voltages.append(read_multi_gain_voltage(ads_devices['ADS2'], '2-3'))  # TEC4
    else:
        voltages.extend([None, None])

    # ADS3 - TEC5 和 TEC6
    if 'ADS3' in ads_devices:
        voltages.append(read_multi_gain_voltage(ads_devices['ADS3'], '0-1'))  # TEC5
        voltages.append(read_multi_gain_voltage(ads_devices['ADS3'], '2-3'))  # TEC6
    else:
        voltages.extend([None, None])

    # ADS4 - TEC7 和 TEC8
    if 'ADS4' in ads_devices:
        voltages.append(read_multi_gain_voltage(ads_devices['ADS4'], '0-1'))  # TEC7
        voltages.append(read_multi_gain_voltage(ads_devices['ADS4'], '2-3'))  # TEC8
    else:
        voltages.extend([None, None])

    return voltages


def initialize_csv_file(filename):
    """初始化CSV文件，写入表头"""
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Timestamp', 'DateTime']
            
            # 为每个TEC添加多增益列和最优值列
            for i in range(1, 9):
                tec_name = f'TEC{i}'
                # 各增益下的电压值
                for gain in GAINS_TO_TEST:
                    header.append(f'{tec_name}_Gain{gain}(V)')
                # 最优电压值和对应的增益
                header.append(f'{tec_name}_Optimal(V)')
                header.append(f'{tec_name}_OptimalGain')
            
            writer.writerow(header)
        print(f"CSV文件已初始化: {filename}")
    except Exception as e:
        print(f"CSV文件初始化失败: {e}")


def save_data(filename, timestamp, datetime_str, voltages):
    """保存数据到CSV文件"""
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp, datetime_str]
            
            # 为每个TEC添加数据
            for voltage_data in voltages:
                if voltage_data is None:
                    # 如果整个TEC数据缺失，填充空值
                    row.extend([''] * COLUMNS_PER_TEC)
                else:
                    # 添加各增益下的电压值
                    for gain in GAINS_TO_TEST:
                        v = voltage_data.get(f'gain_{gain}')
                        row.append("" if v is None else f"{v:.9f}")
                    
                    # 添加最优电压值
                    opt_v = voltage_data.get('optimal_voltage')
                    row.append("" if opt_v is None else f"{opt_v:.9f}")
                    
                    # 添加最优增益
                    opt_g = voltage_data.get('optimal_gain')
                    row.append("" if opt_g is None else str(opt_g))
            
            writer.writerow(row)
    except Exception as e:
        print(f"数据保存失败: {e}")


def display_voltages(sample_count, datetime_str, voltages):
    """在控制台友好输出电压"""
    print(f"[{sample_count:04d}] {datetime_str}")
    tec_names = ['TEC1', 'TEC2', 'TEC3', 'TEC4', 'TEC5', 'TEC6', 'TEC7', 'TEC8']
    
    for i in range(0, 8, 2):
        v1_data = voltages[i]
        v2_data = voltages[i + 1]
        
        # 显示TEC1/TEC2 或 TEC3/TEC4 等成对的数据
        for idx, (tec_idx, v_data) in enumerate([(i, v1_data), (i+1, v2_data)]):
            if v_data is None:
                print(f"  {tec_names[tec_idx]}: 读取失败")
            else:
                opt_v = v_data.get('optimal_voltage')
                opt_g = v_data.get('optimal_gain')
                
                if opt_v is not None:
                    print(f"  {tec_names[tec_idx]}: {opt_v:8.6f} V ({opt_v*1000:8.3f} mV) [最优增益: {opt_g}]")
                else:
                    print(f"  {tec_names[tec_idx]}: 所有增益下均读取失败")
        
        # 在每对TEC之间打印空行（除了最后一对）
        if i < len(tec_names) - 2:
            print()


def main():
    # 将文件名生成逻辑移入 main 函数
    start_time = datetime.now()
    data_file = f"TEC_multi_gain_data_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"======================= [{start_time}] Program started ========================")
    
    print("=" * 80)
    print("多增益4个ADS1115数据采集系统")
    print("=" * 80)
    print(f"采集间隔: {SAMPLE_INTERVAL}秒")
    print(f"数据文件: {data_file}")
    print(f"采集通道: 8个TEC芯片")
    print(f"增益设置: {', '.join([f'{g} ({GAIN_SETTINGS[g][1]})' for g in GAINS_TO_TEST])}")
    print("  ADS1115 #1 (0x48): TEC1 (A0-A1), TEC2 (A2-A3)")
    print("  ADS1115 #2 (0x49): TEC3 (A0-A1), TEC4 (A2-A3)")
    print("  ADS1115 #3 (0x4A): TEC5 (A0-A1), TEC6 (A2-A3)")
    print("  ADS1115 #4 (0x4B): TEC7 (A0-A1), TEC8 (A2-A3)")
    print("按Ctrl+C停止采集")
    print("=" * 80)

    # 初始化所有ADS1115
    ads_devices = setup_all_ads1115()
    if ads_devices is None or len(ads_devices) == 0:
        print("程序退出")
        return

    print()
    # 初始化CSV文件
    initialize_csv_file(data_file)

    sample_count = 0
    try:
        print("\n开始采集数据...\n")
        while True:
            timestamp = time.time()
            datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            voltages = read_all_voltages_multi_gain(ads_devices)

            if any(v is not None for v in voltages):
                save_data(data_file, timestamp, datetime_str, voltages)
                sample_count += 1
                display_voltages(sample_count, datetime_str, voltages)
            else:
                print(f"[{datetime_str}] 所有通道读取失败，跳过本次采集")

            time.sleep(SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("数据采集已停止")
        print(f"总共采集了 {sample_count} 个样本")
        print(f"数据已保存至: {data_file}")
        print("=" * 80)
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("程序异常退出")


if __name__ == "__main__":
    main()

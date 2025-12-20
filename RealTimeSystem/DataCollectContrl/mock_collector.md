# 模拟数据采集脚本 (mock_collector.py)

## 概述

该脚本用于在没有 ADS1115 硬件的情况下，模拟 `Full_collector.py` 的行为，生成模拟的热电芯片电压数据。

## 功能特点

1. **模拟电压生成** - 生成具有趋势和噪声的真实感电压数据
2. **CSV 格式兼容** - 输出格式与 `Full_collector.py` 完全一致
3. **可配置参数** - 支持自定义采集间隔和输出目录

## 模拟数据特征

生成的模拟电压数据具有以下特征：
- **基准值**: 0.3V ~ 0.7V 之间
- **缓慢趋势**: 5-10分钟周期的正弦波动
- **随机噪声**: 约 5mV 标准差的高斯噪声
- **偶发突变**: 5% 概率出现约 20mV 的突变

## 使用方法

### 基本使用

```bash
# 使用默认设置（10秒间隔，输出到当前目录）
python mock_collector.py

# 指定输出目录
python mock_collector.py --output-dir /home/pi/dev/ads1115_project/Themoelectric

# 指定采集间隔
python mock_collector.py --interval 5

# 安静模式（减少输出）
python mock_collector.py --quiet
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `.` | 数据文件输出目录 |
| `--interval` | 10 | 采集间隔（秒） |
| `--quiet` | False | 安静模式 |

## 输出格式

### CSV 文件

文件名格式: `TEC_multi_gain_data_{YYYYMMDD_HHMMSS}.csv`

列结构（与 Full_collector.py 完全一致）:
- `Timestamp` - Unix 时间戳
- `DateTime` - 格式化时间字符串
- 对于每个 TEC (1-8):
  - `TEC{N}_Gain16(V)` - 增益16下的电压
  - `TEC{N}_Gain8(V)` - 增益8下的电压
  - `TEC{N}_Gain4(V)` - 增益4下的电压
  - `TEC{N}_Gain2(V)` - 增益2下的电压
  - `TEC{N}_Gain1(V)` - 增益1下的电压
  - `TEC{N}_Optimal(V)` - 最优电压值
  - `TEC{N}_OptimalGain` - 最优增益

### 控制台输出

```
[0001] 2025-12-18 12:00:00
------------------------------------------------------------
  TEC1/Yellow : 0.450123 V (450.123 mV)
  TEC2/UV     : 0.520456 V (520.456 mV)

  TEC3/IR     : 0.480789 V (480.789 mV)
  TEC4/Red    : 0.550012 V (550.012 mV)
  ...
```

## 与 pi_sender.py 配合使用

### 在树莓派上测试

```bash
# 终端1: 启动模拟数据采集
python mock_collector.py --output-dir /home/pi/dev/ads1115_project/Themoelectric --interval 10

# 终端2: 启动数据转发
python pi_sender.py --host <主机IP> --port 5000 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
```

## 代码结构

### MockVoltageGenerator 类

核心的电压生成器类：

```python
class MockVoltageGenerator:
    def __init__(self, num_channels: int = 8):
        # 初始化基准电压和趋势参数
        
    def generate(self) -> list:
        # 生成一组8通道的电压数据
        
    def _generate_channel_data(self, channel_idx: int, current_time: float) -> dict:
        # 生成单个通道的完整数据（包含各增益值）
```

## 注意事项

1. 此脚本仅用于测试，不能替代真实的 ADS1115 硬件采集
2. 模拟数据的统计特性与真实数据可能有差异
3. 如需更真实的模拟，可以基于历史数据进行参数调整

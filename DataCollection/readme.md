# DataCollection - 数据采集系统

本目录包含TEC1-04906热电芯片数据采集系统的完整方案，包括硬件连线方案和Python控制代码。

## 目录结构

```
DataCollection/
├── readme.md                      # 本文件，使用说明
├── 单ADS1115连线方案.md             # 单个ADS1115的连线方案（2个TEC芯片）
├── 完整4ADS1115连线方案.md          # 完整方案的连线方案（4个ADS1115 + 8个TEC芯片）
├── single_ads1115_collector.py   # 单个ADS1115的数据采集脚本
└── full_4ads1115_collector.py    # 完整4个ADS1115的数据采集脚本
```

## 项目概述

本项目旨在搭建一个自动化的多通道数据采集系统，用于采集8个TEC1-04906热电芯片产生的电压信号。系统采用Raspberry Pi 5作为主控，通过I2C总线连接4个ADS1115高精度ADC模块，实现8路差分电压的同步采集。

### 技术规格

- **主控**: Raspberry Pi 5
- **ADC模块**: ADS1115（16位分辨率，I2C接口）
- **传感器**: TEC1-04906热电芯片 × 8
- **采集频率**: 10秒/次
- **采集周期**: 7天连续采集
- **数据存储**: CSV格式，保存到SD卡

## 两种实现方案

### 方案一：单ADS1115方案（测试版）

适用于初期测试和验证，使用1个ADS1115采集2个TEC芯片的数据。

- **硬件**: 1个ADS1115 + 2个TEC芯片
- **连线方案**: 参见 `单ADS1115连线方案.md`
- **控制代码**: `single_ads1115_collector.py`

### 方案二：完整4个ADS1115方案（生产版）

完整的生产环境方案，使用4个ADS1115采集8个TEC芯片的数据。

- **硬件**: 4个ADS1115 + 8个TEC芯片
- **连线方案**: 参见 `完整4ADS1115连线方案.md`
- **控制代码**: `full_4ads1115_collector.py`

## Raspberry Pi 5 环境配置

### 1. 启用I2C接口

```bash
sudo raspi-config
```

选择: `3 - Interface Options` → `I5 - I2C` → `Yes`

重启树莓派：
```bash
sudo reboot
```

### 2. 安装Python依赖库

更新系统包：
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

安装Python I2C库：
```bash
sudo apt-get install -y python3-pip python3-dev i2c-tools
```

安装ADS1115驱动库：
```bash
pip3 install adafruit-circuitpython-ads1x15
pip3 install adafruit-blinka
```

### 3. 验证I2C连接

连接好ADS1115后，运行以下命令检测设备：

```bash
sudo i2cdetect -y 1
```

**单ADS1115方案**应该看到：
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
40: -- -- -- -- -- -- -- -- 48 -- -- -- -- -- -- --
```

**完整4个ADS1115方案**应该看到：
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
40: -- -- -- -- -- -- -- -- 48 49 4a 4b -- -- -- --
```

## 使用方法

### 单ADS1115方案

1. **硬件连接**: 按照 `单ADS1115连线方案.md` 完成连线

2. **运行采集脚本**:
```bash
cd /home/runner/work/Thermoelectric_Project/Thermoelectric_Project/DataCollection
python3 single_ads1115_collector.py
```

3. **停止采集**: 按 `Ctrl+C`

4. **查看数据**: 数据保存在 `data.csv` 文件中

### 完整4个ADS1115方案

1. **硬件连接**: 按照 `完整4ADS1115连线方案.md` 完成连线

2. **运行采集脚本**:
```bash
cd /home/runner/work/Thermoelectric_Project/Thermoelectric_Project/DataCollection
python3 full_4ads1115_collector.py
```

3. **停止采集**: 按 `Ctrl+C`

4. **查看数据**: 数据保存在 `data.csv` 文件中

## 数据格式说明

### 单ADS1115方案的CSV文件格式

| 字段名 | 说明 | 示例 |
|-------|------|------|
| Timestamp | Unix时间戳 | 1699876543.123456 |
| DateTime | 人类可读时间 | 2024-11-13 10:15:43 |
| TEC1_Voltage(V) | TEC1芯片电压（伏特） | 0.003245 |
| TEC2_Voltage(V) | TEC2芯片电压（伏特） | 0.002987 |

### 完整4个ADS1115方案的CSV文件格式

| 字段名 | 说明 |
|-------|------|
| Timestamp | Unix时间戳 |
| DateTime | 人类可读时间 |
| TEC1_Voltage(V) | TEC1芯片电压 |
| TEC2_Voltage(V) | TEC2芯片电压 |
| TEC3_Voltage(V) | TEC3芯片电压 |
| TEC4_Voltage(V) | TEC4芯片电压 |
| TEC5_Voltage(V) | TEC5芯片电压 |
| TEC6_Voltage(V) | TEC6芯片电压 |
| TEC7_Voltage(V) | TEC7芯片电压 |
| TEC8_Voltage(V) | TEC8芯片电压 |

## 长期运行设置（7天连续采集）

### 方法1: 使用systemd服务（推荐）

创建服务文件：
```bash
sudo nano /etc/systemd/system/tec-collector.service
```

写入以下内容：
```ini
[Unit]
Description=TEC Data Collector
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Thermoelectric_Project/DataCollection
ExecStart=/usr/bin/python3 full_4ads1115_collector.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用并启动服务：
```bash
sudo systemctl enable tec-collector.service
sudo systemctl start tec-collector.service
```

查看服务状态：
```bash
sudo systemctl status tec-collector.service
```

查看日志：
```bash
sudo journalctl -u tec-collector.service -f
```

停止服务：
```bash
sudo systemctl stop tec-collector.service
```

### 方法2: 使用screen会话

安装screen：
```bash
sudo apt-get install screen
```

启动screen会话：
```bash
screen -S tec_collector
```

在screen会话中运行脚本：
```bash
python3 full_4ads1115_collector.py
```

断开screen会话（保持程序运行）：按 `Ctrl+A`，然后按 `D`

重新连接到会话：
```bash
screen -r tec_collector
```

### 方法3: 使用nohup后台运行

```bash
nohup python3 full_4ads1115_collector.py > collector.log 2>&1 &
```

查看日志：
```bash
tail -f collector.log
```

## 故障排除

### 问题1: 找不到I2C设备

**症状**: `i2cdetect` 命令找不到设备

**解决方法**:
1. 确认I2C已启用：`sudo raspi-config`
2. 检查连线，特别是SDA、SCL、VDD、GND
3. 检查ADS1115供电是否正常
4. 尝试重启树莓派

### 问题2: 导入模块失败

**症状**: `ModuleNotFoundError: No module named 'adafruit_ads1x15'`

**解决方法**:
```bash
pip3 install --upgrade adafruit-circuitpython-ads1x15
pip3 install --upgrade adafruit-blinka
```

### 问题3: 权限错误

**症状**: `PermissionError: [Errno 13] Permission denied`

**解决方法**:
```bash
sudo usermod -a -G i2c,gpio pi
sudo reboot
```

### 问题4: 电压读数异常

**症状**: 读取的电压值为0或异常大

**解决方法**:
1. 检查TEC芯片连接（P脚和N脚不要接反）
2. 确认差分连接正确（A0-A1为一对，A2-A3为一对）
3. 检查TEC芯片是否有温差（冷热端）
4. 尝试调整增益设置

### 问题5: 多个ADS1115地址冲突

**症状**: `i2cdetect` 只显示一个或部分设备

**解决方法**:
1. 检查每个ADS1115的ADDR引脚连接
2. 确认4个ADS1115的ADDR分别连接到：GND、VDD、SDA、SCL
3. 使用万用表测试ADDR引脚电压

## 数据分析建议

采集到的数据可以用于：

1. **时序分析**: 分析电压随时间的变化趋势
2. **温度计算**: 根据塞贝克系数计算温差
3. **机器学习训练**: 作为CNN/LSTM模型的输入数据
4. **性能评估**: 评估不同TEC芯片的性能差异

数据处理示例（Python）：
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('data.csv')

# 转换时间戳为datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])

# 绘制时序图
plt.figure(figsize=(12, 6))
for col in df.columns[2:]:  # 跳过Timestamp和DateTime
    plt.plot(df['DateTime'], df[col], label=col)
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.savefig('voltage_timeseries.png')
```

## 技术支持

如有问题，请检查：
1. 连线方案文档
2. Raspberry Pi 5 官方文档
3. ADS1115 数据手册
4. Adafruit CircuitPython库文档

## 版本历史

- v1.0 (2024-11): 初始版本，支持单ADS1115和完整4个ADS1115方案
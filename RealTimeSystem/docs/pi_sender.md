# Pi 数据发送模块 (pi_sender.py)

## 概述

该模块运行在 Raspberry Pi 上，负责从 `Full_collector.py` 写入的 CSV 文件中读取最新的电压数据，并通过 HTTP POST 发送到主机端服务器。

## 功能特点

1. **CSV 文件读取** - 从 `Full_collector.py` 生成的 CSV 文件中读取最新数据
2. **HTTP POST 数据发送** - 将 8 通道电压数据以 JSON 格式发送到主机端
3. **自动重试机制** - 网络异常时自动重试，提高数据传输可靠性
4. **本地 CSV 备份** - 可选的本地数据备份，防止数据丢失
5. **连接状态检查** - 启动时检查与主机端的连接状态
6. **发送统计** - 记录发送成功/失败次数和成功率

## 数据格式

发送的 JSON 数据格式：

```json
{
  "timestamp": "2025-01-12 14:23:10",
  "values": [v1, v2, v3, v4, v5, v6, v7, v8]
}
```

通道顺序（对应 TEC1-TEC8）：
1. Yellow (黄色) - TEC1
2. Ultraviolet (紫外) - TEC2
3. Infrared (红外) - TEC3
4. Red (红色) - TEC4
5. Green (绿色) - TEC5
6. Blue (蓝色) - TEC6
7. Transparent (透明) - TEC7
8. Violet (紫色) - TEC8

## 使用方法

### 基本使用（从 CSV 文件读取）

```bash
# 使用默认 CSV 目录
python pi_sender.py --host 192.168.1.100 --port 5000

# 指定 CSV 目录
python pi_sender.py --host 192.168.1.100 --port 5000 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
```

### 完整参数

```bash
python pi_sender.py \
    --host 192.168.1.100 \                              # 主机端IP地址 (必需)
    --port 5000 \                                       # 主机端端口 (默认: 5000)
    --interval 10 \                                     # 发送间隔/秒 (默认: 10)
    --max-retries 3 \                                   # 最大重试次数 (默认: 3)
    --backup-dir ./backup \                             # 备份目录 (默认: ./backup)
    --no-backup \                                       # 禁用本地备份
    --csv-dir /path/to/csv \                            # CSV文件目录
    --test                                              # 使用模拟数据测试
```

### 测试模式

```bash
# 使用模拟数据进行测试（不需要 CSV 文件）
python pi_sender.py --host 192.168.1.100 --test
```

## 数据采集器类

### CSVDataCollector

从 `Full_collector.py` 写入的 CSV 文件中读取数据：

```python
from pi_sender import CSVDataCollector

collector = CSVDataCollector(csv_dir='/home/pi/dev/ads1115_project/Themoelectric')
values, timestamp = collector.collect()
# values: [0.45, 0.52, 0.48, 0.55, 0.50, 0.47, 0.53, 0.49]
# timestamp: "2025-12-18 12:00:00"
```

CSV 文件格式要求：
- 文件名模式: `TEC_multi_gain_data_*.csv`
- 必须包含 `TEC{N}_Optimal(V)` 列（N=1-8）
- 必须包含 `DateTime` 列

### MockDataCollector

用于测试的模拟数据采集器：

```python
from pi_sender import MockDataCollector

collector = MockDataCollector()
values, timestamp = collector.collect()
```

## 系统服务配置

### tec-sender.service

```ini
# /etc/systemd/system/tec-sender.service
[Unit]
Description=TEC Data Sender Service
After=network.target tec-collector.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/dev/ads1115_project/Themoelectric
ExecStart=/home/pi/dev/ads1115_project/py311/bin/python3 /path/to/pi_sender.py --host 192.168.1.100 --port 5000 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable tec-sender
sudo systemctl start tec-sender
```

## 与模拟采集脚本配合使用

当没有 ADS1115 硬件时，可以使用 `mock_collector.py` 生成模拟数据：

```bash
# 终端1: 启动模拟数据采集
python DataCollectContrl/mock_collector.py --output-dir /home/pi/dev/ads1115_project/Themoelectric

# 终端2: 启动数据转发
python pi_sender.py --host 192.168.1.100 --csv-dir /home/pi/dev/ads1115_project/Themoelectric
```

## 日志

程序会同时输出到控制台和 `pi_sender.log` 文件。

## 依赖

```
requests>=2.25.0
```

安装依赖：

```bash
pip install requests
```

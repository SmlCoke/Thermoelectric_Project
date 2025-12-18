# Pi 数据发送模块 (pi_sender.py)

## 概述

该模块运行在 Raspberry Pi 5 上，负责将采集到的热电芯片电压数据通过 HTTP POST 发送到主机端服务器。

## 功能特点

1. **HTTP POST 数据发送** - 将 8 通道电压数据以 JSON 格式发送到主机端
2. **自动重试机制** - 网络异常时自动重试，提高数据传输可靠性
3. **本地 CSV 备份** - 可选的本地数据备份，防止数据丢失
4. **连接状态检查** - 启动时检查与主机端的连接状态
5. **发送统计** - 记录发送成功/失败次数和成功率

## 数据格式

发送的 JSON 数据格式：

```json
{
  "timestamp": "2025-01-12 14:23:10",
  "values": [v1, v2, v3, v4, v5, v6, v7, v8]
}
```

通道顺序：
1. Yellow (黄色)
2. Ultraviolet (紫外)
3. Infrared (红外)
4. Red (红色)
5. Green (绿色)
6. Blue (蓝色)
7. Transparent (透明)
8. Violet (紫色)

## 使用方法

### 基本使用

```bash
python pi_sender.py --host 192.168.1.100 --port 5000
```

### 完整参数

```bash
python pi_sender.py \
    --host 192.168.1.100 \      # 主机端IP地址 (必需)
    --port 5000 \               # 主机端端口 (默认: 5000)
    --interval 10 \             # 采集间隔/秒 (默认: 10)
    --max-retries 3 \           # 最大重试次数 (默认: 3)
    --backup-dir ./backup \     # 备份目录 (默认: ./backup)
    --no-backup \               # 禁用本地备份
    --test                      # 使用模拟数据测试
```

### 测试模式

```bash
# 使用模拟数据进行测试
python pi_sender.py --host 192.168.1.100 --test
```

## 与 Full_collector.py 集成

在实际部署时，需要修改 `MockDataCollector` 类，替换为真实的数据采集接口。

示例集成方式：

```python
class RealDataCollector:
    def __init__(self, collector_script_path):
        # 初始化与采集脚本的接口
        pass
    
    def collect(self) -> List[float]:
        # 从 Full_collector.py 获取真实数据
        # 可以通过读取共享文件、管道或其他IPC方式
        pass
```

## 系统服务配置

如需将此脚本配置为系统服务，可创建 systemd 服务文件：

```ini
# /etc/systemd/system/tec-sender.service
[Unit]
Description=TEC Data Sender Service
After=network.target tec-collector.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/RealTimeSystem
ExecStart=/usr/bin/python3 pi_sender.py --host 192.168.1.100 --port 5000
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

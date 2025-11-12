# Raspberry Pi 3B 自动化数据采集方案

## 概述

本文档提供了多种让Raspberry Pi 3B独立运行7天数据采集任务的自动化方案，同时保证可以随时通过网线连接监控工作进度。

**核心需求**:
- ✅ Pi3B开机自动启动数据采集程序
- ✅ 不需要时时刻刻连接网线
- ✅ 随时可以插上网线通过SSH监控进度
- ✅ 采集间隔：10秒/次，连续运行7天
- ✅ 数据自动保存到SD卡

**工作目录配置**:
- Python虚拟环境: `/home/pi/dev/ads1115_project/py311`
- 采集脚本目录: `/home/pi/dev/ads1115_project/Themoelectric`
- 数据文件: `/home/pi/dev/ads1115_project/Themoelectric/data.csv`

---

## 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|-----|------|------|--------|
| **systemd服务** | 开机自动启动，崩溃自动重启，日志完善 | 配置稍复杂 | ⭐⭐⭐⭐⭐ |
| **cron定时任务** | 配置简单，灵活 | 不自动重启，需配合@reboot | ⭐⭐⭐⭐ |
| **rc.local** | 最简单 | 不推荐（已过时） | ⭐⭐ |
| **screen会话** | 灵活，适合调试 | 需手动启动 | ⭐⭐⭐ |
| **supervisor** | 功能强大 | 需额外安装 | ⭐⭐⭐ |

---

## 方案一：systemd服务（强烈推荐）

### 1.1 优势
- ✅ 开机自动启动
- ✅ 程序崩溃自动重启
- ✅ 完善的日志系统
- ✅ 易于启动/停止/查看状态
- ✅ 系统级集成，稳定可靠

### 1.2 创建systemd服务文件

创建服务配置文件：
```bash
sudo nano /etc/systemd/system/tec-collector.service
```

写入以下内容：
```ini
[Unit]
Description=TEC Thermoelectric Data Collector
Documentation=https://github.com/SmlCoke/Thermoelectric_Project
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/dev/ads1115_project/Themoelectric
Environment="PATH=/home/pi/dev/ads1115_project/py311/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/pi/dev/ads1115_project/py311/bin/python3 /home/pi/dev/ads1115_project/Themoelectric/Full_collector.py

# 自动重启策略
Restart=always
RestartSec=10

# 标准输出和错误输出重定向到日志
StandardOutput=journal
StandardError=journal

# 资源限制（可选）
Nice=0
CPUQuota=50%

[Install]
WantedBy=multi-user.target
```

**关键参数说明**:
- `After=network-online.target`: 等待网络就绪后再启动
- `Restart=always`: 无论何种原因退出都自动重启
- `RestartSec=10`: 重启前等待10秒
- `StandardOutput=journal`: 日志输出到systemd journal

### 1.3 启用并启动服务

重新加载systemd配置：
```bash
sudo systemctl daemon-reload
```

启用服务（开机自动启动）：
```bash
sudo systemctl enable tec-collector.service
```

立即启动服务：
```bash
sudo systemctl start tec-collector.service
```

### 1.4 管理服务

**查看服务状态**:
```bash
sudo systemctl status tec-collector.service
```

**停止服务**:
```bash
sudo systemctl stop tec-collector.service
```

**重启服务**:
```bash
sudo systemctl restart tec-collector.service
```

**禁用服务**（取消开机自动启动）:
```bash
sudo systemctl disable tec-collector.service
```

**查看实时日志**:
```bash
# 实时查看最新日志
sudo journalctl -u tec-collector.service -f

# 查看最近100行日志
sudo journalctl -u tec-collector.service -n 100

# 查看今天的日志
sudo journalctl -u tec-collector.service --since today

# 查看特定时间范围的日志
sudo journalctl -u tec-collector.service --since "2024-11-01" --until "2024-11-02"
```

**查看服务是否自动启动**:
```bash
systemctl is-enabled tec-collector.service
# 输出: enabled
```

### 1.5 远程监控命令

插上网线后，通过SSH连接：
```bash
ssh pi@<树莓派IP地址>
```

监控采集进度：
```bash
# 1. 查看服务状态
sudo systemctl status tec-collector.service

# 2. 查看实时日志
sudo journalctl -u tec-collector.service -f

# 3. 查看数据文件
tail -n 20 /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 4. 实时监控新数据
tail -f /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 5. 统计已采集样本数
wc -l /home/pi/dev/ads1115_project/Themoelectric/data.csv
```

---

## 方案二：cron定时任务

### 2.1 优势
- ✅ 配置简单
- ✅ 系统自带，无需额外安装
- ✅ 灵活的定时控制

### 2.2 配置方法

编辑crontab：
```bash
crontab -e
```

添加以下内容：
```bash
# 开机自动启动数据采集
@reboot sleep 30 && cd /home/pi/dev/ads1115_project/Themoelectric && /home/pi/dev/ads1115_project/py311/bin/python3 Full_collector.py >> /home/pi/collector.log 2>&1 &
```

**说明**:
- `@reboot`: 系统启动时执行
- `sleep 30`: 等待30秒，确保系统完全启动
- `>>`: 将输出追加到日志文件
- `2>&1`: 将错误输出也重定向到日志文件
- `&`: 后台运行

### 2.3 查看日志

```bash
tail -f /home/pi/collector.log
```

### 2.4 停止任务

```bash
# 查找进程ID
ps aux | grep Full_collector.py

# 停止进程
kill <进程ID>
```

### 2.5 验证cron任务

```bash
# 列出所有cron任务
crontab -l

# 查看cron服务状态
sudo systemctl status cron
```

---

## 方案三：screen会话（适合调试）

### 3.1 优势
- ✅ 可以随时连接/断开会话
- ✅ 适合开发调试阶段
- ✅ 可以直接查看程序输出

### 3.2 安装screen

```bash
sudo apt-get update
sudo apt-get install -y screen
```

### 3.3 使用方法

**启动screen会话**:
```bash
screen -S tec_data_collector
```

**在screen会话中启动采集程序**:
```bash
cd /home/pi/dev/ads1115_project/Themoelectric
source /home/pi/dev/ads1115_project/py311/bin/activate
python3 Full_collector.py
```

**断开screen会话**（程序继续运行）:
按 `Ctrl + A`，然后按 `D`

**列出所有screen会话**:
```bash
screen -ls
```

**重新连接到会话**:
```bash
screen -r tec_data_collector
```

**终止会话**:
```bash
screen -X -S tec_data_collector quit
```

### 3.4 开机自动启动screen会话

编辑 `/etc/rc.local`:
```bash
sudo nano /etc/rc.local
```

在 `exit 0` 之前添加：
```bash
# 开机自动启动screen会话运行数据采集
su - pi -c "screen -dmS tec_data_collector bash -c 'cd /home/pi/dev/ads1115_project/Themoelectric && source /home/pi/dev/ads1115_project/py311/bin/activate && python3 Full_collector.py'"
```

**说明**:
- `-dmS`: 创建detached模式的会话
- `su - pi -c`: 以pi用户身份运行

---

## 方案四：supervisor进程管理

### 4.1 优势
- ✅ 专业的进程管理工具
- ✅ Web界面监控（可选）
- ✅ 自动重启，日志管理

### 4.2 安装supervisor

```bash
sudo apt-get update
sudo apt-get install -y supervisor
```

### 4.3 创建配置文件

```bash
sudo nano /etc/supervisor/conf.d/tec-collector.conf
```

写入内容：
```ini
[program:tec-collector]
command=/home/pi/dev/ads1115_project/py311/bin/python3 /home/pi/dev/ads1115_project/Themoelectric/Full_collector.py
directory=/home/pi/dev/ads1115_project/Themoelectric
user=pi
autostart=true
autorestart=true
startretries=3
redirect_stderr=true
stdout_logfile=/var/log/supervisor/tec-collector.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
```

### 4.4 启动supervisor

```bash
# 重新加载配置
sudo supervisorctl reread
sudo supervisorctl update

# 启动程序
sudo supervisorctl start tec-collector

# 查看状态
sudo supervisorctl status tec-collector
```

### 4.5 管理命令

```bash
# 停止
sudo supervisorctl stop tec-collector

# 重启
sudo supervisorctl restart tec-collector

# 查看日志
sudo tail -f /var/log/supervisor/tec-collector.log
```

---

## 方案五：tmux会话（screen的现代替代）

### 5.1 安装tmux

```bash
sudo apt-get install -y tmux
```

### 5.2 使用方法

**启动tmux会话**:
```bash
tmux new -s tec_collector
```

**在tmux会话中运行程序**:
```bash
cd /home/pi/dev/ads1115_project/Themoelectric
source /home/pi/dev/ads1115_project/py311/bin/activate
python3 Full_collector.py
```

**断开会话**:
按 `Ctrl + B`，然后按 `D`

**重新连接**:
```bash
tmux attach -t tec_collector
```

**列出所有会话**:
```bash
tmux ls
```

---

## 推荐配置组合

### 最佳实践配置

**生产环境（7天连续采集）**:
```
主方案: systemd服务
备份方案: cron @reboot
监控方式: SSH远程连接 + journalctl日志
```

**具体步骤**:

1. **配置systemd服务**（方案一）
2. **添加cron备份任务**（方案二），防止systemd失败
3. **测试自动启动**:
   ```bash
   # 重启Pi3B
   sudo reboot
   
   # 等待1-2分钟后SSH连接
   ssh pi@<树莓派IP>
   
   # 检查服务状态
   sudo systemctl status tec-collector.service
   
   # 查看数据是否正在写入
   tail -f /home/pi/dev/ads1115_project/Themoelectric/data.csv
   ```

4. **拔掉网线，让Pi3B独立运行**

5. **需要监控时，插上网线**:
   ```bash
   ssh pi@<树莓派IP>
   sudo journalctl -u tec-collector.service -f
   ```

---

## 网络配置（支持随时插拔网线监控）

### 配置静态IP（推荐）

编辑网络配置：
```bash
sudo nano /etc/dhcpcd.conf
```

在文件末尾添加（根据你的网络环境修改）：
```
# 配置静态IP for eth0
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

重启网络服务：
```bash
sudo systemctl restart dhcpcd
```

**好处**:
- 每次插上网线，IP地址固定不变
- 不需要每次查找IP地址
- 方便SSH连接

### 配置SSH保持连接

编辑SSH配置：
```bash
sudo nano /etc/ssh/sshd_config
```

修改以下参数：
```
ClientAliveInterval 60
ClientAliveCountMax 3
```

重启SSH服务：
```bash
sudo systemctl restart ssh
```

**好处**:
- SSH连接不会轻易断开
- 方便长时间监控

---

## 监控脚本

### 创建便捷监控脚本

创建监控脚本文件：
```bash
nano /home/pi/monitor_collector.sh
```

写入内容：
```bash
#!/bin/bash
# 数据采集监控脚本

echo "=========================================="
echo "TEC数据采集系统监控"
echo "=========================================="
echo ""

# 1. 服务状态
echo "【服务状态】"
sudo systemctl status tec-collector.service --no-pager
echo ""

# 2. 运行时间
echo "【程序运行时间】"
ps -eo pid,etime,cmd | grep Full_collector.py | grep -v grep
echo ""

# 3. 数据文件信息
echo "【数据文件信息】"
DATA_FILE="/home/pi/dev/ads1115_project/Themoelectric/data.csv"
if [ -f "$DATA_FILE" ]; then
    echo "文件大小: $(du -h $DATA_FILE | cut -f1)"
    echo "数据行数: $(wc -l < $DATA_FILE)"
    echo "最新数据:"
    tail -n 5 $DATA_FILE
else
    echo "数据文件不存在！"
fi
echo ""

# 4. 磁盘空间
echo "【SD卡存储空间】"
df -h | grep /dev/root
echo ""

# 5. CPU温度
echo "【系统温度】"
temp=$(vcgencmd measure_temp)
echo "CPU温度: $temp"
echo ""

# 6. 内存使用
echo "【内存使用】"
free -h
echo ""

# 7. 最近日志
echo "【最近日志（最新10行）】"
sudo journalctl -u tec-collector.service -n 10 --no-pager
echo ""

echo "=========================================="
echo "监控完成"
echo "=========================================="
```

添加执行权限：
```bash
chmod +x /home/pi/monitor_collector.sh
```

**使用方法**:
```bash
# 运行监控脚本
/home/pi/monitor_collector.sh

# 或创建别名，方便调用
echo "alias monitor='/home/pi/monitor_collector.sh'" >> ~/.bashrc
source ~/.bashrc

# 之后直接输入
monitor
```

---

## 数据备份策略

### 定期备份到网络存储（可选）

如果有NAS或网络存储，可以配置定期备份：

```bash
crontab -e
```

添加：
```bash
# 每天凌晨2点备份数据到NAS
0 2 * * * rsync -avz /home/pi/dev/ads1115_project/Themoelectric/data.csv pi@nas_ip:/backup/tec_data/
```

### 定期清理旧日志

```bash
crontab -e
```

添加：
```bash
# 每周清理30天前的systemd日志
0 3 * * 0 sudo journalctl --vacuum-time=30d
```

---

## 故障自动恢复

### 监控脚本检测并重启

创建看门狗脚本：
```bash
nano /home/pi/watchdog.sh
```

内容：
```bash
#!/bin/bash
# 看门狗脚本：检测数据采集是否正常，异常时重启服务

DATA_FILE="/home/pi/dev/ads1115_project/Themoelectric/data.csv"
LOG_FILE="/home/pi/watchdog.log"

# 获取文件当前大小
current_size=$(stat -f%z "$DATA_FILE" 2>/dev/null || stat -c%s "$DATA_FILE" 2>/dev/null)

# 读取上次记录的大小
if [ -f /tmp/data_size ]; then
    last_size=$(cat /tmp/data_size)
else
    last_size=0
fi

# 保存当前大小
echo $current_size > /tmp/data_size

# 如果文件大小没有增长（超过2个采集周期：20秒）
if [ "$current_size" == "$last_size" ]; then
    echo "[$(date)] 检测到数据文件未增长，尝试重启服务..." >> $LOG_FILE
    sudo systemctl restart tec-collector.service
    echo "[$(date)] 服务已重启" >> $LOG_FILE
else
    echo "[$(date)] 数据采集正常，文件大小: $current_size" >> $LOG_FILE
fi
```

添加执行权限：
```bash
chmod +x /home/pi/watchdog.sh
```

添加到cron（每分钟检查一次）：
```bash
crontab -e
```

添加：
```bash
* * * * * /home/pi/watchdog.sh
```

---

## 测试验证清单

### 自动启动测试

```bash
# 1. 重启Pi3B
sudo reboot

# 2. 等待2分钟后SSH连接

# 3. 检查服务是否自动启动
sudo systemctl status tec-collector.service
# 预期: active (running)

# 4. 检查数据是否正在写入
tail -f /home/pi/dev/ads1115_project/Themoelectric/data.csv
# 预期: 每10秒新增一行数据

# 5. 检查进程
ps aux | grep Full_collector.py
# 预期: 看到Python进程
```

### 断网运行测试

```bash
# 1. 拔掉网线，让Pi3B独立运行1小时

# 2. 重新插上网线，SSH连接

# 3. 检查数据采集是否连续
wc -l /home/pi/dev/ads1115_project/Themoelectric/data.csv
# 预期: 行数约为 (运行时间秒数 / 10) + 1（表头）

# 4. 检查日志是否有错误
sudo journalctl -u tec-collector.service --since "1 hour ago"
```

### 崩溃恢复测试

```bash
# 1. 手动停止进程（模拟崩溃）
sudo systemctl stop tec-collector.service

# 2. 等待15秒

# 3. 检查服务是否自动重启
sudo systemctl status tec-collector.service
# 预期: active (running)，带有restart字样
```

---

## 常见问题

### Q1: 开机后服务未自动启动
**A**: 
```bash
# 检查服务是否启用
systemctl is-enabled tec-collector.service
# 如果显示disabled，运行：
sudo systemctl enable tec-collector.service
```

### Q2: 无法SSH连接Pi3B
**A**:
```bash
# 确认网线连接正常
# 确认Pi3B已开机（观察电源LED）
# 尝试ping Pi3B
ping 192.168.1.100

# 如果ping不通，检查IP地址（需要连接显示器或串口）
```

### Q3: 数据文件未更新
**A**:
```bash
# 检查服务状态
sudo systemctl status tec-collector.service

# 查看详细日志
sudo journalctl -u tec-collector.service -n 50

# 手动运行脚本，查看错误
cd /home/pi/dev/ads1115_project/Themoelectric
source /home/pi/dev/ads1115_project/py311/bin/activate
python3 Full_collector.py
```

### Q4: 磁盘空间不足
**A**:
```bash
# 检查磁盘空间
df -h

# 清理systemd日志
sudo journalctl --vacuum-size=100M

# 如需要，分割数据文件
```

---

## 总结

### 推荐方案

**最佳方案**: **systemd服务（方案一）**

**原因**:
1. ✅ 开机自动启动，无需人工干预
2. ✅ 自动重启机制，保证7天连续运行
3. ✅ 完善的日志系统，方便远程监控
4. ✅ 系统级集成，稳定可靠
5. ✅ 易于管理，支持远程控制

### 快速部署步骤

```bash
# 1. 创建systemd服务文件
sudo nano /etc/systemd/system/tec-collector.service
# （复制本文档方案一的配置内容）

# 2. 重新加载并启用服务
sudo systemctl daemon-reload
sudo systemctl enable tec-collector.service
sudo systemctl start tec-collector.service

# 3. 验证服务状态
sudo systemctl status tec-collector.service

# 4. 查看实时日志
sudo journalctl -u tec-collector.service -f

# 5. 测试重启
sudo reboot

# 6. 等待2分钟后SSH连接，验证自动启动
ssh pi@192.168.1.100
sudo systemctl status tec-collector.service
```

### 日常运维

**需要监控时**:
```bash
# 插上网线，SSH连接
ssh pi@192.168.1.100

# 快速检查
/home/pi/monitor_collector.sh

# 查看实时数据
tail -f /home/pi/dev/ads1115_project/Themoelectric/data.csv
```

**7天采集结束后**:
```bash
# 停止服务
sudo systemctl stop tec-collector.service

# 备份数据
scp pi@192.168.1.100:/home/pi/dev/ads1115_project/Themoelectric/data.csv ./
```

---

**更新日期**: 2024-11  
**适用系统**: Raspberry Pi 3B / Raspberry Pi OS  
**Python环境**: py311 虚拟环境

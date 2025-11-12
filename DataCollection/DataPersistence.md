# 数据持久化方案说明

## 概述

本文档详细说明了TEC数据采集系统的数据持久化策略，确保在Raspberry Pi 3B内存有限（1GB）的情况下，能够可靠地将采集数据写入SD卡，实现7天连续采集。

---

## 一、数据持久化挑战

### 1.1 面临的问题

1. **有限的RAM**: Raspberry Pi 3B只有1GB内存
2. **长期运行**: 7天连续采集，约60,480个数据点（7天 × 24小时 × 360次/小时）
3. **数据安全**: 防止意外断电或程序崩溃导致数据丢失
4. **写入延迟**: 文件系统缓存可能延迟数据写入SD卡

### 1.2 数据量估算

**单个数据点大小**:
- 时间戳: ~20字节
- 日期时间: ~25字节
- 8个电压值: ~80字节（每个约10字节）
- 分隔符和换行: ~10字节
- **总计**: 约135字节/行

**7天数据总量**:
- 采集次数: 7天 × 24小时 × 3600秒 / 10秒 = 60,480次
- 数据大小: 60,480 × 135字节 ≈ 8.16 MB

**结论**: 数据量不大（<10MB），但需要确保实时写入，避免内存积累。

---

## 二、Python脚本数据持久化优化

### 2.1 原有实现（已完成）

原有的 `Full_collector.py` 和 `single_collector.py` 已经实现了基本的数据写入：

```python
def save_data(filename, timestamp, datetime_str, voltages):
    """保存数据到CSV文件"""
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [timestamp, datetime_str] + [("" if v is None else f"{v:.9f}") for v in voltages]
            writer.writerow(row)
    except Exception as e:
        print(f"数据保存失败: {e}")
```

**特点**:
- ✅ 使用追加模式（'a'）
- ✅ 每次采集后立即调用save_data()
- ✅ 使用with语句，自动关闭文件

### 2.2 增强实现（已更新）

为了确保数据立即写入SD卡，脚本已经进行了以下增强：

```python
def save_data(filename, timestamp, datetime_str, voltages):
    """保存数据到CSV文件，立即写入SD卡确保数据持久化"""
    try:
        # buffering=1 表示行缓冲，每写入一行立即刷新到磁盘
        with open(filename, 'a', newline='', buffering=1) as f:
            writer = csv.writer(f)
            row = [timestamp, datetime_str] + [("" if v is None else f"{v:.9f}") for v in voltages]
            writer.writerow(row)
            # 显式刷新缓冲区，确保数据写入SD卡
            f.flush()
            # 同步到磁盘，防止掉电丢失数据
            import os
            os.fsync(f.fileno())
    except Exception as e:
        print(f"数据保存失败: {e}")
```

**优化要点**:

1. **行缓冲 (`buffering=1`)**
   - 每写入一行后立即刷新缓冲区
   - 减少内存占用
   - 加快数据落盘

2. **显式刷新 (`f.flush()`)**
   - 强制将Python缓冲区的数据写入操作系统缓冲区
   - 确保数据不会滞留在应用层缓冲区

3. **磁盘同步 (`os.fsync()`)**
   - 强制操作系统将数据从缓存写入物理磁盘
   - 防止意外断电导致数据丢失
   - 保证数据持久化到SD卡

### 2.3 性能影响分析

**写入频率**: 10秒/次

**fsync开销**: 通常10-50ms（取决于SD卡性能）

**影响评估**:
- ✅ 10秒间隔足够长，fsync开销可忽略
- ✅ 不会影响采集频率
- ✅ 数据安全性大幅提升

---

## 三、文件系统优化

### 3.1 SD卡文件系统选择

**推荐**: 使用ext4文件系统（Raspberry Pi OS默认）

**原因**:
- ✅ 日志文件系统，数据安全性高
- ✅ 支持fsync，确保数据持久化
- ✅ 性能良好，适合嵌入式系统

**验证文件系统**:
```bash
df -T
# 输出应显示: /dev/root ext4 ...
```

### 3.2 减少SD卡写入磨损

虽然数据需要实时写入，但可以通过以下方法减少SD卡磨损：

#### 方法1: 使用tmpfs存储临时文件

编辑 `/etc/fstab`:
```bash
sudo nano /etc/fstab
```

添加：
```
tmpfs /tmp tmpfs defaults,noatime,nosuid,size=100M 0 0
tmpfs /var/tmp tmpfs defaults,noatime,nosuid,size=100M 0 0
tmpfs /var/log tmpfs defaults,noatime,nosuid,size=100M 0 0
```

**说明**:
- tmpfs存储在RAM中，减少SD卡写入
- 系统日志和临时文件不写入SD卡
- 重启后tmpfs内容会丢失，但不影响数据采集

**注意**: 由于Pi3B内存有限（1GB），tmpfs大小不宜过大。

#### 方法2: 禁用swap（推荐）

```bash
# 禁用swap
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile

# 验证
free -h
# Swap行应显示: 0B
```

**好处**:
- 减少SD卡写入
- 释放SD卡空间
- 提高系统响应速度

#### 方法3: 使用noatime挂载选项

编辑 `/etc/fstab`:
```bash
sudo nano /etc/fstab
```

修改根文件系统挂载选项，添加 `noatime`:
```
/dev/mmcblk0p2  /  ext4  defaults,noatime  0  1
```

重新挂载：
```bash
sudo mount -o remount /
```

**说明**:
- noatime禁用访问时间记录
- 减少SD卡写入次数
- 对数据采集无影响

---

## 四、数据完整性保障

### 4.1 文件完整性检查

创建数据验证脚本：
```bash
nano /home/pi/verify_data.sh
```

内容：
```bash
#!/bin/bash
# 数据完整性检查脚本

DATA_FILE="/home/pi/dev/ads1115_project/Themoelectric/data.csv"

echo "=========================================="
echo "数据完整性检查"
echo "=========================================="

# 1. 检查文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 数据文件不存在: $DATA_FILE"
    exit 1
fi

# 2. 文件基本信息
echo "✅ 文件存在"
echo "文件路径: $DATA_FILE"
echo "文件大小: $(du -h $DATA_FILE | cut -f1)"
echo "数据行数: $(wc -l < $DATA_FILE)"
echo ""

# 3. 检查表头
echo "检查CSV表头..."
HEADER=$(head -n 1 $DATA_FILE)
EXPECTED="Timestamp,DateTime,TEC1_Voltage(V),TEC2_Voltage(V),TEC3_Voltage(V),TEC4_Voltage(V),TEC5_Voltage(V),TEC6_Voltage(V),TEC7_Voltage(V),TEC8_Voltage(V)"

if [ "$HEADER" == "$EXPECTED" ]; then
    echo "✅ 表头正确"
else
    echo "⚠️  表头异常"
    echo "当前表头: $HEADER"
fi
echo ""

# 4. 检查最新数据
echo "最新5行数据:"
tail -n 5 $DATA_FILE
echo ""

# 5. 检查数据连续性（时间戳间隔）
echo "检查时间间隔（最新10个数据点）..."
tail -n 11 $DATA_FILE | head -n 10 | awk -F',' '{print $1}' | awk '
NR>1 {
    diff = $1 - prev;
    if (diff > 15 || diff < 5) {
        printf "⚠️  异常时间间隔: %.2f 秒\n", diff;
    }
    prev = $1;
}
NR==1 {prev = $1}
END {print "时间间隔检查完成"}
'
echo ""

# 6. 检查空值
echo "检查空值..."
EMPTY_COUNT=$(grep ',,' $DATA_FILE | wc -l)
if [ $EMPTY_COUNT -gt 0 ]; then
    echo "⚠️  发现 $EMPTY_COUNT 行包含空值"
else
    echo "✅ 无空值数据"
fi
echo ""

# 7. 预估剩余SD卡空间
echo "SD卡存储状态..."
df -h / | grep -v Filesystem
AVAILABLE=$(df / | tail -n 1 | awk '{print $4}')
echo "可用空间（KB）: $AVAILABLE"

# 计算可存储天数（假设每天约1.2MB）
DAYS=$((AVAILABLE / 1200))
echo "预估可存储天数: $DAYS 天"
echo ""

echo "=========================================="
echo "检查完成"
echo "=========================================="
```

添加执行权限：
```bash
chmod +x /home/pi/verify_data.sh
```

**使用方法**:
```bash
/home/pi/verify_data.sh
```

### 4.2 定期备份

#### 方法1: 本地备份（每天）

```bash
crontab -e
```

添加：
```bash
# 每天凌晨3点备份数据文件
0 3 * * * cp /home/pi/dev/ads1115_project/Themoelectric/data.csv /home/pi/dev/ads1115_project/Themoelectric/data_backup_$(date +\%Y\%m\%d).csv
```

#### 方法2: 远程备份（可选）

如果有网络存储或另一台电脑：

```bash
crontab -e
```

添加：
```bash
# 每12小时通过SCP备份到远程服务器
0 */12 * * * scp /home/pi/dev/ads1115_project/Themoelectric/data.csv user@remote_server:/backup/tec_data/
```

### 4.3 数据恢复

如果数据文件损坏，可以：

1. **从备份恢复**:
```bash
cp /home/pi/dev/ads1115_project/Themoelectric/data_backup_YYYYMMDD.csv /home/pi/dev/ads1115_project/Themoelectric/data.csv
```

2. **重新开始采集**:
```bash
# 停止服务
sudo systemctl stop tec-collector.service

# 删除损坏的文件
rm /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 重新启动服务（会自动创建新文件）
sudo systemctl start tec-collector.service
```

---

## 五、内存管理优化

### 5.1 监控内存使用

创建内存监控脚本：
```bash
nano /home/pi/monitor_memory.sh
```

内容：
```bash
#!/bin/bash
# 内存监控脚本

echo "=========================================="
echo "内存使用情况"
echo "=========================================="

# 1. 总体内存使用
free -h

echo ""
echo "=========================================="
echo "进程内存使用（Top 10）"
echo "=========================================="

# 2. 进程内存使用排序
ps aux --sort=-%mem | head -n 11

echo ""
echo "=========================================="
echo "数据采集程序内存使用"
echo "=========================================="

# 3. 采集程序内存使用
ps aux | grep Full_collector.py | grep -v grep

echo ""
```

添加执行权限并运行：
```bash
chmod +x /home/pi/monitor_memory.sh
/home/pi/monitor_memory.sh
```

### 5.2 内存使用预估

**Python脚本内存占用**:
- Python解释器: ~10-20 MB
- 库（adafruit等）: ~5-10 MB
- 数据缓冲: <1 MB（已优化为行缓冲）
- **总计**: <30 MB

**系统内存占用**:
- 操作系统: ~200-300 MB
- 其他服务: ~50-100 MB
- **总计**: ~300-400 MB

**可用内存**:
- 总内存: 1024 MB
- 已用: ~430 MB
- **剩余**: ~590 MB

**结论**: 内存充足，无需担心内存不足问题。

### 5.3 防止内存泄漏

Python脚本已经采用了良好的编程实践：

1. **使用with语句**: 自动关闭文件，释放资源
2. **无全局变量累积**: 每次采集后变量被覆盖
3. **异常处理**: 捕获异常，防止资源泄漏

**验证内存稳定性**:
```bash
# 运行24小时后检查内存
free -h

# 应该与启动时相差不大（<50MB）
```

---

## 六、磁盘空间管理

### 6.1 监控磁盘空间

```bash
# 查看磁盘空间
df -h

# 查看数据目录空间
du -sh /home/pi/dev/ads1115_project/Themoelectric/
```

### 6.2 自动清理旧日志

```bash
crontab -e
```

添加：
```bash
# 每周清理systemd日志（保留7天）
0 2 * * 0 sudo journalctl --vacuum-time=7d

# 每周清理超过30天的备份文件
0 3 * * 0 find /home/pi/dev/ads1115_project/Themoelectric/ -name "data_backup_*.csv" -mtime +30 -delete
```

### 6.3 磁盘空间不足告警

创建告警脚本：
```bash
nano /home/pi/disk_alert.sh
```

内容：
```bash
#!/bin/bash
# 磁盘空间告警脚本

THRESHOLD=90  # 告警阈值：90%

USAGE=$(df / | tail -n 1 | awk '{print $5}' | sed 's/%//')

if [ $USAGE -gt $THRESHOLD ]; then
    echo "[$(date)] 警告：磁盘使用率 ${USAGE}%，超过阈值 ${THRESHOLD}%" >> /home/pi/disk_alert.log
    # 可以在这里添加发送邮件或短信通知的代码
fi
```

添加到cron（每小时检查）：
```bash
crontab -e
```

添加：
```bash
0 * * * * /home/pi/disk_alert.sh
```

---

## 七、数据采集脚本工作流程

### 7.1 完整流程图

```
启动程序
    ↓
初始化I2C总线
    ↓
初始化ADS1115设备
    ↓
创建/打开data.csv文件
    ↓
写入CSV表头（如果是新文件）
    ↓
┌─────────────────────┐
│  进入采集循环       │
│                     │
│  1. 读取当前时间    │
│  2. 读取8路电压值   │
│  3. 格式化数据      │
│  4. 写入CSV文件     │
│     - 追加模式打开  │
│     - 行缓冲写入    │
│     - flush()刷新   │
│     - fsync()同步   │
│  5. 关闭文件        │
│  6. 显示数据        │
│  7. 等待10秒        │
│                     │
└─────────────────────┘
    ↓
    └──→ 循环继续（按Ctrl+C退出）
```

### 7.2 关键代码片段

```python
def main():
    # 初始化设备
    ads_devices = setup_all_ads1115()
    
    # 初始化CSV文件（创建新文件或追加到现有文件）
    initialize_csv_file(DATA_FILE)
    
    # 采集循环
    while True:
        timestamp = time.time()
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 读取电压
        voltages = read_all_voltages(ads_devices)
        
        # 保存数据（立即写入SD卡）
        save_data(DATA_FILE, timestamp, datetime_str, voltages)
        
        # 等待下一个采集周期
        time.sleep(SAMPLE_INTERVAL)
```

---

## 八、故障场景与应对

### 8.1 意外断电

**场景**: 数据采集过程中突然断电

**数据保障**:
- ✅ 已通过fsync()写入的数据安全
- ⚠️  最后一次采集（10秒内）可能丢失（但概率极低）

**恢复方法**:
1. 重新上电，Pi3B自动启动
2. systemd服务自动启动采集程序
3. 程序以追加模式打开data.csv，继续采集
4. 检查数据连续性：
```bash
/home/pi/verify_data.sh
```

### 8.2 SD卡写入失败

**场景**: SD卡损坏或写保护

**检测方法**:
```bash
# 查看systemd日志
sudo journalctl -u tec-collector.service -n 50

# 查找"数据保存失败"错误信息
```

**应对措施**:
1. 停止服务
2. 检查SD卡状态：
```bash
sudo dmesg | grep mmc
```
3. 如需要，更换SD卡并恢复数据

### 8.3 文件系统只读

**场景**: 文件系统错误导致只读模式

**检测**:
```bash
mount | grep "ro,"
```

**解决**:
```bash
# 重新挂载为读写
sudo mount -o remount,rw /

# 如果失败，需要重启并运行fsck
sudo reboot
# 启动时会自动运行文件系统检查
```

---

## 九、测试与验证

### 9.1 数据写入测试

```bash
# 1. 启动采集程序
sudo systemctl start tec-collector.service

# 2. 等待1分钟（6个数据点）

# 3. 检查数据文件
tail -n 10 /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 4. 验证数据连续性
/home/pi/verify_data.sh
```

### 9.2 断电恢复测试

```bash
# 1. 启动采集，等待5分钟

# 2. 记录当前数据行数
wc -l /home/pi/dev/ads1115_project/Themoelectric/data.csv

# 3. 直接拔电源（模拟断电）

# 4. 重新上电，等待2分钟

# 5. SSH连接，检查数据
wc -l /home/pi/dev/ads1115_project/Themoelectric/data.csv
# 行数应该与断电前相差不多（最多损失1-2行）
```

### 9.3 长期稳定性测试

```bash
# 运行24小时后检查

# 1. 内存使用
free -h

# 2. 磁盘空间
df -h

# 3. 数据完整性
/home/pi/verify_data.sh

# 4. 进程状态
ps aux | grep Full_collector.py

# 5. 服务运行时间
sudo systemctl status tec-collector.service
```

---

## 十、总结

### 10.1 数据持久化保障措施

| 层次 | 措施 | 作用 |
|-----|------|------|
| **应用层** | 行缓冲、flush()、fsync() | 确保数据立即写入SD卡 |
| **文件系统** | ext4日志、noatime | 提高可靠性，减少写入 |
| **硬件层** | 优质SD卡、稳定供电 | 硬件可靠性保障 |
| **监控层** | 数据验证、告警脚本 | 及时发现问题 |
| **备份层** | 定期备份 | 数据安全最后保障 |

### 10.2 关键配置检查清单

**数据采集脚本**:
- [x] 使用追加模式打开文件
- [x] 设置行缓冲（buffering=1）
- [x] 显式调用flush()
- [x] 调用fsync()同步磁盘
- [x] 异常处理

**系统配置**:
- [ ] 禁用swap
- [ ] 配置noatime挂载选项
- [ ] 配置systemd服务自动重启
- [ ] 设置磁盘空间监控

**监控与备份**:
- [ ] 部署数据验证脚本
- [ ] 配置定期备份
- [ ] 设置磁盘告警

### 10.3 预期数据可靠性

在正确配置的情况下：

- **数据丢失率**: <0.01%（极低）
- **最大数据丢失**: 1-2个数据点（断电时）
- **内存占用**: <30 MB（稳定）
- **磁盘使用**: ~8-10 MB/7天（充足）

**结论**: 采用行缓冲+fsync的数据持久化方案，配合systemd自动重启和定期备份，可以确保7天连续采集的数据安全性和可靠性。

---

**文档更新日期**: 2024-11  
**适用脚本**: Full_collector.py, single_collector.py  
**Python版本**: Python 3.7+  
**系统**: Raspberry Pi 3B / Raspberry Pi OS

# Raspberry Pi 3B 低功耗与散热优化配置

## 概述

由于需要在10摄氏度左右的环境中连续采集7天数据，Raspberry Pi 3B必须进行严格的功耗管理和散热优化。本文档提供了详细的可关闭功能清单及相应的配置命令。

## 一、可关闭功能清单

### 1.1 无线通信功能

#### WiFi（推荐关闭）
- **功耗节省**: ~200-300mA
- **适用场景**: 使用有线网络时，可以关闭WiFi
- **影响**: 无法使用无线网络，但可以随时通过有线网络连接

#### 蓝牙（强烈推荐关闭）
- **功耗节省**: ~30-50mA
- **适用场景**: 数据采集不需要蓝牙功能
- **影响**: 无法使用蓝牙设备

### 1.2 显示与图形功能

#### HDMI输出（推荐关闭）
- **功耗节省**: ~25-30mA
- **适用场景**: 无头（headless）运行，通过SSH远程控制
- **影响**: 无法通过HDMI显示器查看画面

#### GPU内存（推荐降低）
- **功耗节省**: ~10-20mA
- **适用场景**: 不需要图形界面的命令行应用
- **影响**: 降低图形性能，但对数据采集无影响

### 1.3 USB与外设功能

#### USB端口（慎重关闭）
- **功耗节省**: ~100-140mA
- **适用场景**: 确认不需要USB设备时
- **注意**: 如果需要使用USB网卡或其他USB设备，请勿关闭

### 1.4 系统服务

#### 无用系统服务（推荐关闭）
- **功耗节省**: ~20-50mA
- **服务列表**:
  - `avahi-daemon`: mDNS/DNS-SD服务
  - `triggerhappy`: 热键守护进程
  - `bluetooth.service`: 蓝牙服务
  - `hciuart.service`: 蓝牙UART服务
  - `cups`: 打印服务
  - `ModemManager`: 调制解调器管理

### 1.5 LED指示灯（可选关闭）

#### 状态LED
- **功耗节省**: ~5-10mA
- **适用场景**: 无需通过LED灯查看状态
- **影响**: 无法通过板载LED查看运行状态

### 1.6 CPU性能调节

#### CPU降频（推荐配置）
- **功耗节省**: ~100-300mA（根据负载）
- **适用场景**: 数据采集任务CPU占用率低
- **影响**: 可能轻微延长采集周期，但10秒间隔足够

---

## 二、配置方法

### 2.1 关闭WiFi

#### 临时关闭（重启后恢复）
```bash
sudo ifconfig wlan0 down
```

#### 永久关闭（推荐）
编辑 `/boot/config.txt`:
```bash
sudo nano /boot/config.txt
```

在文件末尾添加：
```
# 关闭WiFi
dtoverlay=disable-wifi
```

或者，禁用驱动加载：
```bash
sudo nano /etc/modprobe.d/raspi-blacklist.conf
```

添加：
```
blacklist brcmfmac
blacklist brcmutil
```

保存后重启：
```bash
sudo reboot
```

#### 验证WiFi已关闭
```bash
ifconfig wlan0
# 应显示: wlan0: error fetching interface information: Device not found
```

---

### 2.2 关闭蓝牙

#### 方法1: 通过 config.txt（推荐）
编辑配置文件：
```bash
sudo nano /boot/config.txt
```

添加：
```
# 关闭蓝牙
dtoverlay=disable-bt
```

#### 方法2: 禁用蓝牙服务
```bash
sudo systemctl disable bluetooth.service
sudo systemctl disable hciuart.service
```

#### 方法3: 完全禁用蓝牙驱动
```bash
sudo nano /etc/modprobe.d/raspi-blacklist.conf
```

添加：
```
blacklist btbcm
blacklist hci_uart
```

保存后重启：
```bash
sudo reboot
```

#### 验证蓝牙已关闭
```bash
systemctl status bluetooth.service
# 应显示: inactive (dead)

hciconfig
# 应显示: Can't get device info
```

---

### 2.3 关闭HDMI输出

#### 临时关闭（立即生效，重启后恢复）
```bash
sudo /usr/bin/tvservice -o
```

#### 永久关闭（开机自动关闭）
编辑 `/etc/rc.local`:
```bash
sudo nano /etc/rc.local
```

在 `exit 0` 之前添加：
```bash
# 关闭HDMI以节省功耗
/usr/bin/tvservice -o
```

#### 重新启用HDMI（如需临时使用显示器）
```bash
sudo /usr/bin/tvservice -p
sudo fbset -depth 8
sudo fbset -depth 16
```

#### 验证HDMI状态
```bash
/usr/bin/tvservice -s
# 关闭状态应显示: state 0x120002 [TV is off]
```

---

### 2.4 降低GPU内存

编辑 `/boot/config.txt`:
```bash
sudo nano /boot/config.txt
```

修改或添加：
```
# 降低GPU内存（单位: MB）
gpu_mem=16
```

保存后重启：
```bash
sudo reboot
```

#### 验证GPU内存配置
```bash
vcgencmd get_mem gpu
# 应显示: gpu=16M
```

---

### 2.5 关闭不必要的系统服务

#### 查看所有运行服务
```bash
systemctl list-units --type=service --state=running
```

#### 关闭推荐服务列表

**关闭Avahi（mDNS服务）**:
```bash
sudo systemctl disable avahi-daemon.service
sudo systemctl stop avahi-daemon.service
```

**关闭Triggerhappy（热键守护进程）**:
```bash
sudo systemctl disable triggerhappy.service
sudo systemctl stop triggerhappy.service
```

**关闭CUPS（打印服务）**（如已安装）:
```bash
sudo systemctl disable cups.service
sudo systemctl stop cups.service
```

**关闭ModemManager**（如已安装）:
```bash
sudo systemctl disable ModemManager.service
sudo systemctl stop ModemManager.service
```

**关闭音频服务**（PulseAudio）:
```bash
sudo systemctl --user disable pulseaudio.service
sudo systemctl --user stop pulseaudio.service
```

#### 批量关闭脚本
创建一个脚本文件：
```bash
sudo nano /home/pi/disable_services.sh
```

内容：
```bash
#!/bin/bash
# 批量关闭不必要的服务

services=(
    "avahi-daemon.service"
    "triggerhappy.service"
    "bluetooth.service"
    "hciuart.service"
)

for service in "${services[@]}"; do
    if systemctl is-enabled "$service" &>/dev/null; then
        echo "正在禁用 $service..."
        sudo systemctl disable "$service"
        sudo systemctl stop "$service"
    else
        echo "$service 不存在或已禁用"
    fi
done

echo "服务优化完成！"
```

添加执行权限并运行：
```bash
chmod +x /home/pi/disable_services.sh
sudo /home/pi/disable_services.sh
```

---

### 2.6 关闭LED指示灯

编辑 `/boot/config.txt`:
```bash
sudo nano /boot/config.txt
```

添加：
```
# 关闭ACT LED（绿灯，SD卡活动指示灯）
dtparam=act_led_trigger=none
dtparam=act_led_activelow=off

# 关闭PWR LED（红灯，电源指示灯）
dtparam=pwr_led_trigger=none
dtparam=pwr_led_activelow=off
```

保存后重启：
```bash
sudo reboot
```

---

### 2.7 CPU性能调节（降频省电）

#### 查看当前CPU频率策略
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# 默认通常为: ondemand
```

#### 设置为省电模式
```bash
echo "powersave" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

#### 永久设置省电模式
安装 `cpufrequtils`:
```bash
sudo apt-get install cpufrequtils
```

编辑配置文件：
```bash
sudo nano /etc/default/cpufrequtils
```

添加：
```
GOVERNOR="powersave"
```

保存后重启生效。

#### 验证CPU频率
```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# 应显示较低频率，如: 600000 (600MHz)
```

---

### 2.8 禁用不必要的内核模块

编辑模块黑名单：
```bash
sudo nano /etc/modprobe.d/raspi-blacklist.conf
```

添加（根据实际需求选择）：
```
# WiFi和蓝牙
blacklist brcmfmac
blacklist brcmutil
blacklist btbcm
blacklist hci_uart

# 声卡（如果不需要音频）
blacklist snd_bcm2835

# 摄像头（如果不需要）
blacklist bcm2835_v4l2
```

保存后重启：
```bash
sudo reboot
```

---

## 三、完整优化脚本

### 3.1 一键优化脚本

创建优化脚本：
```bash
sudo nano /home/pi/optimize_pi3b.sh
```

脚本内容：
```bash
#!/bin/bash
# Raspberry Pi 3B 低功耗优化脚本
# 用于数据采集项目

echo "=========================================="
echo "Raspberry Pi 3B 低功耗优化"
echo "=========================================="

# 1. 备份配置文件
echo "备份 /boot/config.txt..."
sudo cp /boot/config.txt /boot/config.txt.backup

# 2. 关闭WiFi和蓝牙
echo "配置关闭WiFi和蓝牙..."
if ! grep -q "dtoverlay=disable-wifi" /boot/config.txt; then
    echo "dtoverlay=disable-wifi" | sudo tee -a /boot/config.txt
fi

if ! grep -q "dtoverlay=disable-bt" /boot/config.txt; then
    echo "dtoverlay=disable-bt" | sudo tee -a /boot/config.txt
fi

# 3. 降低GPU内存
echo "降低GPU内存至16MB..."
if grep -q "^gpu_mem=" /boot/config.txt; then
    sudo sed -i 's/^gpu_mem=.*/gpu_mem=16/' /boot/config.txt
else
    echo "gpu_mem=16" | sudo tee -a /boot/config.txt
fi

# 4. 配置关闭LED
echo "配置关闭LED指示灯..."
if ! grep -q "act_led_trigger=none" /boot/config.txt; then
    echo "dtparam=act_led_trigger=none" | sudo tee -a /boot/config.txt
    echo "dtparam=act_led_activelow=off" | sudo tee -a /boot/config.txt
fi

# 5. 关闭HDMI（通过rc.local）
echo "配置开机自动关闭HDMI..."
if ! grep -q "tvservice -o" /etc/rc.local; then
    sudo sed -i '/^exit 0/i /usr/bin/tvservice -o' /etc/rc.local
fi

# 6. 禁用不必要的服务
echo "禁用不必要的系统服务..."
sudo systemctl disable avahi-daemon.service 2>/dev/null
sudo systemctl disable triggerhappy.service 2>/dev/null
sudo systemctl disable bluetooth.service 2>/dev/null
sudo systemctl disable hciuart.service 2>/dev/null

# 7. 安装并配置CPU省电模式
echo "配置CPU省电模式..."
sudo apt-get install -y cpufrequtils
echo 'GOVERNOR="powersave"' | sudo tee /etc/default/cpufrequtils

# 8. 创建模块黑名单
echo "创建内核模块黑名单..."
cat << 'EOF' | sudo tee /etc/modprobe.d/raspi-blacklist.conf
# WiFi和蓝牙
blacklist brcmfmac
blacklist brcmutil
blacklist btbcm
blacklist hci_uart

# 声卡
blacklist snd_bcm2835
EOF

echo "=========================================="
echo "优化配置完成！"
echo "请重启系统使配置生效: sudo reboot"
echo "=========================================="
```

添加执行权限：
```bash
chmod +x /home/pi/optimize_pi3b.sh
```

运行脚本：
```bash
sudo /home/pi/optimize_pi3b.sh
```

重启系统：
```bash
sudo reboot
```

---

### 3.2 恢复脚本（如需撤销优化）

创建恢复脚本：
```bash
sudo nano /home/pi/restore_pi3b.sh
```

脚本内容：
```bash
#!/bin/bash
# Raspberry Pi 3B 恢复默认配置脚本

echo "=========================================="
echo "恢复Raspberry Pi 3B默认配置"
echo "=========================================="

# 恢复备份的config.txt
if [ -f /boot/config.txt.backup ]; then
    echo "恢复 /boot/config.txt..."
    sudo cp /boot/config.txt.backup /boot/config.txt
else
    echo "未找到备份文件，手动恢复..."
    sudo sed -i '/dtoverlay=disable-wifi/d' /boot/config.txt
    sudo sed -i '/dtoverlay=disable-bt/d' /boot/config.txt
    sudo sed -i '/act_led_trigger/d' /boot/config.txt
    sudo sed -i '/act_led_activelow/d' /boot/config.txt
fi

# 重新启用服务
echo "重新启用系统服务..."
sudo systemctl enable avahi-daemon.service 2>/dev/null
sudo systemctl enable triggerhappy.service 2>/dev/null
sudo systemctl enable bluetooth.service 2>/dev/null
sudo systemctl enable hciuart.service 2>/dev/null

# 删除模块黑名单
echo "删除模块黑名单..."
sudo rm -f /etc/modprobe.d/raspi-blacklist.conf

# 恢复rc.local
echo "恢复rc.local..."
sudo sed -i '/tvservice -o/d' /etc/rc.local

echo "=========================================="
echo "恢复完成！请重启系统: sudo reboot"
echo "=========================================="
```

添加执行权限：
```bash
chmod +x /home/pi/restore_pi3b.sh
```

---

## 四、功耗监控

### 4.1 监控CPU温度
```bash
vcgencmd measure_temp
```

### 4.2 监控CPU频率
```bash
watch -n 1 cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

### 4.3 监控电压（需要外部工具）
```bash
vcgencmd measure_volts
```

### 4.4 监控系统负载
```bash
top
# 或
htop
```

---

## 五、优化效果预估

### 5.1 功耗对比

| 配置项 | 默认状态 | 优化后 | 节省功耗 |
|-------|---------|--------|---------|
| WiFi | 开启 | 关闭 | ~250mA |
| 蓝牙 | 开启 | 关闭 | ~40mA |
| HDMI | 开启 | 关闭 | ~25mA |
| LED | 开启 | 关闭 | ~10mA |
| GPU内存 | 64MB | 16MB | ~15mA |
| CPU模式 | ondemand | powersave | ~100mA（平均）|
| 不必要服务 | 运行中 | 已禁用 | ~30mA |
| **总计** | - | - | **~470mA** |

### 5.2 续航时间预估

假设使用5V 2.5A电源适配器（12.5W）：
- **优化前**: 功耗约700-800mA（3.5-4W）
- **优化后**: 功耗约300-400mA（1.5-2W）
- **节能效果**: 约50%功耗降低

### 5.3 发热量预估

功耗降低直接导致发热量降低，优化后：
- CPU温度降低约5-10°C
- 在10°C环境中运行更稳定
- 散热压力显著减小

---

## 六、注意事项与建议

### 6.1 关键注意事项

1. **网络连接**: 如果关闭WiFi，确保有有线网络连接（以太网）用于SSH访问
2. **测试验证**: 关闭功能前，先在测试环境验证对采集任务无影响
3. **备份配置**: 修改配置文件前务必备份
4. **循序渐进**: 建议逐项关闭功能，每次重启后验证系统正常运行

### 6.2 推荐配置组合

#### 最小化功耗配置（适合7天连续采集）
- ✅ 关闭WiFi
- ✅ 关闭蓝牙
- ✅ 关闭HDMI
- ✅ 降低GPU内存至16MB
- ✅ 关闭LED
- ✅ 禁用不必要服务
- ✅ CPU省电模式

#### 便于调试配置（开发测试阶段）
- ⚠️ 保持WiFi开启（或使用有线网络）
- ✅ 关闭蓝牙
- ⚠️ 保持HDMI开启（便于连接显示器调试）
- ✅ 降低GPU内存
- ⚠️ 保持LED开启（便于观察状态）
- ✅ 禁用部分服务
- ⚠️ CPU保持ondemand模式

### 6.3 额外散热建议

1. **被动散热**:
   - 安装铝制散热片（CPU、内存芯片）
   - 使用导热硅脂增强散热效果
   
2. **主动散热**（可选）:
   - 安装5V小风扇（注意：风扇本身也耗电~100mA）
   - 使用温控风扇（温度>50°C时启动）

3. **环境散热**:
   - 确保Pi3B周围有良好通风
   - 避免密闭容器，使用通风外壳
   - 10°C环境温度已经有利于散热

---

## 七、验证清单

优化完成后，使用以下清单验证：

### 7.1 功能验证
```bash
# 1. WiFi已关闭
ifconfig wlan0
# 预期: Device not found

# 2. 蓝牙已关闭
systemctl status bluetooth
# 预期: inactive (dead)

# 3. HDMI已关闭
/usr/bin/tvservice -s
# 预期: TV is off

# 4. GPU内存已降低
vcgencmd get_mem gpu
# 预期: gpu=16M

# 5. 不必要服务已禁用
systemctl is-enabled avahi-daemon
# 预期: disabled

# 6. CPU省电模式已启用
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# 预期: powersave

# 7. 温度监控
vcgencmd measure_temp
# 预期: 在10°C环境中，CPU温度应在20-35°C范围内
```

### 7.2 采集功能验证
```bash
# 测试数据采集脚本是否正常运行
cd /home/pi/dev/ads1115_project/Themoelectric
source /home/pi/dev/ads1115_project/py311/bin/activate
python3 Full_collector.py
# 观察几个采集周期，确认数据正常写入data.csv
```

### 7.3 网络访问验证
```bash
# 插上网线，确认可以SSH连接
# 如果关闭了WiFi，确保通过有线网络仍能访问Pi3B
```

---

## 八、问题排查

### 8.1 WiFi/蓝牙关闭后无法重新启用
**原因**: 驱动被加入黑名单  
**解决**: 删除 `/etc/modprobe.d/raspi-blacklist.conf` 中的相关条目，重启

### 8.2 关闭HDMI后无法显示
**原因**: HDMI输出已关闭  
**解决**: 
```bash
sudo /usr/bin/tvservice -p
sudo fbset -depth 8
sudo fbset -depth 16
```

### 8.3 CPU过热（>60°C）
**原因**: 散热不足  
**解决**: 
1. 检查散热片是否安装正确
2. 改善通风环境
3. 暂时恢复CPU ondemand模式

### 8.4 数据采集速度变慢
**原因**: CPU频率过低  
**解决**: 
```bash
echo "ondemand" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

---

## 九、总结

通过以上优化配置，Raspberry Pi 3B的功耗可降低约50%，发热量显著减少，非常适合在10°C环境中进行长达7天的连续数据采集任务。

### 快速操作指南

**初次配置时**:
1. 下载并运行优化脚本: `sudo /home/pi/optimize_pi3b.sh`
2. 重启系统: `sudo reboot`
3. 验证优化效果（参见第七节）
4. 启动数据采集任务

**日常监控**:
```bash
# 查看温度
vcgencmd measure_temp

# 查看CPU频率
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# 查看系统负载
top
```

**如需恢复默认配置**:
```bash
sudo /home/pi/restore_pi3b.sh
sudo reboot
```

---

**更新日期**: 2024-11
**适用版本**: Raspberry Pi 3B / Raspberry Pi OS (Bullseye/Bookworm)

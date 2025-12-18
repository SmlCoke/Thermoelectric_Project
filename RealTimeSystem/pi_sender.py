"""
边缘端(Raspberry Pi 5)数据发送模块

该模块负责：
1. 从采集脚本获取电压数据
2. 通过 HTTP POST 将数据发送到主机端
3. 网络异常时自动重试
4. 可选的本地 CSV 备份

使用方式：
    python pi_sender.py --host 192.168.1.100 --port 5000
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import csv

# 第三方库
try:
    import requests
except ImportError:
    print("请安装 requests 库: pip install requests")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pi_sender.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class VoltageData:
    """电压数据结构"""
    timestamp: str
    values: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())


class DataSender:
    """
    数据发送器类
    
    负责将采集到的电压数据通过HTTP POST发送到主机端
    """
    
    # 通道名称
    CHANNEL_NAMES = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    
    def __init__(
        self, 
        host: str, 
        port: int = 5000, 
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 10.0,
        backup_csv: bool = True,
        backup_dir: str = "./backup"
    ):
        """
        初始化数据发送器
        
        参数:
            host: str, 主机端IP地址
            port: int, 主机端端口号
            max_retries: int, 最大重试次数
            retry_delay: float, 重试间隔(秒)
            timeout: float, HTTP请求超时时间(秒)
            backup_csv: bool, 是否启用本地CSV备份
            backup_dir: str, 备份文件目录
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.data_url = f"{self.base_url}/data"
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        self.backup_csv = backup_csv
        self.backup_dir = Path(backup_dir)
        
        # 创建备份目录
        if self.backup_csv:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self._init_backup_file()
        
        # 统计信息
        self.send_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        logger.info(f"数据发送器初始化完成")
        logger.info(f"  目标地址: {self.data_url}")
        logger.info(f"  重试次数: {self.max_retries}")
        logger.info(f"  CSV备份: {'启用' if self.backup_csv else '禁用'}")
    
    def _init_backup_file(self):
        """初始化备份CSV文件"""
        date_str = datetime.now().strftime("%Y%m%d")
        self.backup_file = self.backup_dir / f"voltage_backup_{date_str}.csv"
        
        # 如果文件不存在，写入表头
        if not self.backup_file.exists():
            with open(self.backup_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["Timestamp"] + self.CHANNEL_NAMES
                writer.writerow(header)
            logger.info(f"创建备份文件: {self.backup_file}")
    
    def _backup_data(self, data: VoltageData):
        """
        将数据备份到本地CSV文件
        
        参数:
            data: VoltageData, 要备份的数据
        """
        try:
            # 检查是否需要创建新的日期文件
            current_date = datetime.now().strftime("%Y%m%d")
            if current_date not in str(self.backup_file):
                self._init_backup_file()
            
            with open(self.backup_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [data.timestamp] + data.values
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"备份数据失败: {e}")
    
    def send_data(self, values: List[float], timestamp: Optional[str] = None) -> bool:
        """
        发送电压数据到主机端
        
        参数:
            values: List[float], 8个通道的电压值
            timestamp: str, 时间戳 (可选，默认使用当前时间)
        
        返回:
            bool, 发送是否成功
        """
        # 验证数据
        if len(values) != 8:
            logger.error(f"数据长度错误: 期望8个通道，实际{len(values)}个")
            return False
        
        # 创建数据对象
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        data = VoltageData(timestamp=timestamp, values=values)
        
        # 本地备份
        if self.backup_csv:
            self._backup_data(data)
        
        # 发送数据
        self.send_count += 1
        success = self._send_with_retry(data)
        
        if success:
            self.success_count += 1
            logger.info(f"数据发送成功 [#{self.send_count}]: {timestamp}")
        else:
            self.failure_count += 1
            logger.error(f"数据发送失败 [#{self.send_count}]: {timestamp}")
        
        return success
    
    def _send_with_retry(self, data: VoltageData) -> bool:
        """
        带重试机制的数据发送
        
        参数:
            data: VoltageData, 要发送的数据
        
        返回:
            bool, 发送是否成功
        """
        payload = data.to_dict()
        headers = {'Content-Type': 'application/json'}
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.data_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return True
                else:
                    logger.warning(f"服务器返回错误 (尝试 {attempt}/{self.max_retries}): "
                                 f"状态码 {response.status_code}")
                    
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"连接错误 (尝试 {attempt}/{self.max_retries}): {e}")
            except requests.exceptions.Timeout as e:
                logger.warning(f"请求超时 (尝试 {attempt}/{self.max_retries}): {e}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常 (尝试 {attempt}/{self.max_retries}): {e}")
            
            # 重试前等待
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        
        return False
    
    def check_connection(self) -> bool:
        """
        检查与主机端的连接状态
        
        返回:
            bool, 连接是否正常
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"连接检查失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取发送统计信息
        
        返回:
            Dict, 统计信息
        """
        return {
            "total_sent": self.send_count,
            "success": self.success_count,
            "failure": self.failure_count,
            "success_rate": self.success_count / max(self.send_count, 1) * 100
        }


class MockDataCollector:
    """
    模拟数据采集器 (用于测试)
    
    在实际部署时，应该从 Full_collector.py 获取真实数据
    """
    
    def __init__(self, base_values: Optional[List[float]] = None):
        """
        初始化模拟采集器
        
        参数:
            base_values: List[float], 基准电压值
        """
        import random
        self.random = random
        
        if base_values is None:
            # 默认的基准电压值
            self.base_values = [0.5, 0.6, 0.55, 0.52, 0.58, 0.54, 0.53, 0.57]
        else:
            self.base_values = base_values
    
    def collect(self) -> List[float]:
        """
        采集一次电压数据 (模拟)
        
        返回:
            List[float], 8个通道的电压值
        """
        # 添加随机噪声
        values = [
            v + self.random.gauss(0, 0.02) 
            for v in self.base_values
        ]
        return values


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Raspberry Pi 数据发送器')
    
    parser.add_argument('--host', type=str, required=True,
                       help='主机端IP地址')
    parser.add_argument('--port', type=int, default=5000,
                       help='主机端端口号 (默认: 5000)')
    parser.add_argument('--interval', type=float, default=10.0,
                       help='采集间隔(秒) (默认: 10)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='最大重试次数 (默认: 3)')
    parser.add_argument('--no-backup', action='store_true',
                       help='禁用本地CSV备份')
    parser.add_argument('--backup-dir', type=str, default='./backup',
                       help='备份文件目录 (默认: ./backup)')
    parser.add_argument('--test', action='store_true',
                       help='使用模拟数据进行测试')
    
    args = parser.parse_args()
    
    # 创建发送器
    sender = DataSender(
        host=args.host,
        port=args.port,
        max_retries=args.max_retries,
        backup_csv=not args.no_backup,
        backup_dir=args.backup_dir
    )
    
    # 检查连接
    logger.info("检查与主机端的连接...")
    if sender.check_connection():
        logger.info("连接正常!")
    else:
        logger.warning("无法连接到主机端，将继续尝试发送...")
    
    # 创建数据采集器
    if args.test:
        logger.info("使用模拟数据进行测试")
        collector = MockDataCollector()
    else:
        # 在实际部署时，需要替换这个分支的代码，连接到真实的 Full_collector.py
        # 当前保持使用 MockDataCollector 以便于开发和测试
        # 例如：可以通过读取共享文件、管道或socket与采集脚本通信
        logger.info("注意: 当前使用模拟数据采集器")
        logger.info("在实际部署时，请修改此处代码连接到 Full_collector.py")
        collector = MockDataCollector()
    
    # 主循环
    logger.info(f"开始数据采集和发送 (间隔: {args.interval}秒)")
    logger.info("按 Ctrl+C 停止")
    
    try:
        while True:
            # 采集数据
            values = collector.collect()
            
            # 发送数据
            sender.send_data(values)
            
            # 等待下一次采集
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
        
        # 打印统计信息
        stats = sender.get_statistics()
        logger.info("=" * 40)
        logger.info("发送统计:")
        logger.info(f"  总发送次数: {stats['total_sent']}")
        logger.info(f"  成功: {stats['success']}")
        logger.info(f"  失败: {stats['failure']}")
        logger.info(f"  成功率: {stats['success_rate']:.1f}%")
        logger.info("=" * 40)


if __name__ == '__main__':
    main()

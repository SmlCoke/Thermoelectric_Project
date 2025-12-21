"""
GUI 可视化应用程序

该模块负责：
1. 使用 PyQt5 构建专业的图形用户界面
2. 使用 matplotlib 进行实时数据可视化
3. 显示历史数据、预测结果和对比信息
4. 提供用户交互控件（通道选择、预测步数等）

使用方式：
    python gui_app.py --port 5000 --model-path ../TimeSeries/Prac_train/checkpoints/best_model.pth
"""

import os
import sys
import time
import logging
import argparse
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque

# 第三方库
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QComboBox, QPushButton, QGroupBox, QStatusBar, QFrame,
        QGridLayout, QSplitter, QSizePolicy, QSpacerItem
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QPalette, QColor
except ImportError:
    print("请安装 PyQt5: pip install PyQt5")
    sys.exit(1)

try:
    import matplotlib
    # 只在未设置后端时才设置 Qt5Agg 后端
    if matplotlib.get_backend() == 'agg' or not matplotlib.get_backend():
        matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
except ImportError:
    print("请安装 matplotlib: pip install matplotlib")
    sys.exit(1)

import numpy as np

# 导入自定义模块
from server import DataServer, SlidingWindow
from inference_engine import create_inference_engine, InferenceEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 信号类，用于线程间通信
class SignalEmitter(QObject):
    """信号发射器，用于从后台线程更新 GUI"""
    data_updated = pyqtSignal()
    inference_completed = pyqtSignal(object)
    status_changed = pyqtSignal(str)


class PlotCanvas(FigureCanvas):
    """
    Matplotlib 绑定的 PyQt5 画布
    
    负责绑定 matplotlib 图表和 PyQt5 界面
    """
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        """
        初始化画布
        
        参数:
            parent: QWidget, 父控件
            width: int, 宽度（英寸）
            height: int, 高度（英寸）
            dpi: int, 分辨率
        """
        # 创建 Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('#F0F0F0')
        
        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 设置大小策略
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        
        # 初始化子图
        self.axes = []
        self._init_subplots()
    
    def _init_subplots(self):
        """初始化子图布局"""
        # 清除现有图形
        self.fig.clear()
        self.axes = []
        
        # 创建一个主图（单通道或全通道会动态调整）
        ax = self.fig.add_subplot(111)
        self.axes.append(ax)
        
        self.fig.tight_layout(pad=3.0)
    
    def setup_multi_channel(self, num_channels: int = 8):
        """
        设置多通道布局（4行2列）
        
        参数:
            num_channels: int, 通道数量
        """
        self.fig.clear()
        self.axes = []
        
        rows, cols = 4, 2
        for i in range(num_channels):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self.axes.append(ax)
        
        self.fig.tight_layout(pad=2.0)
        self.draw()
    
    def setup_single_channel(self):
        """设置单通道布局"""
        self.fig.clear()
        self.axes = []
        
        ax = self.fig.add_subplot(111)
        self.axes.append(ax)
        
        self.fig.tight_layout(pad=3.0)
        self.draw()
    
    def clear_all(self):
        """清除所有图表"""
        for ax in self.axes:
            ax.clear()
        self.draw()


class MainWindow(QMainWindow):
    """
    主窗口类
    
    实现完整的 GUI 界面和功能
    """
    
    # 通道名称和颜色
    CHANNEL_NAMES = [
        "Yellow", "Ultraviolet", "Infrared", "Red",
        "Green", "Blue", "Transparent", "Violet"
    ]
    
    CHANNEL_COLORS = [
        '#FFD700', '#9400D3', '#8B0000', '#FF0000',
        '#00FF00', '#0000FF', '#808080', '#EE82EE'
    ]
    
    def __init__(
        self,
        server: DataServer,
        inference_engine,
        parent=None
    ):
        """
        初始化主窗口
        
        参数:
            server: DataServer, 数据服务器实例
            inference_engine: InferenceEngine, 推理引擎实例
            parent: QWidget, 父控件
        """
        super(MainWindow, self).__init__(parent)
        
        self.server = server
        self.engine = inference_engine
        
        # 信号发射器
        self.signals = SignalEmitter()
        self.signals.data_updated.connect(self._on_data_updated)
        self.signals.inference_completed.connect(self._on_inference_completed)
        self.signals.status_changed.connect(self._on_status_changed)
        
        # 当前状态
        self.current_channel = -1  # -1 表示全通道
        self.predict_steps = 10
        self.last_prediction = None
        self.prediction_history = deque(maxlen=10)
        
        # 初始化界面
        self._init_ui()
        
        # 设置推理回调
        self.server.set_inference_callback(self._run_inference)
        
        # 启动定时器更新界面
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(1000)  # 每秒更新一次
    
    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("热电芯片电压实时监测与预测系统")
        self.setMinimumSize(1200, 800)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                font-size: 12px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QPushButton {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        
        # 创建中心控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：控制面板
        left_panel = self._create_control_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：图表区域
        right_panel = self._create_plot_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setSizes([300, 900])
        
        # 创建状态栏
        self._create_status_bar()
    
    def _create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # 系统状态组
        status_group = QGroupBox("系统状态")
        status_layout = QGridLayout()
        
        # 状态标签
        self.status_label = QLabel("等待数据...")
        self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        status_layout.addWidget(QLabel("当前状态:"), 0, 0)
        status_layout.addWidget(self.status_label, 0, 1)
        
        # 窗口大小
        self.window_label = QLabel("0 / 60")
        status_layout.addWidget(QLabel("窗口数据:"), 1, 0)
        status_layout.addWidget(self.window_label, 1, 1)
        
        # 接收计数
        self.receive_label = QLabel("0")
        status_layout.addWidget(QLabel("接收次数:"), 2, 0)
        status_layout.addWidget(self.receive_label, 2, 1)
        
        # 最新时间戳
        self.timestamp_label = QLabel("--")
        status_layout.addWidget(QLabel("最新数据:"), 3, 0)
        status_layout.addWidget(self.timestamp_label, 3, 1)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 通道选择组
        channel_group = QGroupBox("通道选择")
        channel_layout = QVBoxLayout()
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("全部通道", -1)
        for i, name in enumerate(self.CHANNEL_NAMES):
            self.channel_combo.addItem(f"{i+1}. {name}", i)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        
        channel_layout.addWidget(QLabel("选择显示的通道:"))
        channel_layout.addWidget(self.channel_combo)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # 预测设置组
        predict_group = QGroupBox("预测设置")
        predict_layout = QVBoxLayout()
        
        self.steps_combo = QComboBox()
        self.steps_combo.addItem("1 步预测", 1)
        self.steps_combo.addItem("10 步预测", 10)
        self.steps_combo.setCurrentIndex(1)  # 默认10步
        self.steps_combo.currentIndexChanged.connect(self._on_steps_changed)
        
        predict_layout.addWidget(QLabel("预测步数:"))
        predict_layout.addWidget(self.steps_combo)
        
        predict_group.setLayout(predict_layout)
        layout.addWidget(predict_group)
        
        # 最新电压值组
        voltage_group = QGroupBox("当前电压值")
        voltage_layout = QGridLayout()
        
        self.voltage_labels = []
        for i, name in enumerate(self.CHANNEL_NAMES):
            row, col = i // 2, i % 2
            label_name = QLabel(f"{name}:")
            label_name.setStyleSheet(f"color: {self.CHANNEL_COLORS[i]};")
            label_value = QLabel("--")
            label_value.setStyleSheet("font-family: monospace; font-size: 11px;")
            voltage_layout.addWidget(label_name, row, col * 2)
            voltage_layout.addWidget(label_value, row, col * 2 + 1)
            self.voltage_labels.append(label_value)
        
        voltage_group.setLayout(voltage_layout)
        layout.addWidget(voltage_group)
        
        # 操作按钮组
        button_group = QGroupBox("操作")
        button_layout = QVBoxLayout()
        
        # 清除数据按钮
        clear_btn = QPushButton("清除数据")
        clear_btn.clicked.connect(self._on_clear_clicked)
        clear_btn.setStyleSheet("background-color: #f44336;")
        button_layout.addWidget(clear_btn)
        
        # 刷新图表按钮
        refresh_btn = QPushButton("刷新图表")
        refresh_btn.clicked.connect(self._update_plot)
        button_layout.addWidget(refresh_btn)
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        # 添加弹簧
        layout.addStretch()
        
        # 模型信息
        model_group = QGroupBox("模型信息")
        model_layout = QVBoxLayout()
        
        model_info = self.engine.get_model_info()
        model_type = model_info.get('model_type', 'N/A').upper()
        window_size = model_info.get('window_size', 'N/A')
        device = str(model_info.get('device', 'N/A'))
        
        model_layout.addWidget(QLabel(f"模型类型: {model_type}"))
        model_layout.addWidget(QLabel(f"窗口大小: {window_size}"))
        model_layout.addWidget(QLabel(f"设备: {device}"))
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        return panel
    
    def _create_plot_panel(self) -> QWidget:
        """创建图表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建标题
        title_label = QLabel("实时电压数据与预测")
        title_label.setFont(QFont('Arial', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 创建图表画布
        self.canvas = PlotCanvas(self, width=10, height=8, dpi=100)
        layout.addWidget(self.canvas)
        
        # 图例说明
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        
        legend_items = [
            ("历史数据", "#2196F3"),
            ("预测结果", "#FF5722"),
            ("实际测量", "#4CAF50")
        ]
        
        for text, color in legend_items:
            label = QLabel(f"● {text}")
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            legend_layout.addWidget(label)
            legend_layout.addSpacing(20)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        return panel
    
    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 服务器状态
        self.server_status = QLabel("服务器: 运行中")
        self.server_status.setStyleSheet("color: green;")
        self.status_bar.addPermanentWidget(self.server_status)
        
        # 分隔符
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        self.status_bar.addPermanentWidget(separator)
        
        # 端口信息
        port_label = QLabel(f"端口: {self.server.port}")
        self.status_bar.addPermanentWidget(port_label)
    
    def _on_channel_changed(self, index: int):
        """通道选择改变事件"""
        self.current_channel = self.channel_combo.currentData()
        logger.info(f"切换到通道: {self.current_channel}")
        
        # 重新设置图表布局
        if self.current_channel == -1:
            self.canvas.setup_multi_channel(8)
        else:
            self.canvas.setup_single_channel()
        
        self._update_plot()
    
    def _on_steps_changed(self, index: int):
        """预测步数改变事件"""
        self.predict_steps = self.steps_combo.currentData()
        logger.info(f"预测步数改变为: {self.predict_steps}")
    
    def _on_clear_clicked(self):
        """清除数据按钮点击事件"""
        self.server.window.clear()
        self.last_prediction = None
        self.prediction_history.clear()
        self.canvas.clear_all()
        self._update_status_labels()
        logger.info("数据已清除")
    
    def _on_data_updated(self):
        """数据更新信号处理"""
        self._update_status_labels()
        # 不在这里更新图表，避免频繁更新导致阻塞
        # 图表会通过定时器或推理完成后更新
    
    def _on_inference_completed(self, result):
        """推理完成信号处理"""
        self.last_prediction = result
        self.prediction_history.append(result)
        self._update_status("推理完成")
        # 使用 QTimer 确保图表更新不阻塞信号处理
        # 即使图表更新很快，延迟调用也能确保事件循环正常运行
        QTimer.singleShot(0, self._update_plot)
    
    def _on_status_changed(self, status: str):
        """状态改变信号处理"""
        self._update_status(status)
    
    def _update_status(self, status: str):
        """更新状态显示"""
        self.status_label.setText(status)
        
        # 根据状态设置颜色
        if "完成" in status:
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif "等待" in status:
            self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        elif "推理中" in status:
            self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #666666; font-weight: bold;")
    
    def _update_status_labels(self):
        """更新状态标签"""
        window = self.server.window
        
        # 窗口大小
        current_size = window.get_current_size()
        self.window_label.setText(f"{current_size} / {window.window_size}")
        
        # 接收计数
        self.receive_label.setText(str(self.server.receive_count))
        
        # 最新时间戳
        timestamp = window.get_latest_timestamp()
        if timestamp:
            self.timestamp_label.setText(timestamp)
        
        # 更新电压值
        values = window.get_latest_values()
        if values:
            for i, (label, value) in enumerate(zip(self.voltage_labels, values)):
                label.setText(f"{value:.4f} V")
        
        # 更新状态
        if window.is_ready():
            self._update_status("数据就绪")
        else:
            self._update_status(f"等待数据... ({current_size}/{window.window_size})")
    
    def _update_display(self):
        """定时器更新显示"""
        self._update_status_labels()
    
    def _run_inference(self):
        """执行推理（在后台线程中调用）"""
        try:
            self.signals.status_changed.emit("推理中...")
            
            # 获取窗口数据
            data = self.server.window.get_data()
            if data is None:
                return
            
            # 获取时间戳
            timestamp = self.server.window.get_latest_timestamp() or ""
            
            # 执行推理
            result = self.engine.predict(data, steps=self.predict_steps, timestamp=timestamp)
            
            # 保存预测结果
            self.server.window.save_prediction(result.predictions, timestamp)
            
            # 发送信号
            self.signals.inference_completed.emit(result)
            
            logger.info(f"推理完成: {self.predict_steps} 步预测")
            
        except Exception as e:
            logger.error(f"推理出错: {e}")
            self.signals.status_changed.emit(f"推理出错: {e}")
    
    def _update_plot(self):
        """更新图表"""
        try:
            window = self.server.window
            data = window.get_data()
            
            if self.current_channel == -1:
                # 全通道显示
                self._plot_all_channels(data)
            else:
                # 单通道显示
                self._plot_single_channel(data, self.current_channel)
            
            # 使用 draw_idle 替代 draw，避免阻塞主线程
            self.canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"更新图表出错: {e}")
    
    def _plot_all_channels(self, data: Optional[np.ndarray]):
        """
        绘制所有通道
        
        参数:
            data: np.ndarray or None, 窗口数据
        """
        # 确保有8个子图
        if len(self.canvas.axes) != 8:
            self.canvas.setup_multi_channel(8)
        
        for i, ax in enumerate(self.canvas.axes):
            ax.clear()
            
            if data is not None and len(data) > 0:
                # 绘制历史数据
                time_axis = np.arange(len(data))
                ax.plot(time_axis, data[:, i], 
                       color=self.CHANNEL_COLORS[i], 
                       linewidth=1.5, 
                       label='历史数据')
                
                # 绘制预测
                if self.last_prediction is not None:
                    pred_data = self.last_prediction.predictions
                    pred_time = np.arange(len(data), len(data) + len(pred_data))
                    ax.plot(pred_time, pred_data[:, i],
                           color='#FF5722',
                           linewidth=2,
                           linestyle='--',
                           marker='o',
                           markersize=3,
                           label='预测')
            
            # 设置标题和标签
            ax.set_title(f'{self.CHANNEL_NAMES[i]}', fontsize=10, fontweight='bold')
            ax.set_xlabel('时间步', fontsize=8)
            ax.set_ylabel('电压 (V)', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)
            
            if data is not None and len(data) > 0:
                ax.legend(loc='upper left', fontsize=7)
        
        # 不再调用 tight_layout，避免阻塞主线程
        # tight_layout 仅在初始化时（setup_multi_channel/setup_single_channel）调用一次
    
    def _plot_single_channel(self, data: Optional[np.ndarray], channel: int):
        """
        绘制单个通道
        
        参数:
            data: np.ndarray or None, 窗口数据
            channel: int, 通道索引
        """
        # 确保只有1个子图
        if len(self.canvas.axes) != 1:
            self.canvas.setup_single_channel()
        
        ax = self.canvas.axes[0]
        ax.clear()
        
        if data is not None and len(data) > 0:
            # 绘制历史数据
            time_axis = np.arange(len(data))
            ax.plot(time_axis, data[:, channel],
                   color=self.CHANNEL_COLORS[channel],
                   linewidth=2,
                   label='历史数据')
            
            # 绘制预测
            if self.last_prediction is not None:
                pred_data = self.last_prediction.predictions
                pred_time = np.arange(len(data), len(data) + len(pred_data))
                ax.plot(pred_time, pred_data[:, channel],
                       color='#FF5722',
                       linewidth=2.5,
                       linestyle='--',
                       marker='o',
                       markersize=5,
                       label=f'预测 ({self.predict_steps}步)')
                
                # 连接历史数据和预测
                if len(time_axis) > 0 and len(pred_time) > 0:
                    ax.plot([time_axis[-1], pred_time[0]],
                           [data[-1, channel], pred_data[0, channel]],
                           color='#FF5722',
                           linewidth=1,
                           linestyle=':',
                           alpha=0.5)
        
        # 设置标题和标签
        ax.set_title(f'{self.CHANNEL_NAMES[channel]} 通道电压', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步 (10秒/步)', fontsize=12)
        ax.set_ylabel('电压 (V)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        # 不再调用 tight_layout，避免阻塞主线程
        # tight_layout 仅在初始化时（setup_multi_channel/setup_single_channel）调用一次
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        logger.info("正在关闭应用程序...")
        self.update_timer.stop()
        event.accept()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='热电芯片电压实时监测与预测系统')
    
    parser.add_argument('--port', type=int, default=5000,
                       help='HTTP 服务端口 (默认: 5000)')
    parser.add_argument('--window-size', type=int, default=60,
                       help='滑动窗口大小 (默认: 60)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    logger.info("=" * 60)
    logger.info("热电芯片电压实时监测与预测系统")
    logger.info("=" * 60)
    
    engine = create_inference_engine(args.model_path)
    
    # 创建数据服务器
    server = DataServer(
        port=args.port,
        window_size=args.window_size
    )
    
    # 在后台线程启动服务器
    server.run_in_thread()
    
    # 创建 Qt 应用
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = MainWindow(server, engine)
    window.show()
    
    logger.info("=" * 60)
    logger.info("GUI 已启动")
    logger.info(f"HTTP 服务: http://0.0.0.0:{args.port}")
    logger.info("等待 Raspberry Pi 发送数据...")
    logger.info("=" * 60)
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

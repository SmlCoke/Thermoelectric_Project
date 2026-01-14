# Thermoelectric_Project

热电芯片天空辐射采集、预测与实时展示的完整项目仓库。项目已结题，完整内容与实验结果可参考根目录下的结题文档：

- [`第二组-设计报告.pdf`](./第二组-设计报告.pdf)
- [`第二组-总结报告.pdf`](./第二组-总结报告.pdf)

## 仓库目录速览（仅需关注 `.md` 与 `.py`）

- **Code_1/**：早期的时间序列模拟代码（数据生成与训练推理均为模拟环境，已与最终版本 `TimeSeries/` 不一致）
- **DataCollectCode/**：树莓派 3B 与 PC 的通信与采集控制脚本，包含真实采集与模拟采集、数据转发等代码，文档见 `DataCollectCode/docs/`
- **FigureProcess/**：天空图像的处理与分析脚本
- **Initial/**：最初方案书，内容与最终实现差异较大，可忽略
- **RealTimeSystem/**：实时采集 + 预测 + GUI 展示的控制与通信脚本，完整说明见其中的 `README.md`
- **TimeSeries/**：核心时间序列预测模型代码（数据处理、训练、推理与文档）

## 推荐阅读路径

1. **整体方案与成果**：查看根目录的两份 PDF 结题报告，快速了解项目背景、实验流程与结果。
2. **数据采集**：阅读 `DataCollectCode/docs/readme.md` 及同目录下的接线、自动化、低功耗等文档，使用 `Full_collector.py`/`single_collector.py` 或 `mock_collector.py` 进行真实或模拟采集，并可通过 `pi_sender.py` 将数据转发到主机。
3. **实时系统**：参见 `RealTimeSystem/README.md`，按说明在主机端启动 `server.py` / `gui_app.py`，在树莓派端运行采集与转发脚本，实现“采集→推理→可视化”的闭环。
4. **模型训练与推理**：`TimeSeries/README.md` 提供基于 LSTM/GRU 的训练与预测流程，更多细节见 `TimeSeries/docs/`。
5. **图像处理**：若需查看天空图像分析流程，可参考 `FigureProcess/` 下的脚本。

## 快速指引

- **采集与转发**：`DataCollectCode/Full_collector.py`（真实硬件）或 `mock_collector.py`（模拟），可搭配 `pi_sender.py` 将数据通过 HTTP POST 发往主机。
- **实时展示**：在主机端运行 `RealTimeSystem/gui_app.py`（可指定模型路径），并保持树莓派端采集/转发脚本运行。
- **模型训练/推理**：进入 `TimeSeries/src/` 运行 `train.py` / `predict.py`（依赖见 `TimeSeries/requirements.txt`）。

## 说明

- 当前仓库目录与早期版本相比已有较大调整，旧的 `DataCollection` 路径和说明已废弃，请以本 README 及各子目录文档为准。
- 贡献或二次开发时，仅需关注各目录下的 `.md` 文档和 `.py` 源码；其它文件为数据、日志或历史资料。 

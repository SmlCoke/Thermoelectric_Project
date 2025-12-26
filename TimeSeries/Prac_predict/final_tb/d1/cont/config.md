## 方法
我们研究off1中从第77个数据点开始的时间序列预测，
降采样数据集对应的输入序列时间点为：77，78，79，...，136，共60个时间点，
预测序列时间点为：137，138，...，146，共10个时间点

原始数据集的位序y与降采样数据集的位序x之间的关系为：$y = 6x - 4$
我们假设这里的$y$和$x$都从1开始编号。但是在程序中，idx通常从0开始编号，因此在程序实现中有所不同。
我们将其映射为原始数据集，输入序列时间点分别为：
458，464，470，...，812，共60个时间点，
预测序列时间点为：818，824，830，836，842，848，854，860，866，872，共10个时间点。

我们需要用未使用降采样数据集的模型，来预测这10个时间点的数据。
由于模型配置为输入60个时间点，输出10个时间点，因此我们需要分别在以下10个时间点执行单步预测：
758，764，770，776，782，788，794，800，806，812，共10个时间点。


## 程序
在程序中，我们获取这一时间段的预测数据和图像的方法为：
- 对于在降采样数据集上训练的模型，在执行 `predict.py` 时，指定参数 `--start_idx` 为76，这里的指定为76是因为程序中idx从0开始编号。
  ```bash
  
- 对于在未降采样数据集上训练的模型，在执行 `predict.py` 时，需要单独使用 `custom` 功能。同时，注意脚本内部的预测目标输入起始点序列为：758-1, 764-1, 770-1, 776-1, 782-1, 788-1, 794-1, 800-1, 806-1, 812-1。
  ```bash
  python predict.py --model_path D:\Courses\Thermoelectric_Project\TimeSeries\Prac_train\final_tb\raw\checkpoints\best_model.pth --csv_path D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\TrainSet\final_tb\tests\raw\data1210_clean.csv --save_path D:\Courses\Thermoelectric_Project\TimeSeries\Prac_predict\final_tb\raw\cont\data_Infrared.npy --custom --channel Infrared --plot_path D:\Courses\Thermoelectric_Project\TimeSeries\Prac_predict\final_tb\raw\cont\data_Infrared.svg
  ```


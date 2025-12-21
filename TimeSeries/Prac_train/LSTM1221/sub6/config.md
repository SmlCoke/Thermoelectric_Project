## 训练配置
在`sub6`数据集下训练

```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir D:\Courses\Thermoelectric_Project\TimeSeries\Prac_data\TrainSet\dataset\sub6 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir D:\Courses\Thermoelectric_Project\TimeSeries\Prac_train\LSTM1221\checkpoints --log_dir  D:\Courses\Thermoelectric_Project\TimeSeries\Prac_train\LSTM1221\logs 
```


## 预测
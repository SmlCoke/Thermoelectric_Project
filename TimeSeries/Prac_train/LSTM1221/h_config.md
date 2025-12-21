## 步骤
请严格按照如下步骤执行指令
### 1. 克隆仓库
```bash
git clone git@github.com:SmlCoke/Thermoelectric_Project.git
cd ./Thermoelectric_Project/TimeSeries/src
```

### 2. 训练
**先训练任务2.1，结束之后将 `TimeSeries/Prac_train/LSTM1221/d1_sub8` 文件夹打包发给我。后面两个任务在本周训练结束发给我即可。**
#### 2.1 d1_sub8
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/dataset/d1_sub8 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/LSTM1221/d1_sub8/checkpoints --log_dir  ../Prac_train/LSTM1221/d1_sub8/logs 
```
训练结束之后，在 `TimeSeries/Prac_train/LSTM1221/d1_sub8` 中存放着模型权重文件，将这个文件夹发给我。

#### 2.2 d2_sub8
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/dataset/d2_sub8 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/LSTM1221/d2_sub8/checkpoints --log_dir  ../Prac_train/LSTM1221/d2_sub8/logs 
```
训练结束之后，在 `TimeSeries/Prac_train/LSTM1221/d2_sub8` 中存放着模型权重文件，将这个文件夹发给我。

#### 2.3 sub8_d4
```bash
python train.py --model lstm --hidden_size 256 --num_layers 2 --num_epochs 100 --data_dir ../Prac_data/TrainSet/dataset/sub8_d4 --window_size 60 --predict_steps 10 --batch_size 32 --stride 1 --save_dir ../Prac_train/LSTM1221/sub8_d4/checkpoints --log_dir  ../Prac_train/LSTM1221/sub8_d4/logs 
```
训练结束之后，在 `TimeSeries/Prac_train/LSTM1221/sub8_d4` 中存放着模型权重文件，将这个文件夹发给我。
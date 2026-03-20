### 先决条件

确保你有一台带有NVIDIA GPU和Python 2.7的机器，并且有大约100 GB的磁盘空间。<br>

pytorch==1.1.0 <br>
点击==7.0 <br>
numpy==1.16.5 <br>
tqdm==4.35.0 <br>

### 训练数据下载
你可以使用
```
bash tools/download.sh
```
下载数据<br>
其余数据和训练模型可以从 [百度云](https://pan.baidu.com/s/1oHdwYDSJXC1mlmvu8cQhKw)(密码:3jot) 或者 [谷歌云端硬盘](https://drive.google.com/drive/folders/13e-b76otJukupbjfC-n1s05L202PaFKQ?usp=sharing)
解压feature1.zip和feature2.zip，并将它们合并到data/rcnn_feature/<br>
使用
```
bash 工具/处理.sh
```
处理数据<br>

### 训练
跑
```
CUDA_VISIBLE_DEVICES=1 nohup python train.py > train.out &
```
训练一个模型

### 测试
跑
```
CUDA_VISIBLE_DEVICES=1 nohup python eval.py > test.out &
```
评估模型

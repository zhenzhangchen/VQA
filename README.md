### 先决条件

确保你有一台带有NVIDIA GPU和Python 2.7的机器，并且有大约100 GB的磁盘空间。<br>

pytorch==1.1.0 <br>
点击==7.0 <换行>
numpy==1.16.5 <br>
tqdm==4.35.0 <br>

### 训练数据下载
首先通过tools文件中，使用download.sh下载数据文件，通过process.sh对文件进行处理
你可以
```
bash tools/download.sh
```
下载数据<换行>
其余数据和训练模型可以从 [百度云](https://pan.baidu.com/s/1oHdwYDSJXC1mlmvu8cQhKw)(密码:3jot) 或者 [谷歌云端硬盘](https://drive.google.com/drive/folders/13e-b76otJukupbjfC-n1s05L202PaFKQ?usp=sharing)
解压feature1.zip和feature2.zip，并将它们合并到data/rcnn_feature/<br>
使用
```
bash 工具/处理.sh
```
处理数据<换行>

阅读文献Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering
了解模型结构，并下载论文原代码，理清原论文模型架构并复现原论文。
之后阅读改代码即可。

代码在源代码上进行大幅度修改
mian.py 训练文件
eval.py 测试文件
tools.py 共享层
utils.py 工具类
language_model.py 文本模型类
fusion_modules.py 多模态融合类
classifier.py 分类器
fc.py 全连接层类等
dataset.py 数据集预处理类
coor_main.py 模型架构类
attention.py 注意力机制类
loss_functions.py 损失函数类
...

### 训练
```
CUDA_VISIBLE_DEVICES=1 nohup python train.py > train.out &
```
nohup python train.py  --load_weight='output/batch_128_39.pth' --had_trained_epoch=40 > v3_lr=0.002_add.out &
```

训练一个模型

### 测试

```
CUDA_VISIBLE_DEVICES=1 nohup python eval.py > test.out &
```
nohup python test.py  --load_weight='output/batch_128_39.pth' --had_trained_epoch=40 > v3_lr=0.002_add.out &
```
评估模型

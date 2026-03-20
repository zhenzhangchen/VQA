### 先决条件

确保你有一台带有NVIDIA GPU和Python 2.7的机器，并且有大约100 GB的磁盘空间。<br>

配置anaconda环境变量<br>
回显 $PATH<br>
vim ~/.bashrc<br>
导出 PATH=""/path/to/anaconda/bin:$PATH"<br>
 source ~/.bashrc<br>

服务器环境配置步骤：<br>
1.下载anaconda<br>
2. 执行 anaconda.sh<br>
3. 根据项目要求创建anaconda环境<br>
conda create -n env_name python=2.7<br>
4.conda env list查看conda环境是否创建成功<br>
5.conda activate env_name激活conda环境<br>
6. 安装pytorch+cuda<br>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118<br>
7. export PATH="/path/to/anaconda/bin:$PATH"   注：/path/to/anaconda/bin是刚刚anaconda安装路径<br>
8. 源 ~/.bashrc<br>
9. 好的！<br>

pytorch==1.1.0 <br>
点击==7.0
numpy==1.16.5 <br>
tqdm==4.35.0 <br>

### 训练数据下载
首先通过tools文件中，使用download.sh下载数据文件，通过process.sh对文件进行处理<br>

你可以换行

```
bash tools/download.sh
```
下载数据

其余数据和训练模型可以从 [百度云](https://pan.baidu.com/s/1oHdwYDSJXC1mlmvu8cQhKw)(密码:3jot) 或者 [谷歌云端硬盘](https://drive.google.com/drive/folders/13e-b76otJukupbjfC-n1s05L202PaFKQ?usp=sharing)
解压feature1.zip和feature2.zip，并将它们合并到data/rcnn_feature/<br>
使用
```
bash 工具/处理.sh
```
处理数据


阅读文献自底向和自顶向注意力在图像字幕和视觉问答中的应用<br>
了解模型结构，并下载论文原代码，理清原论文模型架构并复现原论文。<br>
之后阅读改代码即可。<换行>

代码在源代码上进行大幅度修改
mian.py 训练文件<br>
eval.py 测试文件<br>
tools.py 共享层<br>
utils.py 工具类

language_model.py 文本模型类<br>
fusion_modules.py 多模态融合类
分类器.py 分类器
fc.py 全连接层类等<br>
dataset.py 数据集预处理类
coor_main.py 模型架构类
attention.py 注意力机制类<br>
loss_functions.py 损失函数类<换行>
请输入具体的网页文本内容，以便我进行翻译。

### 训练
```
CUDA_VISIBLE_DEVICES=1 nohup python train.py > train.out &

nohup python train.py  --load_weight='output/batch_128_39.pth' --had_trained_epoch=40 > v3_lr=0.002_add.out &
```

训练一个模型
### 测试

```
CUDA_VISIBLE_DEVICES=1 nohup python eval.py > test.out &

nohup python test.py  --load_weight='output/batch_128_39.pth' --had_trained_epoch=40 > v3_lr=0.002_add.out &
```
评估模型

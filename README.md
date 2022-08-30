# 古诗生成
使用lstm生成古诗

## 准备
### 环境准备
运行如下代码准备环境
```
conda create -n zh python=3.9
conda activate zh
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

### 数据准备
*数据和训练好的模型都在[这里](https://drive.google.com/drive/folders/11b3KNTUzzoGQWM0XasTPNWICrqn4z8mp?usp=sharing)*

下载好相关文件按照如下位置存放

> ./dataset/processedData/data.mat \
> ./dataset/processedData/vocab.pt \
> ./dataset/raw_data/poetry.txt \
> ./results/net.pt

* 如果想自己重新预处理数据集，可以将 *poetry.txt* 下载后运行如下

```commandline
python run.py --preprocess
```

## 训练 & 测试
### 训练
* 运行如下代码进行训练
```
python run.py --mode train --epochs 120
```

## 测试
* 运行如下代码进行测试
```
python run.py --mode val
```

## 实验结果
测试用 `春`,`夏`,`秋`,`冬`作为头的藏头诗。
* 结果
> 春风三月客，夏序武官城。 秋花颜时节，冬芳樽收逃。\
> 春风正相望，夏木最深林。 秋阴匀暮绿，冬鳞繁宛色。\
> 春来半夜玩，夏满将闲思。 秋浓疏树红，冬歇红绿除。\
> 春日清淮近，夏江农宴归。 秋云居住稀，冬村稀繁白。\
> 春色舍阳华，夏芳归圣越。 秋草深不闲，冬芳处春处。

* 分析
生成的诗句可以看出，诗句读起来并不是很通顺。意思表达也
不清不楚。


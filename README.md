结合句法的方面级情感分析方法研究
---
### 运行环境
* Python 3.9.1
* PyTorch 1.8.1

### 如何运行
---
- Step1 下载glove.840B.300d.zip并解压到`./dataset/glove`目录下
```Bash
wget http://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.840B.300d.zip
unzip glove.840B.300d.zip -d ./dataset/glove/
```
- Step2 构建Vocab
```Bash
# 可以指定不同的数据集进行词汇构建
./build_vocab.sh
```

- Step3 模型训练

```Bash
# 可以指定不同数据集进行模型训练
./train.sh
```
模型 checkpoints 和最优模型保存在 `./saved_models`. 

- 模型评估
```Bash
# 可以指定不同数据集进行模型评估
./evaluate.sh
```

# 图像识别练手项目 CV_Project

使用性别识别作为目标问题的深度学习练手项目

一般的深度学习pytorch训练代码都使用了较为固定的编写方法。

首先使用Dataset和DataLoader读取数据

然后继承``nn.module``类编写神经网络结构

然后编写神经网络训练流程

对于此训练流程，首先需要定义优化器optim和损失loss计算实例，然后通过多个epoch的训练对神经网络的参数进行更新。更新过程中的关键在于优化参数，即首先使用模型计算出待预测数据的结果，然后对这个结果计算loss，然后对这个loss进行梯度传递操作backward，从而使用optim通过梯度下降更新模型参数，不断循环就可以更新模型参数从而训练模型。

## 方案1：基于CNN的图像识别算法

- 主脚本：mainCNN.py
- 模型算法：CIFAR 10 model

![](./asset/image.png)

- 运行方法：运行mainCNN.py即可，loss和accuracy在``CNN/trainingLogs``下以图片形式存放

# 一个新的网络设计方式

## NNNModel

### 训练

* 设备 4090

- 数据集 mnist
- 批次大小 1024
- 显存占用 21301 M
- 一批次速度 29秒
- 26批次达到 94.36 的准确率 不再上涨

## DiyModel()

* 标准做法

### 训练

* 设备 4090

- 数据集 mnist
- 批次大小 1024
- 显存占用 12117 M
- 一批次速度 22秒
- 6批次达到 95.25 的准确率

## DiyModel2

### 训练

* 设备 4090

- 数据集 mnist
- 批次大小 1024
- 显存占用 13477 M
- 一批次速度 25秒
- 6批次达到 95.25 的准确率

* 网络图

- ![网络图.JPG](img/%E7%BD%91%E7%BB%9C%E5%9B%BE.JPG)

* 思路

- 模拟人类神经元思考方式
import os
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms


# 用torch加载MNIST数据集（直接交给GPT）
def load_MNIST_data(root='./datas/mnist', batch_size=8, download=True, resize=(224, 224)):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 加载测试集
    testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# 用torch加载MNIST数据集（直接交给GPT）
def load_cf_data(batch_size=4, resize=(224, 224)):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize),
    ])

    # 加载训练集
    cifar10_train = torchvision.datasets.CIFAR10(root='./datas/cf10', train=True, download=True,
                                                 transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='./datas/cf10', train=False, download=True,
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def main():
    trainloader, testloader = load_MNIST_data()

    print("训练集大小：", len(trainloader.dataset))

    print("测试集大小：", len(testloader.dataset))
    # 训练集大小： 60000
    # 测试集大小： 10000


if __name__ == '__main__':
    main()

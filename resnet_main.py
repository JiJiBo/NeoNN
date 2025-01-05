import torch
import torchvision.models as models

from load_data import load_MNIST_data, load_cf_data

def main():
    train_loader, test_loader = load_cf_data()
    # 加载ResNet模型
    resnet = models.resnet18(pretrained=False, num_classes=10)
    resnet = resnet.cuda()

    # 定义训练超参数
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)

    # 训练ResNet
    for epoch in range(10):
        resnet.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 测试准确率
    correct, total = 0, 0
    resnet.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"ResNet Test Accuracy: {100 * correct / total}%")
if __name__ == '__main__':
    main()
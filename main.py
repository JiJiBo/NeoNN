import os
import shutil

import torch
from tqdm import tqdm

from Config import device
from load_data import load_MNIST_data, load_cf_data
from model import DiyModel


def train():
    model = DiyModel()
    model.to(device=device)
    print(device)
    trainloader, testloader = load_MNIST_data()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 3
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.mkdir("models")
    model.train()
    # best_loss = int(1e9)
    for epoch in range(epochs):
        datas = tqdm(trainloader)
        for data in datas:
            inputs, labels = data[0].to(device=device), data[1].to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            # if l.item() < best_loss:
            #     best_loss = l.item()
            #     torch.save(model.state_dict(), "models/best_model.pth")
            #     print(f"New best loss: {best_loss}")
            datas.set_description(f"Epoch: {epoch + 1}/{epochs} Loss: {l.item()}")

    torch.save(model.state_dict(), "models/last_model.pth")


def predict():
    model = DiyModel()
    model.load_state_dict(torch.load("models/last_model.pth"))
    model.to(device=device)
    model.eval()
    trainloader, testloader = load_MNIST_data()
    datas = tqdm(testloader)
    # 计算准确率

    correct = 0
    count = 0
    with torch.no_grad():
        for data in datas:
            inputs, labels = data[0].to(device=device), data[1].to(device=device)
            count += len(labels)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum().item()
            datas.set_description(f"Accuracy: {100 * correct / count}")
    print(f"Accuracy: {100 * correct / len(testloader.dataset)}")


if __name__ == '__main__':
    train()
    predict()

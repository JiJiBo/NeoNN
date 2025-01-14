import os
import shutil

import torch
from tqdm import tqdm

from Config import device
from load_data import load_MNIST_data, load_cf_data
from model.model import DiyModel
from model.model2 import DiyModel2
from model.model3 import DiyModel3
from model.nnn import NNNModel


def train():
    model = DiyModel3()
    model.to(device=device)
    print(device)
    trainloader, testloader = load_MNIST_data()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20, 30 ],
        gamma=0.1
    )
    epochs = 26
    if not os.path.exists("models"):
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
        scheduler.step()
        predict(model)
    torch.save(model.state_dict(), "models/last_model.pth")


def predict(model=None):
    if model is None:
        model = DiyModel3()
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

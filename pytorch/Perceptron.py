#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: pythonProject
@File   : Perceptron.py
@Author : 91317
@Time   : 2021/2/28 
"""
import time

import torchvision
import torchvision.transforms as transforms
from torch.nn import init
import torch
import numpy as np
import random
import torch.nn as nn
import torch.utils.data as Data

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def get_data_iter():
    # 读取数据
    mnist_train = torchvision.datasets.FashionMNIST(root='F:/Jupyter_note/DL_pytorch/FashionMNIST',
                                                    train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='F:/Jupyter_note/DL_pytorch/FashionMNIST',
                                                   train=False, download=True, transform=transforms.ToTensor())
    # 生成iter
    batch_size = 256
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(Net, self).__init__()
        self.linear_1 = nn.Linear(num_inputs, num_hiddens)
        self.ReLU = nn.ReLU()
        self.linear_2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = x.view(x.shape[0], -1)
        y = self.linear_1(y)
        y = self.ReLU(y)
        y = self.linear_2(y)
        return y

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    net = Net(num_inputs, num_outputs, num_hiddens)
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.1)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    train_iter, test_iter = get_data_iter()
    num_epochs = 5
    # cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    start = time.time()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            optimizer.zero_grad()
            # 计算梯度并更新
            l.backward()
            optimizer.step()
            # 计算预测acc
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    print(time.time() - start)

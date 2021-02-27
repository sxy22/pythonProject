#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Project: pythonProject
@File   : linreg.py
@Author : 91317
@Time   : 2021/2/27 
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


def generate_data(n):
    # 生成n个数据
    torch.manual_seed(10)
    num_inputs = 2
    num_examples = n
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.1, size=labels.size()), dtype=torch.float32)
    return features, labels

if __name__ == '__main__':

    # 获得数据
    num_inputs = 2
    features, labels = generate_data(1000)

    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 随机读取小批量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    # 模型
    net = LinearNet(num_inputs)
    # 初始化参数
    init.normal_(net.linear.weight, mean=0, std=0.1)
    init.constant_(net.linear.bias, val=0)

    # 损失
    loss = nn.MSELoss()
    # optim
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    # 训练
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            # print(output.shape)
            l = loss(output, y.view(output.size()))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))

    for param in net.parameters():
        print(param)
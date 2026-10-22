import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
#创建数据集
true_w = torch.tensor([2.0,-3.4])#区别于前一个，这个有两个权重
true_b = 4.2#偏置
features,labels = d2l.synthetic_data(true_w,true_b,1000)#创建数据集，给予权重和标签
#读取数据集
def load_array(data_arrays,batch_size,is_train = True):#载入数据集，True表示是否是训练，下面是训练就打乱
    dataset = data.TensorDataset(*data_arrays)#存起来，分别传参
    return data.DataLoader(dataset,batch_size,shuffle=is_train)#打乱以后，按照batch_size一批批取数据
#取出小样本
batch_size = 10#步长
data_iter = load_array((features,labels),batch_size)#取出小样本
#设计模型
from torch import nn
net = nn.Sequential(nn.Linear(2,1))#构建一个全连接层
#初始化模型
net[0].weight.data.normal_(0,0.01)#初始化为均值为0的正态分布
net[0].bias.data.fill_(0)#偏置初始化为0

import torch.optim as optim
trainer = optim.SGD(net.parameters(), lr=0.03)  # 学习率0.03
#定义损失函数
loss = nn.MSELoss()
#开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
l = loss(net(features),labels)
print(f'epoch{epoch+1},loss{l:f}')

w = net[0].weight.data
print('w的估计误差：',true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：',true_b - b)
import torch
from torch import nn
from d2l import torch as d2l

def main():
    batch_size = 256 #小样本大小
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)# 加载训练集和测试集

    num_inputs, num_outputs, num_hiddens = 784, 10, 256 #三层
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01) #生成正态的随机数
    b1 = nn.Parameter(torch.zeros(num_hiddens)) #初始化偏置全0
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01) #生成正态的随机参数
    b2 = nn.Parameter(torch.zeros(num_outputs)) #b = 0
    params = [W1, b1, W2, b2] # 存下参数

    def relu(X): #定义激活函数
        return torch.max(X, torch.zeros_like(X)) 

    def net(X): #搭建模型
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return H @ W2 + b2

    loss = nn.CrossEntropyLoss(reduction='none')# 定义损失函数
    num_epochs, lr = 10, 0.1 # 学习次数和学习率
    updater = torch.optim.SGD(params, lr=lr) #迭代器使用随机梯度下降

    for epoch in range(num_epochs): 
        for X, y in train_iter:
            y_hat = net(X) #预测值
            l = loss(y_hat, y)
            updater.zero_grad()# 梯度归零
            l.mean().backward() # 反向传播计算梯度
            updater.step() 
        metric = d2l.Accumulator(2)
        for X, y in test_iter:
            metric.add(d2l.accuracy(net(X), y), y.numel())
        print(f"epoch {epoch+1}, test acc {metric[0]/metric[1]:.3f}")

    input("按回车退出...")

if __name__ == '__main__':
    main()
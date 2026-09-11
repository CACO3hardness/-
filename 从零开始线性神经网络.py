import random
import torch

def synthetic_data(w,b,num_examples): #合成数据，输入真实的参数
    X = torch.normal(0,1,(num_examples,len(w))) #从正态分布中采样
    y = torch.matmul(X,w) + b #前面表示矩阵乘法
    y += torch.normal(0,0.01,y.shape)#添加噪声，让数据更接近真实世界
    return X,y.reshape((-1,1))#把一维向量变成列向量，方便矩阵运算

true_w = torch.tensor([2.0,3.0,4.0])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b,1000)
#读取数据集
def data_iter(batch_size,features,labels): #每次取出一个小批量
    num_examples = len(features)
    indices = list(range(num_examples))#list 转变成列表，准备0~999的索引
    random.shuffle(indices) #打乱样本顺序，让每次迭代看到不同顺序的样本
    for i in range(0,num_examples,batch_size): #按批次遍历
        batch_indices = torch.tensor(
            indices[i: min(i+batch_size,num_examples)])#得到i 到 i + batch_examples这一段的索引
        yield features[batch_indices],labels[batch_indices]#yield不会一次性返回所有数据，每次调用只返回当前批次，下次从暂停的地方开始

#初始化模型参数
w = torch.normal(0,0.01, size=(3,1),requires_grad = True)
b = torch.zeros(1,requires_grad = True)#初始化为0，requires_grad = True跟踪b和w变化

#定义损失函数
def squared_loss(y_hat,y):#均方损失
    return (y_hat - y.reshape(y_hat.shape))**2 / 2;
#定义优化算法
def sgd(params,lr,batch_size): #lr : learning rate,params:包含所有要更新的参数
    #小批量随机梯度下降
    with torch.no_grad(): #这一段不需要追踪和记录
        for param in params:
            param -= lr * param.grad / batch_size#更新参数，步长 = 学习率 * 平均梯度
            param.grad.zero_()#清零 防止累加梯度
# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b
#训练
batch_size = 10;
lr = 0.03#学习率
num_epochs = 3#取样次数
net = linreg#输入模型
loss = squared_loss#均方损失

for epoch in range(num_epochs):#进行三次完整遍历
    for X,y in data_iter(batch_size,features,labels):#取出小数集
        l = loss(net(X,w,b),y)#计算损失
        l.sum().backward()#计算损失函数的梯度
        sgd([w,b],lr,batch_size)#sgd梯度下降调整参数
    with torch.no_grad():#关闭记录
        train_l = loss(net(features,w,b),labels)#做出预测
        print(f'epoch{epoch + 1},loss{float(train_l.mean()):f}')#打印平均（mean）损失
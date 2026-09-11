%matplotlib inline
import torch
import torchvision # 计算机视觉包
from torch.utils import data # 数据加载
from torchvision import transforms # 图像预处理工具
from d2l import torch as d2l

d2l.use_svg_display() # 输出矢量图

trans = transforms.ToTensor() #将图像或者数组变成张量，将像素值缩放，改变形状到通道在前
mnist_train = torchvision.datasets.FashionMNIST(# 加载训练集
    root = "../data",train = True,transform = trans,download = True)
mnist_test = torchvision.datasets.FashionMNIST(# 加载测试集
    root="../data",train=False,transform=trans,download=True)

len(mnist_train),len(mnist_test)
mnist_train[0][0].shape

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels] #数字转文本

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    #scale 控制图像大小
    figsize = (num_cols * scale,num_rows * scale)
    
    #创建子图网格，返回画布fig和坐标轴数组
    _, axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    
    #将axes 展平为一维数组
    axes = axes.flatten()
    
    #遍历所有子图和对应图像
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        #转成能显示的形式
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        #如果有标题设置标题
        if titles:
            ax.set_title(titles[i])

    return axes

X,y = next(iter(data.DataLoader(mnist_train,batch_size = 18)))#
show_images(X.reshape(18,28,28),2,9,titles = get_fashion_mnist_labels(y));#输出图像

batch_size = 256

def get_dataloader_workers():
    return 4
    
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=get_dataloader_workers())# 重新设计一个加载器

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()] #转换成张量
    if resize:# 如果要切换大小就转换一下
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)# 串联起来

    mnist_train = torchvision.datasets.FashionMNIST(#加载训练集
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(#加载测试集
        root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))#返回一个元组，包含这两个集

train_iter,test_iter = load_data_fashion_mnist(32,resize = 64)
for X,y in train_iter:
    print(X.shape,X.dtype,y.shape,y.dtype)
    break;
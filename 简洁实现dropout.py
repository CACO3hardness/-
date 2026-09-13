from IPython import display
import matplotlib_inline.backend_inline
display.set_matplotlib_formats = matplotlib_inline.backend_inline.set_matplotlib_formats

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from d2l import torch as d2l

# 超参数
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

# 网络
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

# 数据
batch_size = 256
trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST(root="./data", train=True,
                                    transform=trans, download=True)
mnist_test = datasets.FashionMNIST(root="./data", train=False,
                                   transform=trans, download=True)

train_iter = DataLoader(mnist_train, batch_size=batch_size,
                        shuffle=True, num_workers=0)
test_iter = DataLoader(mnist_test, batch_size=batch_size,
                       shuffle=False, num_workers=0)

# 训练
num_epochs, lr = 10, 0.5
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    train_loss, train_acc = d2l.train_epoch_ch3(net, train_iter, loss, trainer)
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    print(f'epoch {epoch+1:2d}: '
          f'train loss {train_loss:.4f}, '
          f'train acc {train_acc:.4f}, '
          f'test acc {test_acc:.4f}')
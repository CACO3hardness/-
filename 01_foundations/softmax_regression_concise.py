import torch
from torch import nn
from d2l import torch as d2l

def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    def evaluate_accuracy(net, data_iter):
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in data_iter:
                correct += (net(X).argmax(axis=1) == y).sum().item()
                total += y.numel()
        return correct / total

    for epoch in range(num_epochs):
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            total_loss += l.sum().item()
            correct += (y_hat.argmax(axis=1) == y).sum().item()
            total += y.numel()
        train_acc = correct / total
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch+1}: loss {total_loss/total:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")

if __name__ == '__main__':
    main()
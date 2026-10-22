import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 唯一改动：用 pyd2l 替代 d2l
from pyd2l import torch as d2l


# ============ 1. 初始化模型参数 ============
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# ============ 2. 初始化隐藏状态 ============
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# ============ 3. 前向传播 ============
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


# ============ 4. 包装成模型类 ============
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# ============ 5. 预测函数 ============
def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# ============ 6. 梯度裁剪 ============
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# ============ 7. 训练一个 epoch ============
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# ============ 8. 训练主函数 ============
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False, save_plot="perplexity.png"):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    history = []
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        history.append((epoch + 1, ppl))
        print(f"[epoch {epoch+1:3d}/{num_epochs}] "
              f"ppl={ppl:8.2f}  speed={speed:8.1f} tokens/s", flush=True)

    if save_plot:
        epochs = [h[0] for h in history]
        ppls = [h[1] for h in history]
        plt.figure()
        plt.plot(epochs, ppls, label="train")
        plt.xlabel("epoch")
        plt.ylabel("perplexity")
        plt.legend()
        plt.savefig(save_plot, dpi=120, bbox_inches="tight")
        print(f"困惑度曲线已保存到: {save_plot}")

    print("=" * 50)
    print("time traveller:", predict_ch8("time traveller", 50, net, vocab, device))
    print("traveller     :", predict_ch8("traveller", 50, net, vocab, device))


# ============ 9. 命令行入口 ============
def main():
    parser = argparse.ArgumentParser(description="从零实现 RNN 训练 time_machine 数据集")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--num-hiddens", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--random-iter", action="store_true")
    parser.add_argument("--save-plot", type=str, default="perplexity.png")
    args = parser.parse_args()

    if args.device == "auto":
        device = d2l.try_gpu()
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    train_iter, vocab = d2l.load_data_time_machine(
        args.batch_size, args.num_steps)

    net = RNNModelScratch(
        len(vocab), args.num_hiddens, device,
        get_params, init_rnn_state, rnn)

    train_ch8(net, train_iter, vocab, args.lr, args.num_epochs, device,
              use_random_iter=args.random_iter,
              save_plot=args.save_plot)


if __name__ == "__main__":
    main()
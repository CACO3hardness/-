# Transformer 源码学习

`pytorch_transformer_reference.py` 保存的是 PyTorch `torch.nn.modules.transformer` 模块的学习副本，用于阅读 Transformer、Encoder、Decoder 和多头注意力的组合方式。

需要注意：

- 文件中的 `from .activation`、`from .module` 等是 PyTorch 包内部相对导入。
- 该文件不能脱离 PyTorch 包结构直接执行。
- 如需运行示例，请使用 `torch.nn.Transformer`，不要直接导入这个副本。
- 原始项目和许可证：[PyTorch](https://github.com/pytorch/pytorch)、[BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE)。

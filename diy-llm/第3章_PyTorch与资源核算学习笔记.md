# 第 3 章：PyTorch 与资源核算学习笔记

> 依据 Datawhale 的[《Diy-LLM》第 3 章：PyTorch 与资源核算](https://datawhalechina.github.io/diy-llm/chapter3/chapter3_pytorch%E4%B8%8E%E8%B5%84%E6%BA%90%E6%A0%B8%E7%AE%97.html)整理。

这一章没有直接进入 Transformer，而是先用简单的线性模型讲清楚深度学习训练的共同流程：张量、模型、损失、反向传播、优化器和训练循环。另一条主线是资源核算，也就是在真正训练前先估算显存、计算量和训练时间。

---

## 3.1 为什么需要资源核算

训练大模型的成本很高，不能等代码跑起来以后才发现显存不够，或者训练要几个月。资源核算的目的就是提前回答两个问题：**放不放得下，以及大概要跑多久。**

### 3.1.1 训练时间估算

稠密语言模型预训练常使用一个粗略经验公式：

```text
总计算量（FLOPs）≈ 6 × 参数量 × 训练 token 数
```

得到总计算量后，再除以硬件的实际计算速度：

```text
训练时间 ≈ 总计算量 ÷（GPU 数量 × 单卡有效 FLOP/s）
```

显卡参数表给出的通常是理论峰值，实际训练还会受到数据传输、通信、算子效率等影响，所以需要乘上一个利用率，而不能直接使用峰值。

**Tip:** `6 × 参数量 × token 数` 是快速估算，不是所有模型都严格满足。稀疏模型、长序列注意力和特殊结构还要单独核算。

### 3.1.2 显存估算

训练时显存里不只有模型参数，还包括：

- 参数；
- 梯度；
- 优化器状态；
- 前向传播保存的激活值；
- 临时计算结果和框架缓存。

以朴素 FP32 AdamW 为例，一个参数大约需要：

```text
参数 4 字节 + 梯度 4 字节 + 两份优化器状态 8 字节
= 16 字节/参数
```

这还没有计算激活值，所以只能当作理论起点。batch size、序列长度和层数增大时，激活值也会明显增多。

---

## 3.2 张量（Tensor）

张量是 PyTorch 存储数据的基本单位。输入数据、模型参数、梯度、激活值和优化器状态，最后都以张量形式存在。

### 3.2.1 张量基础

常见创建方式：

```python
import torch

x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
z = torch.zeros(4, 8)
o = torch.ones(4, 8)
r = torch.randn(4, 8)
```

学习张量时先看三个属性：

```python
x.shape   # 每个维度有多长
x.dtype   # 元素的数据类型
x.device  # 位于 CPU 还是 GPU
```

张量的秩就是维度数量。例如 `(4,)` 是一维向量，`(4, 8)` 是二维矩阵。Transformer 中常见的形状是：

```text
[batch, sequence, hidden]
```

拆分多头注意力后，常写成：

```text
[batch, sequence, heads, head_dim]
```

### 3.2.2 张量操作

`view()`、切片和部分转置操作可能只改变看待数据的方式，不复制底层存储，因此比较省内存。

```python
x = torch.arange(12).view(3, 4)
y = x.view(4, 3)
```

转置后的张量可能不连续，此时直接使用 `view()` 会报错，可以先调用：

```python
y = x.transpose(0, 1).contiguous()
```

**Tip:** `.contiguous()` 会重新复制并排列数据，所以它虽然能解决形状问题，也会产生额外内存和时间开销。

逐元素运算是每个位置各算各的，矩阵乘法则会按照行列规则组合信息：

```python
y1 = x * 2
y2 = a @ b
```

若 `a` 的形状是 `(M, K)`，`b` 的形状是 `(K, N)`，结果就是 `(M, N)`。

### 3.2.3 Einops

`einops` 可以直接用名称表达维度，读复杂模型代码时比反复写 `view()`、`transpose()` 更清楚。

```python
from einops import rearrange, reduce

# 把 hidden 拆成 heads 和 head_dim
y = rearrange(x, "b s (h d) -> b s h d", h=8)

# 对 sequence 维求平均
z = reduce(y, "b s h d -> b h d", "mean")
```

个人扩展：第一次读到复杂张量代码时，可以在每一步后面临时打印 `shape`。先把维度流走通，比一开始研究公式更容易定位问题。

---

## 3.3 内存（Memory）

一个张量的内存占用主要由元素数量和数据类型决定：

```text
内存字节数 = 元素数量 × 每个元素的字节数
```

PyTorch 中可以直接计算：

```python
bytes_used = x.numel() * x.element_size()
```

### 3.3.1 常见浮点类型

| 类型 | 每个元素 | 简单特点 |
|---|---:|---|
| FP32 | 4 字节 | 精度和范围较好，但显存占用大 |
| FP16 | 2 字节 | 快且省显存，但数值范围较小 |
| BF16 | 2 字节 | 范围接近 FP32，更适合大模型训练 |

BF16 并不是“比 FP16 更精确”，它主要保留了更大的数值范围，代价是小数精度更低。

### 3.3.2 CPU 与 GPU 内存

张量默认创建在 CPU 上：

```python
x = torch.zeros(32, 32)
y = x.to("cuda:0")
```

也可以直接在 GPU 上创建：

```python
z = torch.zeros(32, 32, device="cuda:0")
```

CPU 和 GPU 之间通过总线传输数据，频繁来回移动会拖慢训练。模型和参与同一次运算的张量也必须位于同一设备。

---

## 3.4 计算效率

### 3.4.1 FLOPs 与 FLOP/s

- FLOPs：完成一次任务总共要做多少次浮点运算，表示计算量。
- FLOP/s：硬件一秒能执行多少次浮点运算，表示速度。

对于 `(M, K) @ (K, N)` 的矩阵乘法，粗略计算量为：

```text
2 × M × K × N FLOPs
```

这里的 2 来自一次乘法和一次加法。

### 3.4.2 自动微分与反向传播

如果参数设置了 `requires_grad=True`，PyTorch 会为相关操作建立计算图：

```python
x = torch.tensor([1., 2., 3.])
w = torch.tensor([1., 1., 1.], requires_grad=True)

pred = x @ w
loss = 0.5 * (pred - 5).pow(2)
loss.backward()
```

`loss.backward()` 会从损失出发，沿计算图反向计算梯度，并把参数梯度放在 `w.grad` 中。一般来说，反向传播比单次前向传播计算量更大，这也是前面经验公式中出现系数 6 的原因。

---

## 3.5 模型构建与训练基础

### 3.5.1 参数初始化

可训练参数通常用 `nn.Parameter` 表示。初始化不能随便放大，否则网络层数增加后可能出现数值爆炸或消失。

```python
import torch.nn as nn

w = nn.Parameter(torch.randn(input_dim, output_dim) / input_dim**0.5)
```

课程用 Xavier 思路说明：根据输入维度缩放权重，可以让不同层的数值范围更稳定。

### 3.5.2 自定义模型

自定义模型通常继承 `nn.Module`，在 `__init__()` 里定义层，在 `forward()` 里写数据如何流动。

```python
class SimpleModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
```

`model.parameters()` 返回可训练参数，`model.state_dict()` 返回参数名称和值，保存模型时经常会用到。

### 3.5.3 随机性与复现

参数初始化、数据打乱和 Dropout 都会带来随机性。为了方便调试，可以同时设置三个随机种子：

```python
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

**Tip:** 设置随机种子可以提高复现性，但不同 GPU、驱动、算子实现和并行方式仍可能产生细小差异。

### 3.5.4 数据加载

大规模 token 数据不能一次全部放入内存，可以使用 `numpy.memmap` 按需读取磁盘文件，再随机切出形状为 `[batch, sequence]` 的训练批次。

如果使用 GPU，固定内存 `pin_memory()` 配合 `non_blocking=True` 可以让数据传输和 GPU 计算尽量重叠，减少 GPU 等待。

### 3.5.5 优化器

优化器根据梯度更新参数。最简单的 SGD 更新规则是：

```text
新参数 = 旧参数 - 学习率 × 梯度
```

Adam、AdamW、AdaGrad 等优化器还会保存历史梯度统计，因此优化器本身也会占显存。

每一步训练后要清理梯度，否则 PyTorch 会继续累积：

```python
optimizer.zero_grad(set_to_none=True)
```

### 3.5.6 训练循环

最基本的训练顺序是：

```text
取一批数据
→ 前向传播
→ 计算损失
→ 反向传播
→ 更新参数
→ 清空梯度
```

对应代码：

```python
for x, y in dataloader:
    pred = model(x)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

### 3.5.7 检查点

检查点是训练过程的存档。想要真正恢复训练，至少要保存模型参数、优化器状态和当前训练步数；实际项目还常保存学习率调度器、随机数状态和混合精度状态。

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step,
}, "checkpoint.pt")
```

只保存模型参数可以用于推理，但不一定能无缝继续训练。

### 3.5.8 混合精度训练

混合精度不是把所有内容都强制变成低精度，而是让适合的矩阵运算使用 BF16/FP16，把数值敏感的操作保留为较高精度。

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    pred = model(x)
    loss = loss_fn(pred, y)
```

这样可以减少显存占用，并利用 GPU 的低精度计算单元，但仍要观察损失是否出现 `nan` 或 `inf`。

---

## 3.6 算术强度与 Roofline

GPU 执行一个操作时，既要进行计算，也要搬运数据。算术强度表示：

```text
算术强度 = FLOPs ÷ 搬运的字节数
```

- 算术强度低：更多时间花在搬数据上，属于 memory-bound。
- 算术强度高：更多时间花在计算上，属于 compute-bound。

ReLU 这类逐元素操作通常计算少、读写多，容易受内存带宽限制；较大的矩阵乘法能重复利用数据，更容易发挥 Tensor Core 的计算能力。

Roofline 图把算术强度放在横轴、实际计算速度放在纵轴。曲线转折点左侧主要受内存带宽限制，右侧主要受硬件峰值算力限制。

个人扩展：程序“GPU 占用率高”不等于计算单元真的高效工作。线程可能正在等待内存或同步，因此分析性能时还要看显存带宽、算子时间和有效吞吐量。

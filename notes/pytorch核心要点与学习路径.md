# PyTorch 核心要点与学习路径

> 结合 CS336 assignment1 中 `utils.py` 暴露的问题整理，2026-07。

---

## 一、核心要点

### 1. Tensor 三要素：shape / dtype / device

- 每个张量都有 `shape`、`dtype`、`device`，三者不匹配是最常见的运行时错误来源。
- **规则：新建张量必须跟随上下文的 device/dtype**。
  - `torch.arange(...)`、`torch.zeros(...)` 等工厂函数默认在 CPU 上创建 float32。
  - 在 `forward` 里新建张量时写 `torch.arange(n, device=x.device)`，否则 GPU 上直接报错。
    （对应 bug：RoPE 中 `torch.arange` 未指定 device）
- 标准做法（PyTorch 源码惯例，见 `nn/modules/linear.py`）：

  ```python
  def __init__(self, in_features, out_features, device=None, dtype=None):
      factory_kwargs = {"device": device, "dtype": dtype}
      self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
  ```

  （对应 bug：`Linear`/`Embedding`/`RMSNorm` 接收了 device/dtype 却没用）

### 2. Autograd：动态计算图

- 每次前向传播动态构建计算图，`backward()` 沿图反向求梯度，图默认用完即弃。
- 梯度能否回传，取决于**运算是否被 autograd 记录**。以下操作会**断梯度**：
  - `tensor.detach()`、`with torch.no_grad():`
  - `tensor.data`（危险，绕过 autograd）
  - **`load_state_dict()`——它是数值拷贝，不是图连接**。
    （对应 bug：`Swiglu.forward` 里用 `load_state_dict` 灌权重，外部传入的 weight 收不到梯度）
- 想让传入的权重参与求导，直接用函数式算子：`x @ w.T`、`F.linear(x, w)`、`einsum`。
- 叶子节点（`requires_grad=True` 的输入/参数）才会累积 `.grad`；中间结果默认不保留。
- 梯度是**累加**的，所以训练循环里要 `optimizer.zero_grad()`。

### 3. nn.Module：状态与计算分离

- **`__init__` 放状态（参数、子模块、buffer），`forward` 只做计算**。
  在 `forward` 里 new 子模块 = 每步都重新初始化 + 白白分配内存 + 参数不被优化器管理。
  （对应 bug：`Swiglu.forward` 每次调用新建 3 个 `Linear`）
- 注册机制：`Module.__setattr__` 会拦截赋值——
  - 赋 `nn.Parameter` → 自动进 `self._parameters`，出现在 `parameters()` / `state_dict()`
  - 赋 `nn.Module` → 自动进 `self._modules`，参数递归可见
  - 普通 Tensor → 什么都不注册，`.to(device)` 也不会搬它
- 三种"模块里的张量"：

  | 类型 | 求梯度 | 进 state_dict | 随 .to() 移动 | 用途 |
  |---|---|---|---|---|
  | `nn.Parameter` | 是 | 是 | 是 | 可学习权重 |
  | `register_buffer(...)` | 否 | 默认是（`persistent=False` 则否） | 是 | RoPE cos/sin 缓存、causal mask、BN 统计量 |
  | 普通属性 Tensor | 否 | 否 | **否** | 尽量别用 |

- 调用用 `module(x)` 而不是 `module.forward(x)`：`__call__` 才会走 hooks。
- `model.train()` / `model.eval()` 只影响 Dropout、BatchNorm 等；`torch.no_grad()` 是另一回事，推理时两个都要。
- `.to(device)` 的原理是 `Module._apply`，递归遍历 `_parameters`、`_buffers`、`_modules`。

### 4. 初始化

- 初始化不是随便填随机数，std 的选择决定深层网络信号是否爆炸/消失：
  - Xavier/Glorot：`std = sqrt(2 / (fan_in + fan_out))`，配 tanh/线性
  - Kaiming/He：`std = sqrt(2 / fan_in)`，配 ReLU
  - CS336 spec：Linear 用 `trunc_normal_(std=sqrt(2/(d_in+d_out)), a=-3σ, b=3σ)`
- **Norm 层的 gain 初始化为 1、bias 为 0**（恒等变换起步），不是 trunc_normal。
  （对应 bug：`RMSNorm.g` 用了 trunc_normal_）
- Embedding：`trunc_normal_(std=1, a=-3, b=3)`（按 CS336 spec）。
- 源码参考：`torch/nn/init.py`（计算 fan_in/fan_out 的逻辑值得读一遍）。

### 5. 数值稳定性

- fp16 最大值约 65504，平方求和极易溢出；bf16 范围大但精度低。
- 模式：**低精度存储，fp32 计算敏感部分**：
  - RMSNorm/LayerNorm：先 `x.float()` 再归一化，最后 cast 回原 dtype（你已做对）
  - Softmax：先减 `max` 再 `exp`（你已做对）
  - 混合精度训练：`torch.autocast` + `GradScaler`（fp16 时）
- Attention 中 `masked_fill(~mask, -inf)`：mask 应为 bool 张量；注意 mask 为 None 的分支。

### 6. 性能意识

- **能预计算的不要在 forward 里重复算**：RoPE 的 cos/sin 表在 `__init__` 里按 `max_seq_len` 算好，`register_buffer(persistent=False)`，forward 只做索引。
- 用现成算子：`F.silu(x)` 优于手写 `x * sigmoid(x)`（有 fused kernel）。
- `einsum`/`einops` 表达清晰且不比手写慢；避免不必要的 `.contiguous()`、`.cpu()` 同步点。
- 大矩阵乘远比小算子多次调用高效——这是 batch 一切的原因。

### 7. 常见坑速查

| 现象 | 原因 |
|---|---|
| device mismatch 报错 | forward 里新建张量没指定 `device=x.device` |
| 参数没被训练 | 普通 Tensor 没包 `nn.Parameter`；或 forward 里现建模块 |
| loss 不降、梯度为 None | 中间 `detach`/`.data`/`load_state_dict` 断图 |
| 保存加载后行为不同 | 忘了 `model.eval()`；buffer 没注册 |
| fp16 下 NaN | 归一化/softmax/log 没上 fp32 或没减 max |
| 显存持续上涨 | 记录 loss 时没 `.item()`，把整张计算图存进了 list |

---

## 二、学习路径

### 阶段 0：对照修复（1-2 天，最优先）

拿着上面的要点回去修 `assignment1-basics/cs336_basics/utils.py`：

- [ ] device/dtype 透传（factory_kwargs 模式）
- [ ] SwiGLU 改为 `__init__` 持有权重，forward 用函数式矩阵乘
- [ ] RoPE 改成 Module + buffer 预计算，`arange` 指定 device
- [ ] 按 handout 修正三处初始化
- [ ] attention 处理 `mask=None`
- [ ] 命名规范（snake_case 函数、`d_model` 拼写、不遮蔽 `max`）

修完跑 `uv run pytest` 验证。**带着问题学，比通读教程快十倍。**

### 阶段 1：pytorch-deep-learning 仓库（约 1 周，挑着做）

路径：`~/Documents/LocalRepo/cs336/pytorch-deep-learning`

| Notebook | 学什么 | 优先级 |
|---|---|---|
| `00_pytorch_fundamentals` | 张量、dtype/device、广播 | 高（快速过） |
| `01_pytorch_workflow` | nn.Module、训练循环、save/load | 高 |
| `05_going_modular` | 工程化组织 | 中 |
| `08_paper_replicating`（复现 ViT） | 从论文到代码，与 CS336 直接互补 | **最高** |
| 02/03/04/06/07/09 | 分类/CV/部署 | 跳过 |

### 阶段 2：源码精读（穿插进行，每次 30-60 分钟）

位置：`assignment1-basics/.venv/lib/python3.13/site-packages/torch/`
（或 github.com/pytorch/pytorch 在线读；编辑器 Cmd+点击 跳转；IPython 里 `nn.Linear??`）

按顺序：

1. `nn/modules/linear.py` —— 最短最典型：Parameter 注册、reset_parameters、factory_kwargs
2. `nn/modules/sparse.py` —— Embedding
3. `nn/modules/normalization.py` —— RMSNorm/LayerNorm，看 gain 怎么初始化
4. `nn/modules/module.py` —— 核心：`__setattr__`、`register_buffer`、`state_dict`、`_apply`
5. `nn/init.py` —— 各初始化的 std 计算

C++ 层（aten/、csrc/、autograd 引擎）现阶段不碰。

### 阶段 3：官方文档（机制类疑问的最终出处）

- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) —— 必读
- [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [Modules note](https://pytorch.org/docs/stable/notes/modules.html)
- [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)（用 GPU 后读）
- [Numerical accuracy](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)

### 阶段 4：回到 CS336 主线

- assignment1 剩余部分（tokenizer、完整 Transformer、训练循环）本身就是最好的练习
- 工具化保障：项目里配 `ruff`，命名/未用导入类问题让工具兜底
- 每实现一个模块，先读 handout 对应小节的 spec 再动手，写完对照 PyTorch 官方同名实现 diff 一遍

### 心法

1. **以修代学**：所有知识点都从"我踩过的坑"出发建立索引。
2. **源码是一手资料**：教程会过时，`nn/modules/` 不会骗你。
3. **深度优先**：你已在写 Transformer，不需要再刷广度型课程。

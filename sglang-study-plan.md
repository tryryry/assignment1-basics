# SGLang 代码学习与首个 PR 计划

## 目标与时间

- 开始日期：2026-08-26
- 工作日：每天 2 小时
- 周六、周日：每天 6 小时
- 每周投入：22 小时
- 第 3 周目标：完成首个可提交的小型 PR
- 第 4 周目标：提交 PR、处理 CI 和 review，并确定第二个贡献方向

默认具备 Python 和 PyTorch 基础。学习目标不是通读全部代码，而是掌握一条完整请求链、能够编写回归测试，并完成一次小而完整的贡献。

## 核心学习路径

```text
HTTP API
  -> TokenizerManager：校验、分词、请求状态、IPC
  -> Scheduler：接收请求、组批、调度、KV Cache
  -> ModelRunner：模型前向与采样
  -> Scheduler：处理执行结果
  -> TokenizerManager：流式或非流式返回
```

核心入口：

- [HTTP generate_request](../sglang/python/sglang/srt/entrypoints/http_server.py#L790)
- [TokenizerManager.generate_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L589)
- [Scheduler.event_loop_normal](../sglang/python/sglang/srt/managers/scheduler.py#L1548)
- [Scheduler.run_batch](../sglang/python/sglang/srt/managers/scheduler.py#L3206)
- [ModelRunner.forward](../sglang/python/sglang/srt/model_executor/model_runner.py#L3000)

## 每日执行模板

### 工作日 2 小时

1. 10 分钟：回顾上一天的结论。
2. 70 分钟：阅读代码或实现功能。
3. 30 分钟：运行代码、测试或调试。
4. 10 分钟：记录结论、疑问和第二天入口。

### 周末 6 小时

1. 2 小时：深度阅读。
2. 2 小时：调试、实验或测试。
3. 1.5 小时：实现代码或绘制流程图。
4. 30 分钟：复盘并整理下周任务。

每天必须留下至少一个可验证产物：调用图、对象转换表、实验记录、失败测试、修复提交或 PR 文案。

---

## 第 1 周：跑起来并掌握请求主链

### Day 1｜8 月 26 日，周三｜2 小时

**目标：建立仓库整体地图。**

- 阅读 SGLang [README](../sglang/README.md)。
- 阅读[贡献指南](../sglang/docs/developer_guide/contribution_guide.md)。
- 浏览 [python/sglang/srt](../sglang/python/sglang/srt/) 一级目录。
- 为以下目录各写一句职责说明：
  - `entrypoints`
  - `managers`
  - `model_executor`
  - `mem_cache`
  - `layers`
  - `models`

**验收：** 不看代码也能解释上述目录的职责和大致依赖关系。

### Day 2｜8 月 27 日，周四｜2 小时

**目标：建立最小开发和测试闭环。**

- 从源码安装 SGLang Python 包。
- 阅读[测试系统说明](../sglang/test/README.md)。
- 阅读[单元测试规范](../sglang/test/registered/unit/README.md)。
- 运行一个现有 CPU 单元测试。
- 确认 pre-commit 可以运行。

**验收：** 至少一个现有测试在本地成功执行，并记录完整命令。

### Day 3｜8 月 28 日，周五｜2 小时

**目标：理解 HTTP 请求入口。**

- 阅读 [HTTP generate_request](../sglang/python/sglang/srt/entrypoints/http_server.py#L790)。
- 对比流式和非流式请求。
- 查看 `/encode` 和 `/classify` 如何复用 TokenizerManager。
- 理解异常如何转换为 HTTP 响应。
- 理解客户端断连如何处理。

**验收：** 画出从 `/generate` 到 `TokenizerManager.generate_request` 的调用图。

### Day 4｜8 月 29 日，周六｜6 小时

**目标：掌握 TokenizerManager 请求发送路径。**

按顺序阅读：

1. [TokenizerManager.__init__](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L257)
2. [TokenizerManager.generate_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L589)
3. [_tokenize_one_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L793)
4. [_validate_one_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L948)
5. [_send_one_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1331)

时间分配：

- 2 小时：阅读代码并追踪字段。
- 2 小时：整理单请求生命周期。
- 1 小时：理解 `GenerateReqInput` 和 `TokenizedGenerateReqInput`。
- 1 小时：用断点或日志验证调用顺序。

**验收：** 完成对象阶段表，至少包含对象类型、关键字段、创建位置、消费位置和所属进程。

### Day 5｜8 月 30 日，周日｜6 小时

**目标：掌握响应、流式输出和取消逻辑。**

阅读：

- [_wait_one_response](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1446)
- [_handle_batch_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1556)
- [abort_request](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1677)
- [handle_loop](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1847)
- [_handle_batch_output](../sglang/python/sglang/srt/managers/tokenizer_manager.py#L1862)

重点回答：

- `rid_to_state`、`event` 和 `out_list` 如何协作？
- Scheduler 输出如何唤醒等待中的 HTTP 请求？
- streaming chunk 如何合并？
- 客户端断连后 abort 如何到达 Scheduler？
- 请求状态在哪些成功或异常路径上被删除？

**验收：** 完成一张包含请求发送和响应返回的完整时序图。

---

## 第 2 周：理解调度、Batch 和 KV Cache

### Day 6｜8 月 31 日，周一｜2 小时

**目标：理解跨组件的数据结构。**

- 阅读 [io_struct.py](../sglang/python/sglang/srt/managers/io_struct.py)。
- 整理 API 输入对象。
- 整理 tokenized 输入对象。
- 整理 Scheduler 和 Detokenizer 输出对象。
- 标记哪些对象会跨进程传输。

**验收：** 完成请求和响应对象转换表。

### Day 7｜9 月 1 日，周二｜2 小时

**目标：理解 Scheduler 的组成。**

- 阅读 [Scheduler.__init__](../sglang/python/sglang/srt/managers/scheduler.py#L309)。
- 不逐行深挖，把初始化过程分成：
  - 配置
  - IPC
  - ModelWorker
  - 内存池
  - KV Cache
  - 调度组件
  - 指标和 watchdog

**验收：** 完成 Scheduler 组件关系图。

### Day 8｜9 月 2 日，周三｜2 小时

**目标：理解正常调度和 overlap 调度。**

- 阅读 [event_loop_normal](../sglang/python/sglang/srt/managers/scheduler.py#L1548)。
- 阅读 [event_loop_overlap](../sglang/python/sglang/srt/managers/scheduler.py#L1578)。
- 回答：
  - 请求在哪里接收？
  - Batch 在哪里创建？
  - Batch 结果在哪里处理？
  - overlap 调度重叠了哪些 CPU/GPU 工作？
  - 为什么需要保存或复制 Batch 状态？

**验收：** 用伪代码重写两个 event loop 的核心流程。

### Day 9｜9 月 3 日，周四｜2 小时

**目标：理解请求如何组成 Batch。**

- 阅读 [schedule_batch.py](../sglang/python/sglang/srt/managers/schedule_batch.py)。
- 重点理解：
  - `Req`
  - `ScheduleBatch`
  - prefill、extend 和 decode
  - `ForwardMode`
  - `req_pool_indices`
  - `seq_lens`
  - `out_cache_loc`

**验收：** 手工模拟两个不同长度请求进入同一个 Batch。

### Day 10｜9 月 4 日，周五｜2 小时

**目标：理解调度策略。**

- 阅读 [schedule_policy.py](../sglang/python/sglang/srt/managers/schedule_policy.py)。
- 理解 FCFS、priority scheduling 和 waiting queue 排序。
- 找出适合纯 CPU 单测的函数。
- 设计至少五个边界测试：
  - 相同优先级
  - 不同优先级
  - 相同到达时间
  - 空队列
  - 非法优先级或未启用 priority

**验收：** 形成测试矩阵，但暂时不修改实现。

### Day 11｜9 月 5 日，周六｜6 小时

**目标：理解一次 Batch 执行。**

阅读：

- [Scheduler.run_batch](../sglang/python/sglang/srt/managers/scheduler.py#L3206)
- [ModelRunner](../sglang/python/sglang/srt/model_executor/model_runner.py#L349)
- [ModelRunner.forward](../sglang/python/sglang/srt/model_executor/model_runner.py#L3000)

追踪：

```text
ScheduleBatch
  -> ForwardBatch
  -> ModelWorker
  -> ModelRunner
  -> model.forward
  -> logits
  -> sampling
  -> BatchResult
```

**验收：**

- 能解释 prefill 和 decode 的输入形状差异。
- 能解释为什么 decode 通常受显存带宽限制。
- 能指出 Scheduler 和 ModelRunner 的责任边界。

### Day 12｜9 月 6 日，周日｜6 小时

**目标：理解 Radix Cache 和 KV Cache 基础。**

- 浏览 [mem_cache](../sglang/python/sglang/srt/mem_cache/)。
- 阅读 [radix_cache.py](../sglang/python/sglang/srt/mem_cache/radix_cache.py)。
- 阅读 [memory_pool.py](../sglang/python/sglang/srt/mem_cache/memory_pool.py)。
- 理解 prefix cache 命中。
- 理解 request-to-token 和 token-to-KV 映射。
- 运行对应单元测试。
- 修改一个测试输入，观察 match、insert 和 evict 行为。

**验收：** 完成 token、request、KV slot 之间的映射图。

---

## 第 3 周：从阅读代码转向修改代码

### Day 13｜9 月 7 日，周一｜2 小时

**目标：学习真实 PR 的贡献模式。**

- 阅读最近 5 至 10 个只改动以下区域的小 PR：
  - TokenizerManager
  - Scheduler
  - `io_struct`
  - unit tests
- 关注：
  - PR 描述结构
  - 改动规模
  - mock 方法
  - 测试注册
  - reviewer 关注点

**验收：** 总结五条 SGLang 实际贡献习惯。

### Day 14｜9 月 8 日，周二｜2 小时

**目标：选择首个 PR 候选。**

优先选择：

1. 可稳定复现的小 bug。
2. 缺少边界覆盖的 CPU 单测。
3. 输入校验或错误消息问题。
4. 纯函数中的重复或错误逻辑。
5. 文档与实现不一致。

首个 PR 暂时避开：

- CUDA kernel
- 分布式通信
- speculative decoding
- 大规模 Scheduler 重构
- 性能关键路径重写
- 新模型支持

**验收：** 确定一个主候选和一个备用候选。

### Day 15｜9 月 9 日，周三｜2 小时

**目标：完成最小复现。**

- 编写独立复现代码或失败测试。
- 确认当前 main 分支确实失败。
- 使用 `git blame` 和历史 PR 理解代码背景。
- 搜索是否已有相同 issue 或 PR。

**验收：** 获得一个稳定失败的测试；此时不要先写修复。

### Day 16｜9 月 10 日，周四｜2 小时

**目标：设计最小修复。**

- 明确预期行为。
- 明确不能改变的行为。
- 列出正常、边界和异常测试。
- 将改动限制在尽量少的文件中。
- 判断是否处于每请求性能关键路径。

**验收：** 写出简短根因说明和测试矩阵。

### Day 17｜9 月 11 日，周五｜2 小时

**目标：实现第一版修复。**

- 不顺便重构无关代码。
- 避免增加 CPU/GPU 同步。
- 保持类型安全。
- 只在不直观的位置添加注释。

**验收：** 失败测试由红转绿，原有相关测试仍然通过。

### Day 18｜9 月 12 日，周六｜6 小时

**目标：完善测试和 CI 接入。**

- 使用 `CustomTestCase`。
- 优先编写 CPU 单测。
- mock 模型加载和 GPU 依赖。
- 按源目录镜像放入 [test/registered/unit](../sglang/test/registered/unit/)。
- 按规范添加 CI 注册。
- 检查 changed-lines coverage。

**验收：**

- 目标测试本地全部通过。
- 测试不启动 server。
- 测试不加载真实模型权重。
- 测试失败时能够准确暴露此次 bug。

### Day 19｜9 月 13 日，周日｜6 小时

**目标：完成 PR 级自检。**

- 运行目标测试。
- 运行同目录相关测试。
- 对改动文件运行 pre-commit。
- 检查完整 `git diff`。
- 删除调试日志和临时代码。
- 编写 PR 描述：
  - 问题
  - 根因
  - 修复
  - 测试
  - 风险

**里程碑：** 首个 PR 达到可提交状态。

---

## 第 4 周：提交 PR 并建立持续贡献能力

### Day 20｜9 月 14 日，周一｜2 小时

**目标：提交 draft PR。**

- 使用具体标题，不写笼统的 `fix bug`。
- 关联对应 issue。
- 描述复现方法。
- 列出完整验证命令。
- 主动说明未覆盖的 GPU 或 E2E 场景。

**验收：** Draft PR 创建完成。

### Day 21｜9 月 15 日，周二｜2 小时

**目标：模拟 reviewer 审查。**

逐项检查：

- 是否存在更简单的修复？
- 是否改变外部 API 行为？
- 是否引入每请求额外开销？
- 是否遗漏 batch、stream 或 abort 分支？
- 测试是否只验证实现细节？
- 错误路径是否会泄漏请求状态？

**验收：** 完成一次自我 review，并解决发现的问题。

### Day 22｜9 月 16 日，周三｜2 小时

**目标：理解并处理 CI。**

- 区分代码失败、环境失败和 flaky。
- 定位失败所属测试 stage。
- 本地只复现相关测试。
- 不通过盲目重跑掩盖确定性失败。

**验收：** 每个失败项都有明确分类和处理结论。

### Day 23｜9 月 17 日，周四｜2 小时

**目标：处理 review 意见。**

- 每条评论先确认问题本质。
- 同类评论一次性解决。
- 优先添加回归测试，再修改实现。
- 回复时说明改动内容和验证结果。

**验收：** 所有已处理评论均有代码或解释支撑。

### Day 24｜9 月 18 日，周五｜2 小时

**目标：复盘首个 PR。**

回答：

- 最耗时的模块是什么？
- 哪个调用关系最容易误解？
- 本地测试环境有什么坑？
- reviewer 最关注什么？
- 下一个 PR 应继续深入哪个方向？

**验收：** 形成下一阶段的学习重点。

### Day 25｜9 月 19 日，周六｜6 小时

**目标：选择一个长期专项。**

可选方向：

- Serving/API：TokenizerManager 与 OpenAI API。
- 调度：Scheduler、priority、batch、chunked prefill。
- 缓存：Radix Cache、HiCache、内存池。
- 模型支持：模型加载、量化和权重映射。
- 性能：overlap、CUDA Graph、benchmark 和 profiling。
- 推测解码：draft、verify 和 EAGLE。

阅读所选方向的设计文档、近期 PR 和对应测试。

**验收：** 形成第二个 PR 候选列表。

### Day 26｜9 月 20 日，周日｜6 小时

**目标：完成第二个候选的前期分析。**

- 搜索已有 issue 和 PR。
- 追踪完整调用链。
- 编写失败测试或 benchmark。
- 分析修改范围。
- 设计验证方案。

**验收：** 能独立判断该候选是否适合贡献。

### Day 27｜9 月 21 日，周一｜2 小时

**目标：建立个人贡献检查表。**

```text
[ ] 问题可以稳定复现
[ ] 根因明确
[ ] 改动范围最小
[ ] 正常、边界、异常测试齐全
[ ] 目标测试通过
[ ] 相关测试通过
[ ] pre-commit 通过
[ ] 没有关键路径额外同步
[ ] 没有调试代码
[ ] PR 描述完整
```

**验收：** 用该清单重新检查第一个 PR。

### Day 28｜9 月 22 日，周二｜2 小时

**目标：最终能力验收。**

在不查笔记的情况下解释：

1. `/generate` 请求如何进入 Scheduler。
2. 单请求和 batch 请求如何分词。
3. Scheduler 如何选择下一个 Batch。
4. prefill 与 decode 有什么区别。
5. KV Cache 如何与请求关联。
6. 输出如何回到 TokenizerManager。
7. streaming 与 abort 如何工作。
8. 如何添加并注册 CPU 单元测试。
9. runtime 改动需要验证哪些性能和并发风险。

**验收：** 每个问题都能指出相应源码入口，并能独立开始第二个 PR。

---

## 达到“可以提交 PR”的判断标准

- 能完整追踪一条生成请求的主链。
- 能在 30 分钟内找到一个行为对应的代码位置。
- 能为 SRT 组件编写不加载模型的单元测试。
- 能先写失败测试，再做最小修复。
- 知道何时需要 CPU、GPU、E2E、准确率或性能验证。
- 能提交一个改动约 1 至 3 个源文件并带回归测试的小型 PR。

按照本计划执行，预计第 3 周可以提交首个小 PR，4 至 6 周后可以开始独立处理普通 bug。

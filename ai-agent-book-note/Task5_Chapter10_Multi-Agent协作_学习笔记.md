# Task 5：Chapter 10 Multi-Agent 协作学习笔记

## 一、本章要解决的问题

多 Agent 系统通过分工、通信和相互验证完成单个 Agent 难以承担的任务，但增加角色数量本身不会自动提升能力。真正需要回答的是：

1. 多个 Agent 是否共享上下文；
2. 控制权按什么拓扑流动；
3. Agent 之间用什么格式传递任务、状态和产物；
4. 谁负责判断完成、取消剩余任务和处理失败；
5. 多 Agent 引入的新信息是否足以覆盖额外成本。

多 Agent 相对单 Agent 的核心判据是：协作过程中是否获得了单个 Agent 生成时没有的新信息。测试结果、真实环境状态、渲染截图和不同数据源都属于新信息；多个相同模型围绕同一段文本反复讨论，通常只是增加计算量。

## 二、两个基本设计维度

### （一）上下文是否共享

**共享上下文**表示后续角色继承前序角色的完整轨迹，只替换系统提示词或加载新的角色 Skill。优点是信息不丢失，缺点是上下文快速膨胀，旧阶段信息会干扰当前角色。

**不共享上下文**表示每个 Agent 拥有独立的消息历史、工具和状态。它们通过结构化参数、共享文件或消息总线协作。优点是隔离、并发和专业化更好，缺点是需要显式处理信息同步、重复消息和调试链路。

### （二）协作拓扑

1. **对等协作**：少量 Agent 平等迭代，例如 Proposer 与 Reviewer。
2. **管理者模式**：Manager 负责任务分解、分配、监控和汇总，Worker 执行子任务。
3. **去中心化模式**：没有持续在线的中心控制者，Agent 根据协议互相移交任务。

上下文共享和协作拓扑是两个独立维度。共享上下文也可以顺序切换多个角色；管理者模式的各 Worker 通常不共享上下文。

## 三、通信机制：多 Agent 版本的 IPC

不共享上下文时，协作对应操作系统的两类进程通信：共享存储和消息传递。

| 操作系统        | 多 Agent 系统      |
| --------------- | ------------------ |
| 进程内存        | Agent 私有轨迹     |
| CPU             | LLM                |
| 内核            | Agent Runtime      |
| 系统调用        | 工具调用           |
| `fork`          | 创建子 Agent       |
| `kill`          | 取消子 Agent       |
| `wait` / 退出码 | 等待结构化结果     |
| 共享内存        | 共享文件系统       |
| 消息传递        | 工具参数或消息总线 |

### （一）三种通信方式

1. **工具调用参数**：把下游 Agent 封装成工具，上游传入结构化任务，适合同步、边界清晰的调用。
2. **共享文件系统**：大文档、代码和长期产物写入文件，消息只传路径和摘要。
3. **消息总线**：Agent 发布和订阅异步消息，适合并行 Worker、进度更新和级联终止。

### （二）最小消息信封

```python
from dataclasses import dataclass
from typing import Any, Literal

MessageType = Literal[
    "task_assigned", "status_update", "result",
    "needs_input", "cancel", "cancel_ack"
]

@dataclass
class Envelope:
    message_id: str
    sender: str
    recipient: str
    type: MessageType
    task_id: str
    payload: dict[str, Any]
    created_at: str
```

`Envelope` 是跨 Agent 的协议对象，不是普通聊天文本。`message_id` 用于去重，`task_id` 关联同一任务，`type` 决定接收方如何处理，`payload` 承载结构化数据。

### （三）消息总线接口

```python
class MessageBus:
    def publish(self, envelope: Envelope) -> None: ...
    def subscribe(self, recipient: str, handler) -> None: ...

bus.subscribe("worker-1", worker.handle)
bus.publish(Envelope(..., type="task_assigned", ...))
```

Runtime 创建并持有消息总线；Manager 和 Worker 各自订阅自己的消息；发送方不直接持有接收方的上下文。消息总线只负责投递，不负责判断任务是否真的完成。

## 四、共享上下文中的角色切换

共享轨迹时有两种常见方式：

| 方式                | 角色载体                  | 工具可见性             | 缓存影响                       | 强制边界                    |
| ------------------- | ------------------------- | ---------------------- | ------------------------------ | --------------------------- |
| `transfer_to_agent` | 替换 system prompt        | 通常替换为角色专属工具 | 前缀变化，缓存受影响           | 强，可在 schema 层隐藏工具  |
| Skill               | 固定 system，按需加载正文 | 通常工具全集不变       | 内容追加到轨迹，稳定前缀可复用 | 弱，需要 Harness 另做权限门 |

角色差异主要是知识、流程和写作方式时，可优先加载 Skill；角色差异涉及资金、合规、数据访问或工具隔离时，应使用独立 Agent 或角色转移，并由 Harness 强制权限。

控制链如下：

```text
当前角色读取共享轨迹
  → 调用 transfer_to_agent 或 load_skill
  → Harness 保存原轨迹
  → 替换角色提示/工具，或把 Skill 正文追加到轨迹
  → 同一任务继续调用模型
```

共享上下文减少交接损失，但不适合把大量独立探索都堆入一个轨迹。

## 五、不共享上下文时的文件系统

成熟系统通常把虚拟文件系统分成四类区域：

1. **Agent Scratchpad**：单个实例私有，保存草稿、临时文件和调试日志，实例结束后可清理。
2. **Shared Workspace**：多 Agent 和用户可见，保存正式中间产物与最终交付物。
3. **Persistent Memory**：跨任务保留的记忆和知识，有独立写入策略。
4. **Read-only Resources**：系统模板、规范和公共资料，只读挂载。

Worker 的试错过程留在 Scratchpad，只把通过验证的产物提交到 Shared Workspace。Manager 上下文保存文件索引、摘要和状态，不保存整本书、整份代码或所有 Worker 轨迹。

多个 Agent 同时写共享文件会产生覆盖风险。常用策略包括文件所有权、独立工作副本、乐观锁、版本号、原子重命名和合并门。不能让多个 Worker 无锁编辑同一个文件。

## 六、多 Agent 何时真正有价值

### （一）新信息判据

| 协作方式                    | 是否产生新信息 | 价值判断                 |
| --------------------------- | -------------- | ------------------------ |
| 同一模型重读自己的答案      | 否             | 容易把正确答案改错       |
| 多 Agent 对同一文本辩论     | 通常否         | 等预算下未必优于单 Agent |
| Reviewer 运行测试           | 是             | 获得执行反馈             |
| Reviewer 查看渲染截图       | 是             | 获得视觉反馈             |
| 多 Worker 查询不同来源      | 是             | 扩大证据覆盖             |
| 不同专业 Agent 使用不同工具 | 是             | 获得互补信息             |

### （二）预算意识

更多 Worker、步骤和 token 不保证更高质量。Manager 应根据子任务复杂度分配轮数、时间和费用，并要求 Worker 前期探索、后期收敛。若多 Agent 的额外收益不足以覆盖通信、重复探索和汇总成本，单 Agent 更合适。

## 七、对等协作与 Loop 工程

### （一）过早终止问题

Agent 常见的停止错误包括：只完成部分任务就宣称完成；一种方案失败后过早放弃；外部流程尚未闭环却报告成功。根因都是让模型自己批准自己的完成状态。

Loop 工程把控制权放到持久循环：

```text
任务状态决定下一项工作
  → Agent 执行
  → 独立验证器检查环境证据
  → 通过后提交进度
  → 失败则修复或重规划
```

上下文可以刷新，但已经验证的进度、失败证据、剩余任务和预算必须持久化。模型可以提出完成，不能批准自己的完成。

### （二）提议者—审核者

```python
candidate = proposer(task, constraints)
evidence = execute_or_render(candidate)
review = reviewer(candidate, evidence)

while review.veto and budget_remaining():
    candidate = proposer.repair(candidate, review.findings)
    evidence = execute_or_render(candidate)
    review = reviewer(candidate, evidence)

publish(candidate) if review.passed else escalate(review)
```

Proposer 创建候选产物；执行器创建测试、状态或截图证据；Reviewer 持有只读证据并返回结构化缺陷；发布器只有在通过后才提交。Reviewer 不能修改测试、证据采集器或发布门槛，否则独立验证会退化为自我批准。

### （三）辩论、头脑风暴与专家小组

辩论通过固定正反立场暴露风险；头脑风暴强调独立产生多样候选后再组合；专家小组按专业领域补齐约束。要获得真正多样性，需要区分模型、工具、证据、职责或可见上下文，并先独立判断再汇总。仅创建多个同模型实例，容易得到同质答案。

## 八、管理者模式

### （一）Manager 的职责

Manager 负责：

1. 理解整体目标并拆分任务；
2. 选择 Worker 和分配预算；
3. 传递自包含上下文；
4. 维护任务状态和依赖关系；
5. 处理超时、重试、替换和取消；
6. 验证子结果并汇总最终产物。

从 Manager 视角看，Worker 是返回结构化结果的工具。Manager 不应保存每个 Worker 的完整思考过程，而应保存任务描述、状态、证据索引和产物路径。

### （二）为什么规划者是瓶颈

Worker 执行再强，也无法修复错误的任务拆分。最强模型和最完整上下文通常优先分配给 Manager；简单 Worker 可使用更小模型和窄工具集。Manager 仍可能成为单点故障，因此任务图、分配理由和失败状态要可检查。

### （三）顺序与并行

子任务有数据依赖时顺序执行，例如先生成术语表，再翻译章节，最后统一审校。互不依赖时可并行，例如多个 Worker 分别搜索不同网站。

并行系统的完成条件不能是“第一个声称成功”，而应是“第一个通过独立验证的成功结果”。

```python
winner = None

for event in result_events:
    if event.type == "result" and verify(event.artifact):
        if atomic_settle_once(event.task_id, event.worker_id):
            winner = event
            broadcast_cancel(except_worker=event.worker_id)
            break
```

`atomic_settle_once` 由锁或事务保护，保证几乎同时到达的两个成功结果只能结算一次。取消后还要等待 `cancel_ack` 或超时，并清理浏览器、进程、文件句柄和网络连接。

## 九、实验 10-2：书籍翻译 Agent

该实验采用 Manager、Glossary、Translation 和 Proofreading 四种角色：

```text
Manager 创建任务计划
  → Glossary Agent 读取全书并生成术语表
  → 多个 Translation Agent 各读一个章节 + 术语表
  → 译文写入共享文件系统
  → Proofreading Agent 检查术语和格式一致性
  → Manager 根据报告派发定向修订
  → 验收通过后汇总交付
```

Glossary Agent 结束后即可释放；Translation Agent 不需要看到其他章节的完整轨迹；Proofreading Agent 关注一致性而不是重新翻译；Manager 只保存进度和文件索引。这种上下文隔离是多 Agent 翻译的主要价值。

## 十、实验 10-3：电话 Agent 与电脑 Agent

Computer Agent 负责浏览器和整体编排，Phone Agent 负责语音提问、ASR、LLM 与 TTS。两者拥有独立轨迹，通过结构化消息交换字段和值。

真正并行要求“询问下一个字段”和“填写上一个字段”在时间上重叠，而不是两个 Agent 名义上同时存在却串行等待。格式错误需要重问，页面异常要暂停，提交表单与真人通话均需要明确授权。

## 十一、实验 10-4：并行搜索与级联终止

Manager 为多个网站创建独立 Worker，每个 Worker 有自己的浏览器会话和任务 ID。Worker 周期性发送状态；某个 Worker 找到目标后，Manager 先验证结果，再原子结算并取消其他 Worker。

所有 Worker 都失败时，应区分“确认未找到”“网站不可访问”“解析失败”和“超时”，不能把技术失败总结为目标不存在。

## 十二、去中心化协作与 A2A

去中心化系统没有持续在线的 Manager。每个 Agent 根据能力目录寻找协作者，通过消息或任务协议移交工作。它适合跨组织、长期存在和无法由单一主体控制的协作，但在身份、权限、结算、冲突和全局可观测性上更复杂。

A2A（Agent-to-Agent）协议关注 Agent 之间的任务发现、消息和生命周期状态。MCP 主要连接 Agent 与工具；A2A 主要连接 Agent 与 Agent。二者可以同时存在：一个 Agent 通过 A2A 接收任务，再通过 MCP 使用外部工具。

跨组织消息必须包含身份、任务 ID、能力描述、状态、产物引用和认证信息，并明确是否允许再次委派。不能把内部完整轨迹或凭证直接传给外部 Agent。

## 十三、六类失败模式

### （一）共享文件并发冲突

多个 Agent 同时改同一文件会覆盖内容或产生不可合并状态。通过所有权、分支/worktree、锁、事务和合并门处理。

### （二）错误级联

Manager 或早期 Worker 的错误会成为后续 Agent 的前提。关键中间产物应先验证，再允许下游读取。

### （三）同质趋同

相同模型、相同提示和相同证据会产生高度相似的错误。通过独立证据、异构工具和职责分离增加认知多样性。

### （四）互相扯皮

职责、目标和资源所有权不明确时，Agent 会反复转交或互相否定。任务协议应明确单一负责人、优先级、交接条件和人工仲裁点。

### （五）循环失控

Agent 可能不断创建子 Agent 或互相发送任务。运行时必须限制最大深度、实例数、轮数、时间和总费用，并允许全局取消。

### （六）理解债与认知投降

Agent 产出速度超过人的理解速度，会使维护者无法解释系统；长期把判断外包给 Agent，也会削弱审查能力。关键架构、权限和验收逻辑必须保持可读、可追溯并由人理解。

## 十四、Agent 社会与涌现

当 Agent 数量扩大并长期互动时，可能出现三类涌现：社交关系与信息传播、市场中的经济行为、信息不对称下的策略博弈。

生成式 Agent 的基本结构包括记忆流、反思和计划：记忆流保存事件，反思把具体经历抽象成认识，计划结合长期目标与当前环境决定行动。反思只更新当前 Agent 的内部认识，不等于经过跨轨迹验证后的长期能力进化。

狼人杀实验采用代码驱动的法官维护全局状态，只向每个角色 Agent 提供其有权看到的信息。法官创建并持有身份、阶段、存活状态和胜负规则；角色 Agent 只负责推理和选择动作。信息权限必须由确定性代码裁剪，不能要求角色模型“自己假装不知道”。

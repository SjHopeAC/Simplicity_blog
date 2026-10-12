# Chapter 1：Agent 基础知识学习笔记

## 一、前置概念：Agent、Environment 与 Harness

### （一）Environment 是什么

我把 Environment 理解为 Agent 之外、能够被观察或改变的世界，例如：

- 文件系统中的文件和进程；
- 浏览器中的网页和表单；
- 数据库里的业务状态；
- 搜索服务、邮件服务和支付系统；
- 用户以及其他 Agent。

Environment 即使与 Agent 运行在同一台电脑、同一进程，也仍然是环境。判断边界的标准不是物理部署位置，而是谁决定状态如何变化：Harness 负责发出动作，环境负责产生真实结果。

### （二）Model 是什么

Model 是决策内核。它接收 Harness 构造的上下文，然后可能输出：

- 给用户的普通文本；
- 工具调用请求；
- 任务计划或中间判断；
- 是否继续、澄清还是结束。

Model 返回 `tool_calls` 只代表“提出行动”，不代表工具已经执行。真实文件写入、网络请求或数据库修改发生在 Harness 或服务端工具执行器调用工具时。

### （三）Harness 是什么

我把 Harness 理解为 Agent 边界内、模型之外的运行与治理层。它主要负责：

1. 创建和维护消息历史；
2. 向模型暴露工具定义；
3. 调用模型并解析返回值；
4. 校验模型提出的工具参数与权限；
5. 执行或委托执行工具；
6. 把工具结果追加回轨迹；
7. 检查结果、处理失败、限制循环次数；
8. 决定何时结束或转交人工。

对象控制关系是：

```text
开发者
  └── 创建 Harness
        ├── 持有 Model 客户端
        ├── 持有 messages / trajectory
        ├── 持有工具描述与工具实现映射
        ├── 调用 Model
        ├── 校验并执行工具
        └── 读取 Environment 返回的 observation
```

> Harness 可以创建沙盒、重置沙盒或限制沙盒权限，但沙盒中实际变化的文件和进程属于 Environment。

## 二、观察空间、动作空间与工具

### （一）观察空间

观察空间表示环境能够通过哪些渠道把信息交给 Agent，例如文件内容、网页文本、截图、终端输出和用户回复。只有进入上下文的信息模型才能看到；真实世界中存在但未进入上下文的信息，对模型相当于不存在。

### （二）动作空间

动作空间表示 Agent 被允许执行哪些行动，例如搜索网页、读取文件、执行 Python、点击按钮、发送邮件或询问用户。模型即使知道应该做什么，如果 Harness 没有提供相应工具，也只能给出文字建议。

### （三）五类工具

1. **感知工具**：把环境信息带入上下文，如搜索、读文件、查询数据库。
2. **执行工具**：改变环境，如写文件、运行代码、调用业务 API。
3. **协作工具**：委托子 Agent 或请求人工确认。
4. **事件触发工具**：邮件、定时器、Webhook 等外部事件主动启动 Agent。
5. **用户沟通工具**：消息、邮件、语音等向用户传递进度和结果。

工具设计应遵循“最小必要能力”：组合和探索可以使用通用工具；支付、删除、部署等高风险操作应使用参数清晰、权限受限、可审计的专用工具。

## 三、工具调用的语法、协议与项目设计

### （一）Python 语法层

下面的 `tools` 只是 Python 的列表和字典：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]
```

Python 不会因为字典里出现 `name: get_weather` 就自动调用同名函数。这段数据只有被模型 API 按工具 schema 解释时才获得协议含义。

### （二）API 协议层

工具调用通常经过四步：

```text
1. Harness 把工具名称、用途和参数 schema 交给模型
2. 模型返回工具调用请求和参数
3. Harness 或服务端执行器真正调用工具
4. 工具结果通过关联 ID 回到模型上下文
```

简化的结果消息为：

```python
tool_message = {
    "role": "tool",
    "tool_call_id": call.id,
    "content": "{\"temperature\": 28}"
}
```

`tool_call_id` 用于把工具结果与模型先前提出的调用一一对应，特别是在一轮并行请求多个工具时避免结果错配。

### （三）项目设计层

项目通常使用一个工具注册表把“模型看到的工具名”映射到“真实 Python 函数”：

```python
tool_registry = {
    "get_weather": get_weather,
    "calculate": calculate,
}

name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)
result = tool_registry[name](**arguments)  # 真实执行从这里开始
```

这里需要始终区分：

- 工具定义注册：只告诉模型有哪些能力；
- 模型返回工具请求：只决定调用什么；
- `tool_registry[name](...)`：真实副作用开始；
- 追加 `tool` 消息：让模型看到执行结果。

## 四、ReAct：把模型、上下文和工具串成循环

### （一）ReAct 是什么

ReAct = Reasoning + Acting。实际运行包含：

```text
思考 Thought → 行动 Action → 观察 Observation → 再思考 → 最终回答
```

**轨迹（Trajectory）**是该过程累积的结构化历史，包括用户消息、模型回复、工具请求和工具结果。

稳定前缀通常是系统提示词与工具定义；轨迹则随运行不断增长。

### （二）最小 ReAct 代码

```python
messages = [{"role": "user", "content": user_request}]

for _ in range(8):
    response = call_model(messages=messages, tools=tools)
    decision = response.message
    messages.append(decision)

    if not decision.tool_calls:
        return decision.content

    for call in decision.tool_calls:
        checked_call = validate(call)
        result = execute_tool(checked_call)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": serialize(result),
        })

raise RuntimeError("超过最大迭代次数")
```

- `for _ in range(8)` 是 Python 循环和终止上限；
- `messages.append(...)` 是 Python 列表追加；
- `decision.tool_calls` 是模型 API 返回的结构；
- `validate()`、`execute_tool()` 是 Harness 设计；
- `execute_tool()` 内部才会访问 Environment。

### （三）完整控制链与返回点

```text
用户向 main.py 提交任务
  ↓
main.py 创建 Agent/Harness 对象
  ↓
Harness 组合系统提示词、工具定义和轨迹
  ↓
Harness 调用 Model
  ↓
Model 返回 tool_calls，控制权回到 Harness
  ↓
Harness 校验并执行工具，Environment 返回 observation
  ↓
Harness 把 observation 追加为 tool 消息
  ↓
控制权回到循环顶部，再次调用 Model
  ↓
Model 返回无 tool_calls 的最终文本
  ↓
控制权回到 Harness，Harness 输出结果并结束
```

## 五、Harness 五要素

### （一）Context：上下文管理

决定模型每一轮能看到什么，包括系统提示词、工具定义、消息历史、知识库和状态摘要。信息不足会导致模型盲目决策；信息过多又会带来成本和注意力稀释。

### （二）Tools：工具接口

决定 Agent 能观察和执行什么。接口应包含清晰名称、用途、参数、示例、错误语义与权限边界。

### （三）Constrain：约束

在行动发生前限制模型可以做什么，例如路径白名单、金额上限、只读模式和高风险操作确认。默认拒绝、显式开放比默认全权访问更安全。

### （四）Verify：验证

检查执行结果是否满足要求，例如测试、Linter、类型检查、JSON 字段校验和数据库状态检查。安全验证优先读取结构化结果，不依赖模型自己声称“已完成”。

### （五）Correct：纠正

验证失败后重试、换策略、回滚、熔断或转交人工。纠正不是无限重试；需要最大次数、错误分类和稳定退出路径。

完整生产循环为：

```text
Environment 提供观察
  ↓
Harness 构造上下文
  ↓
Model 提议行动
  ↓
Harness 约束并校验行动
  ↓
Environment 执行并返回结果
  ↓
Harness 验证结果
  ├── 正确：进入下一轮或结束
  └── 错误：纠正、回滚、重试或人工接管
```

## 六、工作流与自主 Agent

### （一）工作流

工作流的路径由程序预先规定，适合规则明确、顺序稳定、需要可预测性的任务：

```text
输入 → 分类 → 提取 → 生成 → 验证 → 输出
```

实验 1-4 的“改写 → 生图”就是工作流。模型参与节点内容生成，但不能改变整个图的执行顺序。

### （二）自主 Agent

自主 Agent 由模型根据当前上下文动态选择下一步工具，适合无法提前穷举步骤、需要探索或频繁根据观察调整策略的任务。实验 1-2 的真实联网模式属于自主循环。

## 七、护栏与安全性

### （一）上下文层护栏

控制模型能看到什么，包括来源标记、指令与数据分离、内容审核和提示注入检测。它依赖模型或分类器判断，只能降低风险，无法单独保证安全。

### （二）执行层护栏

控制模型能做什么，包括工具风险评级、最小权限、沙盒、预览和人工确认。高风险操作的复核必须位于可能被注入的模型上下文之外。

### （三）数据层护栏

控制世界最终允许被改成什么样，例如数据库行级权限、约束、受控存储过程和不可伪造的访问身份。即使上层 Agent 被攻击，数据层仍能拒绝越权操作。

### （四）何时转交人工

- 连续失败达到阈值；
- 涉及不可逆操作；
- 涉及高金额或敏感数据；
- 用户目标存在无法安全推断的歧义；
- 验证器无法确认结果正确。

## 七、运行截图

1.跑通环境配置

![image-20261012172359073](/Users/simplyjane/Library/Application Support/typora-user-images/image-20261012172359073.png)

2.运行第一个sample

![image-20261012173053123](/Users/simplyjane/Library/Application Support/typora-user-images/image-20261012173053123.png)

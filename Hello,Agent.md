# Hello,Agent!



## 第一章：初识智能体

在人工智能领域，智能体被定义为任何能够通过**传感器（Sensors）**感知其所处**环境（Environment）**，并**自主**地通过**执行器（Actuators）**采取**行动（Action）**以达成特定目标的实体。

### 1.1历史的智能体

#### 1.1.1智能体演进历史

智能体演进的起点，是结构最简单的反射智能体。他们是由工程师明确设计的“条件-动作”规则构成：例如经典的自动恒温器，若传感器感知的室温高于设定值，则启动制冷系统。

Q1：但是如果环境的当前状态不足以作为决策的全部依据，智能体该怎么办？![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/1-figures/1757242319667-1.png)

A1：为了解决此问题，研究者引入了“状态”的概念，发展出**基于模型的反射智能体（Model-Based Reflex Agent）**。

​	这类智能体拥有一个内部的**世界模型**，用于追踪和理解环境中那些无法被直接感知的方面。

​	【例如，一辆在隧道中行驶的自动驾驶汽车，即便摄像头暂时无法感知到前方的车辆，它的内部模型依然会维持对那辆车存在、速度和预估位置的判断。】

Q2：但是只有理解，没有目标，智能体不知道要怎么推进。

A2：这促进了**基于目标的智能体（Goal-Based Agent）**的发展。与前两者不同，它的行为不再是被动地对环境做出反应，而是主动地、有预见性地选择能够导向某个特定未来状态的行动。

​	【经典的例子是 GPS 导航系统：你的目标是到达公司，智能体会基于地图数据（世界模型），通过搜索算法（如 A*算法）来规划（Planning）出一条最优路径。】

Q3：但是目标并不单一，我们不仅要到达公司，还要时间短，省油，避开拥堵路段。

A3：当多个目标需要权衡时，**基于效用的智能体（Utility-Based Agent）**便随之出现。它为每一个可能的世界状态都赋予一个效用值，这个值代表了满意度的高低。智能体的核心目标不再是简单地达成某个特定状态，而是最大化期望效用。

Q4：但是这些依然是依赖于预设，我们想要通过环境自主学习的智能体。

A4：这便是**学习型智能体（Learning Agent）**的核心思想，而**强化学习（Reinforcement Learning, RL）**是实现这一思想最具代表性的路径。

​	一个学习型智能体包含一个性能元件（即我们前面讨论的各类智能体）和一个学习元件。学习元件通过观察性能元件在环境中的行动所带来的结果来不断修正性能元件的决策策略。

------

#### 1.1.2LLM驱动的智能体出现

以GPT为代表的大语言模型的出现，使得学习型智能体的出现成为了可能。

 LLM 智能体则通过在海量数据上的预训练，获得了隐式的世界模型与强大的涌现能力，使其能够以更灵活、更通用的方式应对复杂任务，行为模式不再是工程师既定的。

我们以“***智能体旅行助手***”为例说明这个差异：

​	在 LLM 智能体出现之前，规划旅行通常意味着用户需要在多个专用应用（如天气、地图、预订网站）之间手动切换，并由用户自己扮演信息整合与决策的角色。

​	而一个 LLM 智能体则能将这个流程整合起来。当接收到“规划一次旅行”这样的指令时，它的工作方式体现了以下几点：

- **规划与推理**：智能体首先会将这个高层级目标分解为一系列逻辑子任务，例如：`[确认出行偏好] -> [查询目的地信息] -> [制定行程草案] -> [预订票务住宿]`。这是一个内在的、由模型驱动的规划过程。
- **工具使用**：在执行规划时，智能体识别到信息缺口，会主动调用外部工具来补全。例如，它会调用天气查询接口获取实时天气，并基于“预报有雨”这一信息，在后续规划中倾向于推荐室内活动。
- **动态修正**：在交互过程中，智能体会将用户的反馈（如“这家酒店超出预算”）视为新的约束，并据此调整后续的行动，重新搜索并推荐符合新要求的选项。整个“**查天气 → 调行程 → 订酒店**”的流程，展现了其根据上下文动态修正自身行为的能力。

总而言之，我们正从开发专用自动化工具转向构建能自主解决问题的系统。核心不再是编写代码，而是引导一个通用的“大脑”去规划、行动和学习。

------

#### 1.1.3智能体的分类

对智能体可分为三类：

（1）**基于内部决策架构的分类**

​	第一种分类维度是依据智能体内部决策架构的复杂程度，涵盖了例如：简单的***反应式***智能体，引入内部模型的***模型式***智能体，再到***基于目标***和***基于效用***的智能体...

（2）**基于时间与反应性的分类**

​	可以从智能体处理决策的时间维度进行分类。这个视角关注智能体是在接收到信息后立即行动，还是会经过深思熟虑的规划再行动。这揭示了智能体设计中一个核心权衡：追求速度的**反应性（Reactivity）**与追求最优解的**规划性（Deliberation）**之间的平衡。

- **反应式智能体 (Reactive Agents)**

  这类智能体对环境刺激做出近乎即时的响应，决策延迟极低。例如简单反应式智能体和基于模型智能体。

- **规划式智能体(Deliberative Agents)**

  与反应式智能体相对，规划式（或称审议式）智能体在行动前会进行复杂的思考和规划。它们不会立即对感知做出反应，而是会先利用其内部的世界模型，探索未来的可能性，评估不同行动的后果，以期找到一条能够达成目标的最佳路径 。例如**基于目标**和**基于效用**的智能体是典型的规划式智能体。

- **混合式智能体(Hybrid Agents)**

  现实世界的复杂任务，往往既需要即时反应，也需要长远规划。例如，我们之前提到的智能旅行助手，既要能根据用户的即时反馈（如“这家酒店太贵了”）调整推荐（反应性），又要能规划出为期数天的完整旅行方案（规划性）。因此，混合式智能体旨在结合两者的优点，实现反应与规划的平衡。

**（3）基于知识表示的分类**

- **符号主义 AI（Symbolic AI）**

  其主要优势在于透明和可解释。由于推理步骤明确，其决策过程可以被完整追溯。

- **亚符号主义 AI（连接主义）（Sub-symbolic AI）**

  它能够轻松处理图像、声音等非结构化数据，这在符号主义 AI 看来是极其困难的任务。然而，这种强大的直觉能力也伴随着不透明性。亚符号主义系统通常被视为一个**黑箱**。

- **神经符号主义 AI（Neuro-Symbolic AI）**

  它的目标，是融合两大范式的优点，创造出一个既能像神经网络一样从数据中学习，又能像符号系统一样进行逻辑推理的混合智能体。它试图弥合感知与认知、直觉与理性之间的鸿沟。

​	

​	**大语言模型驱动的智能体**是神经符号主义的一个极佳实践范例。

​	其内核是一个巨大的神经网络，使其具备模式识别和语言生成能力。然而，当它工作时，它会生成一系列结构化的中间步骤，如思想、计划或 API 调用，这些都是明确的、可操作的符号。通过这种方式，它实现了感知与认知、直觉与理性的初步融合。

------

### 1.2智能体的构成与运行原理

#### 1.2.1任务环境

在人工智能领域，通常使用**PEAS 模型**来精确描述一个任务环境，即分析其**性能度量(Performance)、环境(Environment)、执行器(Actuators)和传感器(Sensors)** 。

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/1-figures/1757242319667-6.png)

1. 首先，环境通常是**部分可观察的**。例如，旅行助手在查询航班时，无法一次性获取所有航空公司的全部实时座位信息。它只能通过调用航班预订 API，看到该 API 返回的部分数据。这就要求智能体必须具备记忆（记住已查询过的航线）和探索（尝试不同的查询日期）的能力。
2. 其次，行动的结果也并非总是确定的。根据结果的可预测性，环境可分为**确定性**和**随机性**。旅行助手的任务环境就是典型的随机性环境。当它搜索票价时，两次相邻的调用返回的机票价格和余票数量都可能不同，这就要求智能体必须具备处理不确定性、监控变化并及时决策的能力。
3. 此外，环境中还可能存在其他行动者，从而形成**多智能体(Multi-agent)** 环境。它们的行动（例如，订走最后一张特价票）会直接改变旅行助手所处环境的状态，这对智能体的快速响应和策略选择提出了更高要求。
4. 最后，几乎所有任务都发生在**序贯**且**动态**的环境中。“序贯”意味着当前动作会影响未来；而“动态”则意味着环境自身可能在智能体决策时发生变化。这就要求智能体的“感知-思考-行动-观察”循环必须能够快速、灵活地适应持续变化的世界。

#### 1.2.2智能体的运行机制

智能体并非一次性完成任务，而是通过一个持续的循环与环境进行交互，这个核心机制被称为 **智能体循环 (Agent Loop)**。

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/1-figures/1757242319667-5.png)

这个循环主要包含以下几个相互关联的阶段：

1. **感知 (Perception)**：这是循环的起点。智能体通过其传感器（例如，API 的监听端口、用户输入接口）接收来自环境的输入信息。这些信息，即是**观察 (Observation)**，既可以是用户的初始指令，也可以是上一步行动所导致的环境状态变化反馈。
2. **思考 (Thought)**：智能体的***核心决策阶段***。对于 LLM 智能体而言，这通常是由**大语言模型**驱动的内部推理过程。“思考”阶段可进一步细分为两个关键环节：
   - **规划 (Planning)**：智能体基于观察和记忆，更新对任务和环境的理解，并制定或调整一个行动计划。这可能涉及将复杂目标分解为一系列更具体的子任务。
   - **工具选择 (Tool Selection)**：根据当前计划，智能体从其可用的工具库中，选择最适合执行下一步骤的工具，并确定调用该工具所需的具体参数。
3. **行动 (Action)**：决策完成后，智能体通过其执行器（Actuators）执行具体的行动。这通常表现为调用一个选定的工具（如代码解释器、搜索引擎 API），从而对环境施加影响，意图改变环境的状态。

行动不是循环的终点。智能体的行动会引起**环境 (Environment)** 的**状态变化 (State Change)**，环境随即会产生一个新的**观察 (Observation)** 作为结果反馈。这个新的观察又会在下一轮循环中被智能体的感知系统捕获，形成一个持续的“感知-思考-行动-观察”的闭环。智能体正是通过不断重复这一循环，逐步推进任务，从初始状态向目标状态演进。

#### 1.2.3智能体的交互协议

在工程实践中，为了让 LLM 能够有效驱动这个循环，我们需要一套明确的**交互协议 (Interaction Protocol)** 来规范其与环境之间的信息交换。，这一协议体现在对智能体每一次输出的结构化定义上。智能体的输出是一段遵循特定格式的文本，其中明确地展示了其内部的推理过程与最终决策。

这里的交互范式是`Thought-Action-Observation` 交互范式。

这个结构通常包含两个核心部分：

- **Thought (思考)**：它以自然语言形式阐述了智能体如何分析当前情境、回顾上一步的观察结果、进行自我反思与问题分解，并最终规划出下一步的具体行动。
- **Action (行动)**：这是智能体基于思考后，决定对环境施加的具体操作，通常以函数调用的形式表示。

```Bash
Thought: 用户想知道北京的天气。我需要调用天气查询工具。
Action: get_weather("北京")
Observation: 北京当前天气为晴，气温25摄氏度，微风。
```

1. 这里的`Action`字段构成了对外部世界的指令。一个外部的**解析器 (Parser)** 会捕捉到这个指令，并调用相应的`get_weather`函数。
2. 行动执行后，环境会返回一个结果。感知系统的一个重要职责就是扮演传感器的角色：将这个原始JSON输出处理并封装成一段简洁、清晰的自然语言文本，即观察。
3. 这段`Observation`文本会被反馈给智能体，作为下一轮循环的主要输入信息，供其进行新一轮的`Thought`和`Action`。

------

### 1.3实现智能体

前置知识：

`requests`是 Python 社区中最流行、最易用的选择。

`tavily`是一个强大的 AI 搜索 API 客户端，是专供AI用于获取最新实时的网络搜索结果，且返回的JSON格式有利于AI进行获取解析。

`openai`是 OpenAI 官方提供的 Python SDK，用于调用 GPT 等大语言模型服务。

需准备：

（1）指令模板

```
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""
```

（2）执行任务所用的工具（即各类功能的API接口）

工具 1：查询真实天气

我们将使用免费的天气查询服务 `wttr.in`，它能以 JSON 格式返回指定城市的天气数据。

工具 2：搜索并推荐旅游景点

我们将定义一个新工具 `search_attraction`，它会根据城市和天气状况，互联网上搜索合适的景点。

最后，我们将所有工具函数放入一个字典，供主循环调用：

```python
# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
```

（3）接入LLM大语言模型

```python
from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"
```

（4）实现循环主体

```python
import re

# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL_ID = "YOUR_MODEL_ID"
TAVILY_API_KEY="YOUR_Tavily_KEY"
os.environ['TAVILY_API_KEY'] = "YOUR_TAVILY_API_KEY"

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
    
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
```

输出如下：

```bash
用户输入: 你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。
========================================
--- 循环 1 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 首先需要获取北京今天的天气情况，之后再根据天气情况来推荐旅游景点。
Action: get_weather(city="北京")

Observation: 北京当前天气:Sunny，气温26摄氏度
========================================      
--- 循环 2 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 现在已经知道了北京今天的天气是晴朗且温度适中，接下来可以基于这个信息来推荐一个适合的旅游景点了。
Action: get_attraction(city="北京", weather="Sunny")

Observation: 北京在晴天最值得去的旅游景点是颐和园，因其美丽的湖景和古建筑。另一个推荐是长城，因其壮观的景观和历史意义。
========================================
--- 循环 3 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 已经获得了两个适合晴天游览的景点建议，现在可以根据这些信息给用户提供满意的答复。
Action: Finish[今天北京的天气是晴朗的，气温26摄氏度，非常适合外出游玩。我推荐您去颐和园欣赏美丽的湖景和古建筑，或者前往长城体验其壮观的景观和深厚的历史意义。希望您有一个愉快的旅行！]

任务完成，最终答案: 今天北京的天气是晴朗的，气温26摄氏度，非常适合外出游玩。我推荐您去颐和园欣赏美丽的湖景和古建筑，或者前往长城体验其壮观的景观和深厚的历史意义。希望您有一个愉快的旅行！
```

这个简单的旅行助手案例，集中演示了基于`Thought-Action-Observation`范式的智能体所具备的四项基本能力：任务分解、工具调用、上下文理解和结果合成。正是通过这个循环的不断迭代，智能体才得以将一个模糊的用户意图，转化为一系列具体、可执行的步骤，并最终达成目标。

### 1.4智能体应用的协作模式

基于智能体在任务中的角色和自主性程度，其协作模式主要分为两种：

1. 一种是作为高效工具，深度融入我们的工作流；
2. 一种则是作为自主的协作者，与其他智能体协作完成复杂目标。

- 作为开发者工具的智能体。

  在这种模式下，智能体被深度集成到开发者的工作流中，作为一种强大的辅助工具。它增强而非取代开发者的角色，通过自动化处理繁琐、重复的任务，让开发者能更专注于创造性的核心工作。比如Trae等...

- 作为自主协作者的智能体

  在这种模式下，我们不再是手把手地指导 AI 完成每一步，而是将一个高层级的目标委托给它。智能体会像一个真正的项目成员一样，独立地进行规划、推理、执行和反思，直到最终交付成果。

#### 1.4.2 Workflow（工作流）和Agent的差异

简单来说，**Workflow 是让 AI 按部就班地执行指令，而 Agent 则是赋予 AI 自由度去自主达成目标。**

- 工作流是一种传统的自动化范式，其核心是**对一系列任务或步骤进行预先定义的、结构化的编排**。它本质上是一个精确的、静态的流程图，规定了在何种条件下、以何种顺序执行哪些操作。
- 基于大型语言模型的智能体是一个**具备自主性的、以目标为导向的系统**。它不仅仅是执行预设指令，而是能够在一定程度上理解环境、进行推理、制定计划，并动态地采取行动以达成最终目标。



## 第二章：智能体发展史

AI智能体演进历史

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/2-figures/1757246501849-00.png)

**每一个新范式的出现，都是为了解决上一代范式的核心“痛点”或根本局限。** 而新的解决方案在带来能力飞跃的同时，也引入了新的、在当时难以克服的“局限”，而这又为下一代范式的诞生埋下了伏笔。理解这一“问题驱动”的迭代历程，能帮助我们更深刻地把握现代智能体技术选型背后的深层原因与历史必然性。



### 2.1基于符号与逻辑的早期智能体

人工智能领域的早期探索，深受数理逻辑和计算机科学基本原理的影响。在那个时代，研究者们普遍持有一种信念：人类的智能，尤其是逻辑推理能力，可以被形式化的符号体系所捕捉和复现。这一核心思想催生了人工智能的第一个重要范式——符号主义（Symbolicism），也被称为“逻辑AI”或“传统AI”。

#### 2.1.1物理符号系统假说

物理符号假说(PSSH)宣称:智能的本质,就是符号的计算与处理.

这个假说具有深远的影响。它将对人类心智这一模糊、复杂的哲学问题的研究，转化为了一个可以在计算机上进行工程化实现的具体问题。

#### 2.1.2专家系统

​	**专家系统（Expert System）**是符号主义时代最重要、最成功的应用成果。专家系统的核心目标，是模拟人类专家在特定领域内解决问题的能力。它通过将专家的知识和经验编码成计算机程序，使其能够在面对相似问题时，给出媲美甚至超越人类专家的结论或建议。

专家系统的“智能”主要源于其两大核心组件：**知识库**和**推理机**。

- **知识库（Knowledge Base）**：这是专家系统的知识存储中心，用于存放领域专家的知识和经验。(常用的一种知识表示方法是**产生式规则（Production Rules）**，即一系列“IF-THEN”形式的条件语句)

- **推理机（Inference Engine）**：推理机是专家系统的核心计算引擎。它是一个通用的程序，其任务是根据用户提供的事实，在知识库中寻找并应用相关的规则，从而推导出新的结论。

  ​	推理机的工作方式主要有两种：

  - **正向链（Forward Chaining）**：从已知事实出发，不断匹配规则的IF部分，触发THEN部分的结论，并将新结论加入事实库，直到最终推导出目标或无新规则可匹配。
  - **反向链（Backward Chaining）**：从一个假设的目标（比如“病人是否患有肺炎”）出发，寻找能够推导出该目标的规则，然后将该规则的IF部分作为新的子目标，如此递归下去，直到所有子目标都能被已知事实所证明。

#### 2.1.3符号主义的瓶颈

​	符号主义AI在从“微观世界”走向开放、复杂的现实世界时，遇到了其方法论固有的根本性难题。这些难题主要可归结为两大类：

**（1）常识知识与知识获取瓶颈**

符号主义智能体的“智能”完全依赖于其知识库的质量和完备性。然而，如何构建一个能够支撑真实世界交互的知识库，是一项极其艰巨的任务，主要体现在两个方面：

- **知识获取瓶颈（Knowledge Acquisition Bottleneck）**：

  专家系统的知识需要由人类专家和知识工程师通过繁琐的访谈、提炼和编码过程来构建。这个过程成本高昂、耗时漫长，且难以规模化。

  更重要的是，人类专家的许多知识是内隐的、直觉性的，很难被清晰地表达为“IF-THEN”规则。试图将整个世界的知识都进行手工符号化，被认为是一项几乎不可能完成的任务。

- **常识问题（Common-sense Problem）**：人类行为依赖于庞大的常识背景（例如，“水是湿的”、“绳子可以拉不能推”），但符号系统除非被明确编码，否则对此一无所知。

  **（2）框架问题与系统脆弱性**

  除了知识层面的挑战，符号主义在处理动态变化的世界时也遇到了逻辑上的困境。

  - **框架问题（Frame Problem）**：在一个动态世界中，智能体执行一个动作后，如何高效判断哪些事物未发生改变是一个逻辑难题。为每个动作显式地声明所有不变的状态，在计算上是不可行的，而人类却能毫不费力地忽略不相关的变化。
  - **系统脆弱性（Brittleness）**：符号系统完全依赖预设规则，导致其行为非常“脆弱”。一旦遇到规则之外的任何微小变化或新情况，系统便可能完全失灵.

### 2.2基于规则的聊天机器人

​	本节我们将通过一个具体的编程实践，来直观地感受基于规则的系统是如何工作的。我们将复现人工智能历史上一个极具影响力的早期聊天机器人——ELIZA。

#### 2.2.1ELIZA的设计思想

它从不正面回答问题或提供信息，而是通过识别用户输入中的关键词，然后应用一套预设的转换规则，将用户的陈述转化为一个开放式的提问。例如，当用户说“我为我的男朋友感到难过”时，ELIZA可能会识别出关键词“我为……感到难过”，并应用规则生成回应：“你为什么会为你的男朋友感到难过？”

#### 2.2.2模式匹配与文本转换

ELIZA的算法流程基于**模式匹配（Pattern Matching）与文本替换（Text Substitution）**，可被清晰地分解为以下四个步骤：

1. **关键词识别与排序：**规则库为每个关键词（如 `mother`, `dreamed`, `depressed`）设定一个优先级。当输入包含多个关键词时，程序会选择优先级最高的关键词所对应的规则进行处理。

2. **分解规则**：

   找到关键词后，程序使用带通配符（*）的分解规则来捕获句子的其余部分。

   1. **规则示例**： `* my *`
   2. **用户输入**： `"My mother is afraid of me"`
   3. **捕获结果**： `["", "mother is afraid of me"]`

3. **重组规则**：

   程序从与分解规则关联的一组重组规则中，选择一条来生成回应（通常随机选择以增加多样性），并可选择性地使用上一步捕获的内容。

   1. **规则示例**： `"Tell me more about your family."`
   2. **生成输出**： `"Tell me more about your family."`

4. **代词转换：**在重组前，程序会进行简单的代词转换（如 `I` → `you`, `my` → `your`），以维持对话的连贯性。

#### 2.2.3 核心逻辑

迷你版ELIZA的实现代码,用于展示核心工作机制:

```python
import re
import random

# 定义规则库:模式(正则表达式) -> 响应模板列表
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    r'.*': [
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# 定义代词转换规则
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine",
    "mine": "yours"
}

def swap_pronouns(phrase):
    """
    对输入短语中的代词进行第一/第二人称转换
    """
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)

def respond(user_input):
    """
    根据规则库生成响应
    """
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 捕获匹配到的部分
            captured_group = match.group(1) if match.groups() else ''
            # 进行代词转换
            swapped_group = swap_pronouns(captured_group)
            # 从模板中随机选择一个并格式化
            response = random.choice(responses).format(swapped_group)
            return response
    # 如果没有匹配任何特定规则，使用最后的通配符规则
    return random.choice(rules[r'.*'])

# 主聊天循环
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")
        
>>>
Therapist: Hello! How can I help you today?
You: I am feeling sad today.
Therapist: How long have you been feeling sad today?
You: I need some help with my project.
Therapist: Are you sure you need some help with your project?
You: My mother is not happy with my work.
Therapist: Tell me more about your mother.
You: quit
Therapist: Goodbye. It was nice talking to you.

```

从个人实验中很容易看出其中的缺点:

- **缺乏语义理解**：系统不理解词义。例如，面对“I am **not** happy”的输入，它仍会机械地匹配 `I am (.*)` 规则并生成语义不通的回应，因为它无法理解否定词“not”的作用。
- **无上下文记忆**：系统是**无状态的（Stateless）**，每次回应仅基于当前单句输入，无法进行连贯的多轮对话。
- **规则的扩展性问题**：尝试增加更多规则会导致规则库的规模爆炸式增长，规则间的冲突与优先级管理将变得极其复杂，最终导致系统难以维护。
- 而且回复均为疑问句,其实并没有很好的解决用户的诉求

### 2.3心智社会

​	符号主义的探索和ELIZA的实践，共同指向了一个问题：通过预设规则构建的、单一的、集中的推理引擎，似乎难以通向真正的智能。无论规则库多么庞大，系统在面对真实世界的模糊性、复杂性和无穷变化时，总是显得僵化而脆弱。

​	20世纪70至80年代，符号主义的局限性日益明显。专家系统虽然在高度垂直的领域取得了成功，但它们无法拥有儿童般的常识；SHRDLU虽然能在一个封闭的积木世界中表现出色，但它无法理解这个世界之外的任何事情；ELIZA虽然能模仿对话，但它对对话内容本身一无所知。这些系统都遵循着一种自上而下（Top-down）的设计思路：一个全知全能的中央处理器，根据一套统一的逻辑规则来处理信息和做出决策,没有自己的主观思想。

​	思考者开始在想:什么是理解,什么是常识,智能体应该如何构建?

​	正是基于这样的反思，明斯基提出了一个颠覆性的构想，他不再将心智视为一个金字塔式的层级结构，而是将其看作一个扁平化的、充满了互动与协作的“社会”。

#### 2.3.1协作的智能体

​	在明斯基的理论框架中，这里的智能体指的是一个极其简单的、专门化的心智过程，它自身是“无心”的。例如，一个负责识别线条的`LINE-FINDER`智能体，或一个负责抓握的`GRASP`智能体。

​	这些简单的智能体被组织起来，形成功能更强大的**机构（Agency）**。一个机构是一组协同工作的智能体，旨在完成一个更复杂的任务。

​	**涌现（Emergence）**是理解心智社会理论的关键:复杂的、有目的性的智能行为，并非由某个高级智能体预先规划，而是从大量简单的底层智能体之间的局部交互中自发产生的。

​	让我们以经典的“搭建积木塔”任务为例，来说明这一过程，如图2.6所示。当一个高层目标（如“我要搭一个塔”）出现时，它会激活一个名为`BUILD-TOWER`的高层机构。

1. `BUILD-TOWER`机构并不知道如何执行具体的物理动作，它的唯一作用是激活它的下属机构，比如`BUILDER`。
2. `BUILDER`机构同样很简单，它可能只包含一个循环逻辑：只要塔还没搭完，就激活`ADD-BLOCK`机构。
3. `ADD-BLOCK`机构则负责协调更具体的子任务，它会依次激活`FIND-BLOCK`、`GET-BLOCK`和`PUT-ON-TOP`这三个子机构。
4. 每一个子机构又由更底层的智能体构成。例如，`GET-BLOCK`机构会激活视觉系统中的`SEE-SHAPE`智能体、运动系统中的`REACH`和`GRASP`智能体。

在这个过程中，没有任何一个智能体或机构拥有整个任务的全局规划。当这个由无数“无心”的智能体组成的社会，通过简单的激活和抑制规则相互作用时，一个看似高度智能的行为，搭建积木塔，就自然而然地涌现了出来。

#### 2.3.2对于现代多智能体系统的启发

​	心智社会理论最深远的影响，在于它为**分布式人工智能（Distributed Artificial Intelligence, DAI）**以及后来的**多智能体系统（Multi-Agent System, MAS）**提供了重要的概念基础。

**如果一个心智内部的智能，是通过大量简单智能体的协作而涌现的，那么，在多个独立的、物理上分离的计算实体（计算机、机器人）之间，是否也能通过协作涌现出更强大的“群体智能”？**

心智社会在以下几个方面直接启发了多智能体系统的研究：

- **去中心化控制（Decentralized Control）**：理论的核心在于不存在中央控制器。
- **涌现式计算（Emergent Computation）**：复杂问题的解决方案可以从简单的局部交互规则中自发产生。这启发了MAS中大量基于涌现思想的算法，如蚁群算法、粒子群优化等，用于解决复杂的优化和搜索问题。
- **智能体的社会性（Agent Sociality）**：明斯基的理论强调了智能体之间的交互（激活、抑制）。即系统地研究智能体之间的通信语言（如ACL）、交互协议（如契约网）、协商策略、信任模型乃至组织结构，从而构建起真正的计算社会。

### 2.4范式的演进与现代智能体

​	“心智社会”理论与符号主义在应对真实世界复杂性时暴露的挑战共同指向了一个问题：如果智能无法被完全设计，那么它是否可以被学习出来？

​	这一设问开启了人工智能的“学习”时代。其核心目标不再是手动编码知识，而是构建能从经验和数据中自动获取知识与能力的系统。

#### 2.4.1联结主义

​	与符号主义自上而下、依赖明确逻辑规则的设计哲学不同，联结主义是一种自下而上的方法，其灵感来源于对生物大脑神经网络结构的模仿。它的核心思想可以概括为以下几点：

1. **知识的分布式表示**：知识并非以明确的符号或规则形式存储在某个知识库中，而是以连接权重的形式，分布式地存储在大量简单的处理单元（即人工神经元）的连接之间。整个网络的连接模式本身就构成了知识。
2. **简单的处理单元**：每个神经元只执行非常简单的计算，如接收来自其他神经元的加权输入，通过一个激活函数进行处理，然后将结果输出给下一个神经元。
3. **通过学习调整权重**：系统的智能并非来自于设计者预先编写的复杂程序，而是来自于“学习”过程。系统通过接触大量样本，根据某种学习算法（如反向传播算法）自动、迭代地调整神经元之间的连接权重，从而使得整个网络的输出逐渐接近期望的目标。

在这种范式下，智能体不再是一个被动执行规则的逻辑推理机，而是一个能够通过经验自我优化的适应性系统。

联结主义的兴起，特别是深度学习在21世纪的成功，为智能体赋予了强大的感知和模式识别能力，使其能够直接从原始数据（如图像、声音、文本）中理解世界，这是符号主义时代难以想象的。然而，如何让智能体学会在与环境的动态交互中做出最优的序贯决策，则需要另一种学习范式的补充。

#### 2.4.2基于强化学习的智能体

​	联结主义主要解决了感知问题（例如，“这张图片里有什么？”），但智能体更核心的任务是进行决策（例如，“在这种情况下，我应该做什么？”）。

​	**强化学习（Reinforcement Learning, RL）**正是专注于解决序贯决策问题的学习范式。它并非直接从标注好的静态数据集中学习，而是通过智能体与环境的直接交互，在“试错”中学习如何最大化其长期收益。(正收益给予正向奖励,负收益给予负向奖励)

​	这种通过与环境互动、根据反馈信号来优化自身行为的学习机制，就是强化学习的核心框架。

​	强化学习的框架可以用几个核心要素来描述：

- **智能体（Agent）**：学习者和决策者。在AlphaGo的例子中，就是其决策程序。
- **环境（Environment）**：智能体外部的一切，是智能体与之交互的对象。对AlphaGo而言，就是围棋的规则和对手。
- **状态（State, S）**：对环境在某一时刻的特定描述，是智能体做出决策的依据。例如，棋盘上所有棋子的当前位置。
- **行动（Action, A）**：智能体根据当前状态所能采取的操作。例如，在棋盘的某个合法位置上落下一子。
- **奖励（Reward, R）**：环境在智能体执行一个行动后，反馈给智能体的一个标量信号，用于评价该行动在特定状态下的好坏。例如，在一局棋结束后，胜利获得+1的奖励，失败获得-1的奖励。

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/2-figures/1757246501849-6.png)

这个循环的具体步骤如下：

1. 在时间步t，智能体观察到环境的当前状态St。
2. 基于状态 *St*，智能体根据其内部的**策略（Policy, π）**选择一个行动 At 并执行它。策略本质上是一个从状态到行动的映射，定义了智能体的行为方式。
3. 环境接收到行动 At后，会转移到一个新的状态 St+1。
4. 同时，环境会反馈给智能体一个即时奖励 Rt+1。
5. 智能体利用这个反馈（新状态 St+1 和奖励 Rt+1）来更新和优化其内部策略，以便在未来做出更好的决策。这个更新过程就是学习。

智能体的学习目标，并非最大化某一个时间步的即时奖励，而是最大化从当前时刻开始到未来的**累积奖励（Cumulative Reward）**，也称为**回报（Return）**。这意味着智能体需要具备“远见”，有时为了获得未来更大的奖励，需要牺牲当前的即时奖励。



#### 2.4.3强化学习的预训练

强化学习赋予了智能体从交互中学习决策策略的能力，但这通常需要海量的、针对特定任务的交互数据，导致智能体在学习之初缺乏先验知识，需要从零开始构建对任务的理解。

如何让智能体在开始学习具体任务前，就先具备对世界的广泛理解？这一问题的解决方案，最终在**自然语言处理（Natural Language Processing, NLP）**领域中浮现，其核心便是基于大规模数据的**预训练（Pre-training）**。

**从特定任务到通用模型**

​	在预训练范式出现之前，传统的自然语言处理模型通常是为单一特定任务（如情感分析、机器翻译）在专门标注的中小规模数据集上从零开始独立训练的。这样暴露出各类问题。

​	**预训练与微调**（Pre-training, Fine-tuning）范式的提出彻底改变了这一现状。其核心思想分为两步：

1. **预训练阶段**：首先在一个包含互联网级别海量文本数据的通用语料库上，通过**自监督学习（Self-supervised Learning）**的方式训练一个超大规模的神经网络模型。这个阶段的目标不是完成任何特定任务，而是学习语言本身内在的规律、语法结构、事实知识以及上下文逻辑。最常见的目标是“预测下一个词”。
2. **微调阶段**：完成预训练后，这个模型就已经学习到了和数据集有关的丰富知识。之后，针对特定的下游任务，只需使用少量该任务的标注数据对模型进行微调，即可让模型适应对应任务。

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/2-figures/1757246501849-7.png)

通过在数万亿级别的文本上进行预训练，大型语言模型的神经网络权重实际上已经构建了一个关于世界知识的、高度压缩的隐式模型。当模型的规模（参数量、数据量、计算量）跨越某个阈值后，它们开始展现出未被直接训练的、预料之外的**涌现能力（Emergent Abilities）**，例如：

- **上下文学习（In-context Learning）**：无需调整模型权重，仅在输入中提供**几个示例（Few-shot）**甚至**零个示例（Zero-shot）**，模型就能理解并完成新的任务。
- **思维链（Chain-of-Thought）推理**：通过引导模型在回答复杂问题前，先输出一步步的推理过程，可以显著提升其在逻辑、算术和常识推理任务上的准确性。

至此，智能体发展的历史长河中，几大关键的技术拼图已经悉数登场：符号主义提供了**逻辑推理的框架**，联结主义和强化学习提供了**学习与决策的能力**，而大型语言模型则提供了前所未有的、通过预训练获得的**世界知识和通用推理能力**。

#### 2.4.4基于大语言模型的智能体

随着大型语言模型技术的飞速发展，以LLM为核心的智能体已成为人工智能领域的新范式。

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/2-figures/1757246501849-8.png)

该流程遵循图所示的架构，具体步骤如下：

1. **感知 (Perception)** ：流程始于**感知模块 (Perception Module)**。它通过传感器从**外部环境 (Environment)** 接收原始输入，形成**观察 (Observation)**。这些观察信息（如用户指令、API返回的数据或环境状态的变化）是智能体决策的起点，处理后将被传递给思考阶段。
2. **思考 (Thought)**：这是智能体的认知核心，对应图中的规划模块 (Planning Module)和大型语言模型 (LLM)的协同工作。
   - **规划与分解**：首先，规划模块接收观察信息，进行高级策略制定。它将宏观目标分解为更具体、可执行的步骤。
   - **推理与决策**：随后，作为中枢的**LLM** 接收来自规划模块的指令，并与**记忆模块 (Memory)** 交互以整合历史信息。LLM进行深度推理，最终决策出下一步要执行的具体操作，这通常表现为一个**工具调用 (Tool Call)**。
3. **行动 (Action)** ：决策完成后，便进入行动阶段，由**执行模块 (Execution Module)** 负责。LLM生成的工具调用指令被发送到执行模块。该模块解析指令，从**工具箱 (Tool Use)** 中选择并调用合适的工具来与环境交互或执行任务。这个与环境的实际交互就是智能体的**行动 (Action)**。
4. **观察 (Observation)** 与循环 ：行动会改变环境的状态，并产生结果。
   - 工具执行后会返回一个**工具结果 (Tool Result)** 给LLM，这构成了对行动效果的直接反馈。同时，智能体的行动产生了一个全新的**环境状态**。
   - 这个“工具结果”和“新的环境状态”共同构成了一轮全新的**观察 (Observation)**。这个新的观察会被感知模块再次捕获，同时LLM会根据行动结果**更新记忆 (Memory Update)**，从而启动下一轮“感知-思考-行动”的循环。

#### 2.4.5智能体发展关键节点概览

主要有三大思潮主导着不同时期的研究范式：

1. **符号主义 (Symbolism)** ：以**赫伯特·西蒙 (Herbert A. Simon)** 、**明斯基 (Marvin Minsky)** 等先驱为代表，认为智能的核心在于对符号的操作与逻辑推理。
2. **联结主义 (Connectionism)** ：其灵感源于对大脑神经网络的模拟。尽管早期发展受限，但在**杰弗里·辛顿 (Geoffrey Hinton)** 等研究者的推动下，反向传播算法为神经网络的复苏奠定了基础。最终，随着深度学习时代的到来，这一思想通过卷积神经网络、Transformer等模型成为当前的主流。
3. **行为主义 (Behaviorism)** ：强调智能体通过与环境的互动和试错来学习最优策略，其现代化身为强化学习 。

进入21世纪20年代，这些思想流派以前所未有的方式深度融合。以GPT系列为代表的大语言模型，其本身是联结主义的产物，却成为了执行符号推理、进行工具调用和规划决策的核心“大脑”，形成了神经-符号结合的现代智能体架构。

<img src="https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/2-figures/1757246501849-9.png" alt="图片描述" style="zoom:150%;" />

当代AI agent技术栈预览

![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/2-figures/1757246501849-10.png)



## 第三章：大语言模型基础

### 3.1语言模型与Transformer架构

#### 3.1.1从N—gram到RNN

**语言模型 (Language Model, LM)** 是自然语言处理的核心，其根本任务是计算一个词序列（即一个句子）出现的概率。

**（1）统计语言模型与N-gram思想**

在深度学习兴起之前，统计方法是语言模型的主流。其核心思想是，一个句子出现的概率，等于该句子中每个词出现的条件概率的连乘。对于一个由词 w*1,*w*2,⋯,*wm构成的句子 S，其概率 P(S) 可以表示为：

![image-20260213205156522](D:\notebook\image-20260213205156522.png)

这个公式被称为概率的链式法则。然而直接计算这个公式几乎不可能，因为有的条件概率太难估计，而且词序列或许根本没在训练数据出现过。

为了解决这个问题，研究者引入了**马尔可夫假设 (Markov Assumption)** 。其核心思想是：我们不必回溯一个词的全部历史，可以近似地认为，一个词的出现概率只与它前面有限的 n−1*n*−1 个词有关。基于这个假设建立的语言模型，我们称之为 **N-gram模型**。这里的 "N" 代表我们考虑的上下文窗口大小。

- **Bigram (当 N=2 时)** ：这是最简单的情况，我们假设一个词的出现只与它前面的一个词有关。因此，链式法则中复杂的条件概率 P就可以被近似为更容易计算的形式：

  ![屏幕截图 2026-02-13 205552](D:\notebook\屏幕截图 2026-02-13 205552.png)

- **Trigram (当 N=3 时)** ：类似地，我们假设一个词的出现只与它前面的两个词有关：

![屏幕截图 2026-02-13 205653](D:\notebook\屏幕截图 2026-02-13 205653.png)

这些概率可以通过在大型语料库中进行**最大似然估计(Maximum Likelihood Estimation,MLE)** 来计算。这个术语听起来很复杂，但其思想非常直观：最可能出现的，就是我们在数据中看到次数最多的。例如，对于 Bigram 模型，我们想计算在词 w i−1 出现后，下一个词是 wi 的概率 P(wi∣wi−1)。根据最大似然估计，这个概率可以通过简单的计数来估算：

![4](D:\notebook\4.png)

这里的 `Count()` 函数就代表“计数”：

- Count(wi−1,wi)：表示词对 (wi−1,wi) 在语料库中连续出现的总次数。
- Count(wi−1)：表示单个词 wi−1在语料库中出现的总次数。

公式的含义就是：我们用“词对 Count(wi−1,wi)出现的次数”除以“词 Count(wi−1)出现的总次数”，来作为 P(wi∣wi−1) 的一个近似估计。

为了让这个过程更具体，我们来手动进行一次计算。假设我们拥有一个仅包含以下两句话的迷你语料库：`datawhale agent learns`, `datawhale agent works`。我们的目标是：使用 Bigram (N=2) 模型，估算句子 `datawhale agent learns` 出现的概率。

```
>>>
第一步: P(datawhale) = 2/6 = 0.333
第二步: P(agent|datawhale) = 2/2 = 1.000
第三步: P(learns|agent) = 1/2 = 0.500
最后: P('datawhale agent learns') ≈ 0.333 * 1.000 * 0.500 = 0.167
```

N-gram 模型虽然简单有效，但有两个致命缺陷：

1. **数据稀疏性 (Sparsity)** ：如果一个词序列从未在语料库中出现，其概率估计就为 0。
2. **泛化能力差：**模型无法理解词与词之间的语义相似性。例如，即使模型在语料库中见过很多次 `agent learns`，它也无法将这个知识泛化到语义相似的词上。当我们计算 `robot learns` 的概率时，如果 `robot` 这个词从未出现过，或者 `robot learns` 这个组合从未出现过，模型计算出的概率也会是零。模型无法理解 `agent` 和 `robot` 在语义上的相似性。



**（2）神经网络语言模型与词嵌入**

N-gram 模型的根本缺陷在于它将词视为孤立、离散的符号。为了克服这个问题，研究者们转向了神经网络，并提出了一种思想：用连续的向量来表示词。

2003年，Bengio 等人提出的**前馈神经网络语言模型 (Feedforward Neural Network Language Model)**。

1. **构建一个语义空间**：创建一个高维的连续向量空间，然后将词汇表中的每个词都映射为该空间中的一个点。这个点（即向量）就被称为**词嵌入 (Word Embedding)** 或词向量。在这个空间里，语义上相近的词，它们对应的向量在空间中的位置也相近。例如，`agent` 和 `robot` 的向量会靠得很近，而 `agent` 和 `apple` 的向量会离得很远。

2. **学习从上下文到下一个词的映射**：利用神经网络的强大拟合能力，来学习一个函数。这个函数的输入是前 n−1 个词的词向量，输出是词汇表中每个词在当前上下文后出现的概率分布。

   ![图片描述](https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/3-figures/1757249275674-1.png)

在这个架构中，词嵌入是在模型训练过程中自动学习得到的。模型为了完成“预测下一个词”这个任务，会不断调整每个词的向量位置，最终使这些向量能够蕴含丰富的语义信息。一旦我们将词转换成了向量，我们就可以用数学工具来度量它们之间的关系。最常用的方法是**余弦相似度 (Cosine Similarity)** ，它通过计算两个向量夹角的余弦值来衡量它们的相似性。

![5](D:\notebook\5.png)

这个公式的含义是：

- 如果两个向量方向完全相同，夹角为0°，余弦值为1，表示完全相关。
- 如果两个向量方向正交，夹角为90°，余弦值为0，表示毫无关系。
- 如果两个向量方向完全相反，夹角为180°，余弦值为-1，表示完全负相关。

```python
"""一个著名的例子展示了词向量捕捉到的语义关系： vector('King') - vector('Man') + vector('Woman') 这个向量运算的结果，在向量空间中与 vector('Queen') 的位置惊人地接近。这好比在进行语义的平移：我们从“国王”这个点出发，减去“男性”的向量，再加上“女性”的向量，最终就抵达了“女王”的位置。这证明了词嵌入能够学习到“性别”、“皇室”这类抽象概念。"""

import numpy as np

# 假设我们已经学习到了简化的二维词向量
embeddings = {
    "king": np.array([0.9, 0.8]),
    "queen": np.array([0.9, 0.2]),
    "man": np.array([0.7, 0.9]),
    "woman": np.array([0.7, 0.3])
}

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product

# king - man + woman
result_vec = embeddings["king"] - embeddings["man"] + embeddings["woman"]

# 计算结果向量与 "queen" 的相似度
sim = cosine_similarity(result_vec, embeddings["queen"])

print(f"king - man + woman 的结果向量: {result_vec}")
print(f"该结果与 'queen' 的相似度: {sim:.4f}")

>>>
king - man + woman 的结果向量: [0.9 0.2]
该结果与 'queen' 的相似度: 1.0000

```

神经网络语言模型通过词嵌入，成功解决了 N-gram 模型的泛化能力差的问题。然而，它仍然有一个类似 N-gram 的限制：上下文窗口是固定的。它只能考虑固定数量的前文，这为能处理任意长序列的循环神经网络埋下了伏笔。



**(3)循环神经网络（RNN）与长短时记忆网络（LSTM）**

为了打破固定窗口的限制，**循环神经网络 (Recurrent Neural Network, RNN)** 应运而生，其核心思想是为网络增加“记忆”能力。

RNN 的设计引入了一个**隐藏状态 (hidden state)** 向量，我们可以将其理解为网络的短期记忆。在处理序列的每一步，网络都会读取当前的输入词，并结合它上一刻的记忆（即上一个时间步的隐藏状态），然后生成一个新的记忆（即当前时间步的隐藏状态）传递给下一刻。这个循环往复的过程，使得信息可以在序列中不断向后传递。

标准的 RNN 在实践中存在一个严重的问题：**长期依赖问题 (Long-term Dependency Problem)** 。在训练过程中，模型需要通过反向传播算法根据输出端的误差来调整网络深处的权重。对于 RNN 而言，序列的长度就是网络的深度。当序列很长时，梯度在从后向前传播的过程中会经过多次连乘，这会导致梯度值快速趋向于零（**梯度消失**）或变得极大（**梯度爆炸**）。梯度消失使得模型无法有效学习到序列早期信息对后期输出的影响，即难以捕捉长距离的依赖关系。

为了解决长期依赖问题，**长短时记忆网络 (Long Short-Term Memory, LSTM)** 被设计出来。LSTM 是一种特殊的 RNN，其核心创新在于引入了**细胞状态 (Cell State)** 和一套精密的**门控机制 (Gating Mechanism)** 。细胞状态可以看作是一条独立于隐藏状态的信息通路，允许信息在时间步之间更顺畅地传递。门控机制则是由几个小型神经网络构成，它们可以学习如何有选择地让信息通过，从而控制细胞状态中信息的增加与移除。这些门包括：

- **遗忘门 (Forget Gate)**：决定从上一时刻的细胞状态中丢弃哪些信息。
- **输入门 (Input Gate)**：决定将当前输入中的哪些新信息存入细胞状态。
- **输出门 (Output Gate)**：决定根据当前的细胞状态，输出哪些信息到隐藏状态。

**Tip**:RNN的隐藏状态没有增删效果，无论是什么后来者信息都会一起参杂进最初的记忆导致稀释，而LSTM中不仅有隐藏状态还有细胞状态，细胞状态保存长期信息，隐藏状态保存关键信息就可以达到长短时记忆的效果。

#### 3.1.2Transformer架构解析

我们看到RNN及LSTM通过引入循环结构来处理序列数据，这在一定程度上解决了捕捉长距离依赖的问题。然而，这种循环的计算方式也带来了新的瓶颈：它必须按顺序处理数据。第 t 个时间步的计算，必须等待第 t−1 个时间步完成后才能开始。这意味着 RNN 无法进行大规模的并行计算，在处理长序列时效率低下，这极大地限制了模型规模和训练速度的提升。

Transformer在2017 年诞生，它完全抛弃了循环结构，完全依赖一种名为**注意力 (Attention)** 的机制来捕捉序列内的依赖关系，从而实现了真正意义上的并行计算。

```
输入 → 多头注意力 → 残差连接 + 层归一化 → 前馈网络 → 残差连接 + 层归一化 → 输出

全流程：
源序列 → 词嵌入 + 位置编码 → Encoder 堆叠（N层）→ 上下文向量
                                 ↓
目标序列 → 词嵌入 + 位置编码 → Decoder 堆叠（N层）→ 线性层 + Softmax → 输出概率
```

|     阶段     |                           核心操作                           |                    关键组件                    |           输出结果           |
| :----------: | :----------------------------------------------------------: | :--------------------------------------------: | :--------------------------: |
|  输入预处理  |                      词嵌入 + 位置编码                       |         Embedding、Positional Encoding         | 带位置信息的词向量（512 维） |
| Encoder 编码 |         堆叠 N 层：自注意力→残差 + LN→FFN→残差 + LN          |       多头自注意力、残差、LayerNorm、FFN       |  源序列上下文向量（512 维）  |
| Decoder 解码 | 堆叠 N 层：掩码自注意力→残差 + LN→交叉注意力→残差 + LN→FFN→残差 + LN | 掩码自注意力、交叉注意力、残差、LayerNorm、FFN | 目标序列的特征向量（512 维） |
|    输出层    |                       线性层 + Softmax                       |                Linear、Softmax                 |      目标词表的概率分布      |

**（1）Encoder-Decoder整体架构**

最初的 Transformer 模型在宏观上遵循了一个经典的**编码器-解码器 (Encoder-Decoder)** 架构。

<img src="https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/3-figures/1757249275674-3.png" alt="图片描述" style="zoom: 40%;" />

我们可以将这个结构理解为一个分工明确的团队：

1. **编码器 (Encoder)** ：任务是“**理解**”输入的整个句子。它会读取所有输入词元(3.2.2节)，最终为每个词元生成一个富含上下文信息的向量表示。
2. **解码器 (Decoder)** ：任务是“**生成**”目标句子。它会参考自己已经生成的前文，并“咨询”编码器的理解结果，来生成下一个词。

首先，我们搭建出 Transformer 完整的代码框架，定义好所有需要的类和方法。

```python
import torch
import torch.nn as nn
import math

# --- 占位符模块，将在后续小节中实现 ---

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def forward(self, x):
        pass

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def forward(self, query, key, value, mask):
        pass

class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def forward(self, x):
        pass

# --- 编码器核心层 ---

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# --- 解码器核心层 ---

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.cross_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

```

**（2）输入的前奏**

**1.词向量的转换**（input embedding）

首先需要将一句话通过分词分为各类词向量（例如史蒂夫建造房子，可分为“史蒂夫”，“建造”，“房子”），然后建立起这句话相应的分词表，每个词有其对应的token（例如为史蒂夫分配以一个10的token值去唯一标记这个词），同时分词表还会有三个特殊符号：

<PAD>：若有多个输入句子，用于对其句子的长度

<SOS>：标记句子开始

<EOS>：标记句子结束

最后一个输入的句子：史蒂夫建造房子就转换为了[10,11,12]。

根据词汇表，我们会构建一个embedding矩阵（此矩阵是后来训练出来的，每一张表代表一个句子，每一行代表一个词语，数据由后续输入填充），每一行都是一个词汇，以token作为唯一标记，每一列都是其中的一个维度，一般设置为512大小。

**2.位置的标明**（Positional Encoding）

现在有了词汇表，但词语之间的顺序被打乱了，所以我们现在为了让transformer架构识别位置信息需要加入位置编码。

位置编码生成公式是正弦函数和余线函数。对于每一行的每一个维度（列）都需要使用位置编码生成公式。

<img src="D:\notebook\7.png" alt="7" style="zoom:80%;" />

对于偶数维度使用正弦函数，对于奇数维度使用余弦函数。

```python
class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

位置编码是固定值，不需要学习，而且可表明相对位置关系。

**3.编码器的输入矩阵生成**（Encoder Input）

我们将输入矩阵+位置编码矩阵相加所获得的就是编码器的输入矩阵。

矩阵相加之后，token包含了词的语义信息和位置信息，输入准备在此完成。

**（3）骨架中：从自注意力到多头注意力**

这是骨架中最关键的模块，注意力机制。

想象一下我们阅读这个句子：“The agent learns because **it** is intelligent.”。当我们读到加粗的 "**it**" 时，为了理解它的指代，我们的大脑会不自觉地将更多的注意力放在前面的 "agent" 这个词上。

**自注意力 (Self-Attention)** 机制就是对这种现象的数学建模。它允许模型在处理序列中的每一个词时，都能兼顾句子中的所有其他词，并为这些词分配不同的“注意力权重”。权重越高的词，代表其与当前词的关联性越强，其信息也应该在当前词的表示中占据更大的比重。

为了实现上述过程，自注意力机制为每个输入的词元向量引入了三个可学习的角色：

- **查询 (Query, Q)**：代表当前词元，它正在主动地“查询”其他词元以获取信息。

- **键 (Key, K)**：代表句子中可被查询的词元“标签”或“索引”。

- **值 (Value, V)**：代表词元本身所携带的“内容”或“信息”。

  ***本质是通过Q和K的相似度来决定关注哪个V，举例子【小明想吃苹果，对于“小明”这个词，Q是想知道谁是主语，K是表示小明是主语，V是表示“小明”这个人】***

  

这三个向量都是由原始的词嵌入向量乘以三个不同的、可学习的权重矩阵 (WQ,WK,WV) 得到的。整个计算过程可以分为以下几步：

- 准备：对于句子中的每个词，都通过权重矩阵生成其Q,K,V向量。
- 【若是多头这里进行分头操作】
- 计算相关性得分：要计算词A的新表示，就用词A的Q向量，去和句子中所有词（包括A自己）的K向量进行点积运算。这个得分反映了其他词对于理解词A的重要性。*【也就是量化词A在这句话中的语义，每一个维度乘积表达两个词向量之间的关联程度，比如“小明”与“喜欢”的乘积值大小表达了主语和谓语的关联关系】。*将得到的所有分数除以一个缩放因子*dk*（*dk*是*K*向量的维度），这一步的作用是缩小每个向量的数值范围，以防止值过大导致梯度消失（比如1和100softmax的话，1会被稀释至接近于0）。
- 【若多个句子同时输入长度不一，在前面已经提过需要用到<PAD>，但这个并没有实际意义，所以在计算中我们需要用到掩码矩阵（Mask）忽略这个词向量，这个实现很简单，我们可以将掩码矩阵（对应pad位置设为-1e9）与原矩阵相加得到的依然是很小的负数，在softmax后其注意力权重会接近于0，相当于不参与注意力分配。】
- 归一化：用Softmax函数将分数转换成总和为1的权重概率分布，也就是归一化的过程。
- 加权求和：将上一步得到的权重分别乘以每个词对应的*V*向量，然后将所有结果相加。最终得到的向量，就是词*A*融合了全局上下文信息后的新表示。

这个过程可以用一个简洁的公式：**缩放点积注意力** 来概括：

![6](D:\notebook\6.png)

所以最后得到的结果矩阵维度一定是本句话中的词数量*词数量，以表明每个词向量对其他词向量的关注程度（关联程度）。在这一操作过后，小明这一行维度的token会得到综合了其他上下文，融合了整个句子的词向量。

如果只进行一次上述的注意力计算（即单头），模型可能会只学会关注一种类型的关联。比如，在处理 "it" 时，可能只学会了关注主语。但语言中的关系是复杂的，我们希望模型能同时关注多种关系（如指代关系、时态关系、从属关系等）。多头注意力机制应运而生。它的思想很简单：把一次做完变成分成几组，分开做，再合并。

它将原始的 Q, K, V 向量在维度上切分成 h 份（h 就是“头”数），每一份都独立地进行一次单头注意力的计算。这就好比让 h 个不同的“专家”从不同的角度去审视句子，每个专家都能捕捉到一种不同的特征关系。最后，将这 h 个专家的“意见”（即输出向量）拼接起来，再通过一个线性变换进行整合，就得到了最终的输出。

<img src="https://raw.githubusercontent.com/datawhalechina/Hello-Agents/main/docs/images/3-figures/1757249275674-4.png" alt="图片描述" style="zoom:33%;" />

如图，这种设计让模型能够共同关注来自不同位置、不同表示子空间的信息，极大地增强了模型的表达能力。

多头就是将Q，K，V分为n个头，例如对于512维8头设定，输入向量与W_q,W_k,W_v相乘获得Q，K，V，然后将这三个矩阵进行分头操作，每64列为一个头，共8个。拼接就是简单的矩阵同行拼接在一起变为64*8列。



以下是多头注意力的简单实现：

```python
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定义 Q, K, V 和输出的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        #缩放点积注意力
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 计算注意力得分 (QK^T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 将掩码中为 0 的位置设置为一个非常小的负数，这样 softmax 后会接近 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重 (Softmax)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加权求和 (权重 * V)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 将输入 x 的形状从 (batch_size, seq_length, d_model)
        # 变换为 (batch_size, num_heads, seq_length, d_k)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 将输入 x 的形状从 (batch_size, num_heads, seq_length, d_k)
        # 变回 (batch_size, seq_length, d_model)
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. 对 Q, K, V 进行线性变换
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出并进行最终的线性变换
        output = self.W_o(self.combine_heads(attn_output))
        return output

```

**（4）残差连接 和 层归一化**

这是 Transformer 能训练深层模型、避免梯度消失的核心设计。在 Transformer 的每个编码器和解码器层中，所有子模块（如多头注意力和前馈网络）都被一个 `Add & Norm` 操作包裹。

Transformer 通常有 6/12/24 层，如果没有残差和层归一化：

1. **梯度消失**：模型层数越深，梯度在反向传播时会被不断衰减，底层参数几乎无法更新；
2. **训练爆炸**：参数更新时，特征值的分布会逐渐偏移（内部协变量偏移，ICS），导致模型输出值过大 / 过小，训练崩溃；
3. **特征退化**：深层模型可能学到 “无用特征”，甚至不如浅层模型的效果。

1.残差连接

深层网络容易梯度消失，残差让信息“直接走捷径”，保证训练稳定。作用是让模型只学习对原特征向量的修正量

​	设：

- 输入：`x`（形状 `[batch, seq_len, d_model]`）；
- 多头注意力的计算（可视为一个函数）：`F(x)`（注意力输出）；
- 残差连接输出：`x + F(x)`（逐元素相加）。
- 有残差相当于山顶和山脚之间有一条 “索道”（`x` 直接传递），即使 `F(x)` 学不到有用特征（`F(x)=0`），也能通过 `x + 0 = x` 保留原始特征，梯度也能通过 “索道” 直接传回低层（反向传播时 `d(x+F(x))/dx = 1 + dF(x)/dx`，梯度至少有 “1” 的保底，不会消失）。

2.层归一化（layer normal层）

该操作对单个样本的所有特征进行归一化，使其均值为0，方差为1。这解决了模型训练过程中的**内部协变量偏移 (Internal Covariate Shift)** 问题，使每一层的输入分布保持稳定，从而加速模型收敛并提高训练的稳定性。

不管是线性层、注意力、卷积，本质都是：

```
	y = W * x + b
```

- x：输入特征
- W：权重（参数）
- b：偏置

每经过一层，就是一次乘法与加法的混合运算，随着反向传播的过程，w和b在不断被调整，随着调整次数的增多整体会被增大；

在激活函数的作用下，例如ReLu：负数 → 变 0，正数 → 保留；这样方差会越来越大。

而层归一化就是通过公式计算拉回整体数值均值为0，方差为1。

**（5）前馈神经网络(Feed Forward Network，FFN)**

注意力层的作用是进行信息融合，让每个token去关注其他的token，但是只能产生token之间的两两组合，终究还是线性的。

为了让模型有更强的表达能力，Transformer在每个注意力层后都会加一个前馈神经网络，FFN层会在每个位置上对注意力融合后的表示进行进一步非线性加工，数学表达为：

```
	FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
```

具体操作中，一般要先升维，展开高维特征，在激活函数的作用下进行非线性运算后，在降维，代码如下：

```python
class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x

```

#### 3.1.3Decoder-Only架构

一个完整的Transformer 模型能在很多端到端的场景表现出色。但是当任务转换为构建一个与人对话、创作、作为智能体大脑的通用模型时，或许并不需要那么复杂的结构。

Transformer的设计哲学是“先理解，再生成”。编码器负责深入理解输入的整个句子，形成一个包含全局信息的上下文记忆，然后解码器基于这份记忆来生成翻译。但无论是回答问题、写故事还是生成代码，本质上都是在一个已有的文本序列后面，一个词一个词地添加最合理的内容。基于这个思想，GPT 做了一个大胆的简化：**它完全抛弃了编码器，只保留了解码器部分。** 这就是 **Decoder-Only** 架构的由来。

Decoder-Only 架构的工作模式被称为**自回归 (Autoregressive)**，也就是自己预测结果后加入到自己给出预测提示的序列里 ，用于生成下一次预测。这个听起来很专业的术语，其实描述了一个非常简单的过程：

1. 给模型一个起始文本（例如 “Datawhale Agent is”）。
2. 模型预测出下一个最有可能的词（例如 “a”）。
3. 模型将自己刚刚生成的词 “a” 添加到输入文本的末尾，形成新的输入（“Datawhale Agent is a”）。
4. 模型基于这个新输入，再次预测下一个词（例如 “powerful”）。



但解码器需要保证在预测第t个词的时候，不偷看第t+1个词的答案（也就是避免从训练数据中提前参考不一定同语义的句子原话），通过**掩码自注意力 (Masked Self-Attention)** ，可以实现这一需要：

在自注意力机制计算出注意力分数矩阵（即每个词对其他所有词的关注度得分）之后，但在进行 Softmax 归一化之前，模型会应用一个“掩码”。这个掩码会将所有位于当前位置之后（即目前尚未观测到）的词元对应的分数，替换为一个非常大的负数。

当这个带有负无穷分数的矩阵经过 Softmax 函数时，这些位置的概率就会变为 0。这样一来，模型在计算任何一个位置的输出时，都从数学上被阻止了去关注它后面的信息。这种机制保证了模型在预测下一个词时，能且仅能依赖它已经见过的、位于当前位置之前的所有信息，从而确保了预测的公平性和逻辑的连贯性。

总而言之，从 Transformer 的解码器演变而来的 **Decoder-Only 架构**，通过“预测下一个词”这一简单的范式，开启了我们今天所处的大语言模型时代。



### 3.2与大语言模型的交互

#### 3.2.1提示工程

如果我们把大语言模型比作一个能力极强的“大脑”，那么**提示 (Prompt)** 就是我们与这个“大脑”沟通的语言。提示工程，就是研究如何设计出精准的提示，从而引导模型产生我们期望输出的回复。对于构建智能体而言，一个精心设计的提示能让智能体之间协作分工变得高效。

**（1）模型采样参数**

在使用大模型时，你会经常看到类似`Temperature`这类的可配置参数，其本质是通过调整模型对 “概率分布” 的采样策略，让输出匹配具体场景需求，配置合适的参数可以提升Agent在特定场景的性能。

传统的概率分布是由 Softmax 公式计算得到的，采样参数的本质就是在此基础上，根据不同策略“重新调整”或“截断”分布，从而改变大模型输出的下一个token，也就是选出下一个生成词。

**1.softmax**

<img src="C:\Users\Songjia\AppData\Roaming\Typora\typora-user-images\image-20260220185739175.png" alt="image-20260220185739175" style="zoom: 80%;" />

**2.Temperature（温度）**

**温度是控制模型输出 “随机性” 与 “确定性” 的关键参数。其原理是引入温度系数T>0**：

![image-20260220185936628](C:\Users\Songjia\AppData\Roaming\Typora\typora-user-images\image-20260220185936628.png)

当T变小时，分布“更加陡峭”，高概率项权重进一步放大，生成更“保守”且重复率更高的文本。当T变大时，分布“更加平坦”，低概率项权重提升，生成更“多样”但可能出现不连贯的内容。

- 低温度（0 ⩽ Temperature < 0.3）时输出更 “精准、确定”。适用场景： 事实性任务：如问答、数据计算、代码生成； 严谨性场景：法律条文解读、技术文档撰写、学术概念解释等场景。
- 中温度（0.3 ⩽ Temperature < 0.7）：输出 “平衡、自然”。适用场景： 日常对话：如客服交互、聊天机器人； 常规创作：如邮件撰写、产品文案、简单故事创作。
- 高温度（0.7 ⩽ Temperature < 2）：输出 “创新、发散”。适用场景： 创意性任务：如诗歌创作、科幻故事构思、艺术灵感启发； 发散性思考。

**3.Tok-k**

**其原理是将所有 token 按概率从高到低排序，取排名前 k 个的 token 组成 “候选集”，随后对筛选出的 k 个 token 的概率进行 “归一化”**：

![image-20260220190228946](C:\Users\Songjia\AppData\Roaming\Typora\typora-user-images\image-20260220190228946.png)

Top-k 采样通过 k 值限制候选 token 的数量（只保留前 k 个高概率 token），再从其中采样。当k=1时输出完全确定，退化为 “贪心采样”。

**4.Top-p**

其原理是将所有 token 按概率从高到低排序，从排序后的第一个 token 开始，逐步累加概率，直到累积和首次达到或超过阈值 p，此时累加过程中包含的所有 token 组成 **“核集合”**，最后对核集合进行归一化。

- 与Top-k的区别与联系：相对于固定截断大小的 Top-k，Top-p 能动态适应不同分布的“长尾”特性，对概率分布不均匀的极端情况的适应性更好。

在文本生成中，当同时设置 Top-p、Top-k 和温度系数时，这些参数会按照分层过滤的方式协同工作，其优先级顺序为：温度调整→Top-k→Top-p。温度调整整体分布的陡峭程度，Top-k 会先保留概率最高的 k 个候选，然后 Top-p 会从 Top-k 的结果中选取累积概率≥p 的最小集合作为最终的候选集。不过，通常 Top-k 和 Top-p 二选一即可，若同时设置，实际候选集为两者的交集。 需要注意的是，如果将温度设置为 0，则 Top-k 和 Top-p 将变得无关紧要，因为最有可能的 Token 将成为下一个预测的 Token；如果将 Top-k 设置为 1，温度和 Top-p 也将变得无关紧要，因为只有一个 Token 通过 Top-k 标准，它将是下一个预测的 Token。

**（2）零样本、单样本与少样本提示**

根据我们给模型提供示例（Exemplar）的数量，提示可以分为三种类型。为了更好地理解它们，让我们以一个情感分类任务为例，目标是让模型判断一段文本的情感色彩（如正面、负面或中性）。

1.**零样本提示 (Zero-shot Prompting)** 这指的是我们不给模型任何示例，直接让它根据指令完成任务。这得益于模型在海量数据上预训练后获得的强大泛化能力。

案例： 我们直接向模型下达指令，要求它完成情感分类任务。

```Python
文本:Datawhale的AI Agent课程非常棒！
情感:正面
```

2.**单样本提示 (One-shot Prompting)** 我们给模型提供一个完整的示例，向它展示任务的格式和期望的输出风格。

案例： 我们先给模型一个完整的“问题-答案”对作为示范，然后提出我们的新问题。

```Python
文本:这家餐厅的服务太慢了。
情感:负面

文本:Datawhale的AI Agent课程非常棒！
情感:Copy to clipboardErrorCopied
```

模型会模仿给出的示例格式，为第二段文本补全“正面”。

3.**少样本提示 (Few-shot Prompting)** 我们提供多个示例，这能让模型更准确地理解任务的细节、边界和细微差别，从而获得更好的性能。

案例： 我们提供涵盖了不同情况的多个示例，让模型对任务有更全面的理解。

```Python
文本:这家餐厅的服务太慢了。
情感:负面

文本:这部电影的情节很平淡。
情感:中性

文本:Datawhale的AI Agent课程非常棒！
情感:
```

模型会综合所有示例，更准确地将最后一句的情感分类为“正面”。

**（3）指令调优的影响**

早期的 GPT 模型（如 GPT-3）主要是“文本补全”模型，它们擅长根据前面的文本续写，但不一定能很好地理解并执行人类的指令。

**指令调优 (Instruction Tuning)** 是一种微调技术，它使用大量“指令-回答”格式的数据对预训练模型进行进一步的训练。经过指令调优后，模型能更好地理解并遵循用户的指令。

```
### Instruction:
{用户的问题/指令}

### Response:
{标准答案}

eg：
### Instruction:
解释什么是大模型。

### Response:
大模型是具有海量参数、通过大量数据训练、能理解和生成自然语言的深度学习模型。
```

**（4）基础提示技巧**

1.**角色扮演 (Role-playing)** 通过赋予模型一个特定的角色，我们可以引导它的回答风格、语气和知识范围，使其输出更符合特定场景的需求。

```
# 案例
你现在是一位资深的Python编程专家。请解释一下Python中的GIL（全局解释器锁）是什么，要让一个初学者也能听懂。
```

2.**上下文示例 (In-context Example)** 这与少样本提示的思想一致，通过在提示中提供清晰的输入输出示例，来“教会”模型如何处理我们的请求，尤其是在处理复杂格式或特定风格的任务时非常有效。

```
# 案例
我需要你从产品评论中提取产品名称和用户情感。请严格按照下面的JSON格式输出。

评论:这款“星尘”笔记本电脑的屏幕显示效果惊人，但我不太喜欢它的键盘手感。
输出:{"product_name": "星尘笔记本电脑", "sentiment": "混合"}

评论:我刚买的“声动”耳机音质很棒，续航也超出了我的预期！
输出:
```

**（5）思维链（CoT）**

对于需要逻辑推理、计算或多步骤思考的复杂问题，直接让模型给出答案往往容易出错。**思维链 (Chain-of-Thought, CoT)** 是一种强大的提示技巧，它通过引导模型“一步一步地思考”，提升了模型在复杂任务上的推理能力。

实现 CoT 的关键，是在提示中加入一句简单的引导语，如“请逐步思考”或“Let's think step by step”。

```
# 思维链提示
一个篮球队在一个赛季的80场比赛中赢了60%。在接下来的赛季中，他们打了15场比赛，赢了12场。两个赛季的总胜率是多少？
请一步一步地思考并解答。

>>>
（模型可能会输出）
好的，我们来一步步计算。
第一步:计算第一个赛季赢得的比赛数。
80场 * 60% = 48场。
第二步:计算两个赛季的总比赛数和总胜利数。
总比赛数 = 80 + 15 = 95场。
总胜利数 = 48 + 12 = 60场。
第三步:计算总胜率。
总胜率 = (总胜利数 / 总比赛数) * 100% = (60 / 95) * 100% ≈ 63.16%。
所以，两个赛季的总胜率约为63.16%。
```

#### 3.2.2文本分词

**（1）分词的作用**

我们知道，计算机本质上只能理解数字。因此，在将自然语言文本喂给大语言模型之前，必须先将其转换成模型能够处理的数字格式。这个将文本序列转换为数字序列的过程，就叫做**分词 (Tokenization)** 。**分词器 (Tokenizer)** 的作用，就是定义一套规则，将原始文本切分成一个个最小的单元，我们称之为**词元 (Token)** 。

**按词分词**和**按字符分词**都有问题，按词分词会使得出现“未登录词”，而按字符分词会存在模型需要将字符组合为有意义的词的情况，导致学习力低下。

为了兼顾词表大小和语义表达，现代大语言模型普遍采用**子词分词 (Subword Tokenization)** 算法。它的核心思想是：将常见的词（如 "agent"）保留为完整的词元，同时将不常见的词（如 "Tokenization"）拆分成多个有意义的子词片段（如 "Token" 和 "ization"）。这样既控制了词表的大小，又能让模型通过组合子词来理解和生成新词。

**（2）字节对编码 算法解析**

字节对编码 (Byte-Pair Encoding, BPE) 是最主流的子词分词算法之一，GPT系列模型就采用了这种算法。其核心思想非常简洁，可以理解为一个“贪心”的合并过程：

1. **初始化**：将词表初始化为所有在语料库中出现过的基本字符。
2. **迭代合并**：在语料库上，统计所有相邻词元对的出现频率，找到频率最高的一对，将它们合并成一个新的词元，并加入词表。
3. **重复**：重复第 2 步，直到词表大小达到预设的阈值。

**案例演示：** 假设我们的迷你语料库是 `{"hug": 1, "pug": 1, "pun": 1, "bun": 1}`，并且我们想构建一个大小为 10 的词表。BPE 的训练过程可以用下表3.1来表示：

<img src="C:\Users\Songjia\AppData\Roaming\Typora\typora-user-images\image-20260220214712111.png" alt="image-20260220214712111" style="zoom:67%;" />

理解分词算法的细节并非目的，但作为智能体的开发者，理解分词器的实际影响十分重要，这直接关系到智能体的性能、成本和稳定性：

- **上下文窗口限制**：模型的上下文窗口（如 8K, 128K）是以 **Token 数量**计算的，而不是字符数或单词数。同样一段话，在不同语言（如中英文）或不同分词器下，Token 数量可能相差巨大。精确管理输入长度、避免超出上下文限制是构建长时记忆智能体的基础。
- **API 成本**：大多数模型 API 都是按 Token 数量计费的。了解你的文本会被如何分词，是预估和控制智能体运行成本的关键一步。
- **模型表现的异常**：有时模型的奇怪表现根源在于分词。例如，模型可能很擅长计算 `2 + 2`，但对于 `2+2`（没有空格）就可能出错，因为后者可能被分词器视为一个独立的、不常见的词元。同样，一个词因为首字母大小写不同，也可能被切分成完全不同的 Token 序列，从而影响模型的理解。

#### **3.2.3调用开源大模型**

对于许多需要处理敏感数据、希望离线运行或想精细控制成本的场景，将大语言模型直接部署在本地就显得至关重要。**Hugging Face Transformers** 是一个强大的开源库，它提供了标准化的接口来加载和使用数以万计的预训练模型。我们将使用它来完成本次实践。

在 `transformers` 库中，我们通常使用 `AutoModelForCausalLM` 和 `AutoTokenizer` 这两个类来自动加载与模型匹配的权重和分词器。

- `AutoTokenizer`：自动匹配模型的分词器（把文字转成模型能懂的数字）；
- `AutoModelForCausalLM`：自动加载 “因果语言模型”（大模型本体，负责根据前文生成后文）

为了让大多数读者都能在个人电脑上顺利运行，我们特意选择了一个小规模但功能强大的模型：`Qwen/Qwen1.5-0.5B-Chat`。这是一个由阿里巴巴达摩院开源的拥有约 5 亿参数的对话模型，它体积小、性能优异，非常适合入门学习和本地部署。

#### **3.2.4模型的选择**

我们成功地在本地运行了一个小型的开源语言模型。这自然引出了一个对于智能体开发者而言至关重要的问题：在当前数百个模型百花齐放的背景下，我们应当如何为特定的任务选择最合适的模型？

选择语言模型并非简单地追求“最大、最强”，而是一个在性能、成本、速度和部署方式之间进行权衡的决策过程。本节将首先梳理模型选型的几个关键考量因素，然后对当前主流的闭源与开源模型进行梳理。

在为智能体选择大语言模型时，可以从以下几个维度进行综合评估：

- **性能与能力**：这是最核心的考量。不同的模型擅长的任务不同，有的长于逻辑推理和代码生成，有的则在创意写作或多语言翻译上更胜一筹。
- **成本**：对于闭源模型，成本主要体现在 API 调用费用，通常按 Token 数量计费。对于开源模型，成本则体现在本地部署所需的硬件（GPU、内存）和运维上。
- **速度（延迟）**：对于需要实时交互的智能体（如客服、游戏 NPC），模型的响应速度至关重要。一些轻量级或经过优化的模型（如 GPT-3.5 Turbo, Claude 3.5 Sonnet）在延迟上表现更优。
- **上下文窗口**：模型能一次性处理的 Token 数量上限。对于需要理解长文档、分析代码库或维持长期对话记忆的智能体，选择一个拥有较大上下文窗口（如 128K Token 或更高）的模型是必要的。
- **部署方式**：使用 API 的方式最简单便捷，但数据需要发送给第三方，且受限于服务商的条款。本地部署则能确保数据隐私和最高程度的自主可控，但对技术和硬件要求更高。
- **生态与工具链**：一个模型的流行程度也决定了其周边生态的成熟度。主流模型通常拥有更丰富的社区支持、教程、预训练模型、微调工具和兼容的开发框架（如 LangChain, LlamaIndex, Hugging Face Transformers），这能极大地加速开发进程，降低开发难度。
- **可微调性与定制化**：对于需要处理特定领域数据或执行特定任务的智能体，模型的微调能力至关重要。一些模型提供了便捷的微调接口和工具，允许开发者使用自己的数据集对模型进行定制化训练，从而显著提升模型在特定场景下的性能和准确性。开源模型在这方面通常提供更大的灵活性。

**闭源模型**通常代表了当前 AI 技术的最前沿，并提供稳定、易用的 API 服务，是构建高性能智能体的首选。**开源模型**为开发者提供了最高程度的灵活性、透明度和自主性，催生了繁荣的社区生态。它们允许开发者在本地部署、进行定制化微调，并拥有完整的模型控制权。

### 3.3大语言模型的缩放法则与局限性

#### 3.3.1缩放法则

**缩放法则（Scaling Laws）**是近年来大语言模型领域最重要的发现之一。它揭示了模型性能与模型参数量、训练数据量以及计算资源之间存在着可预测的幂律关系。这一发现为大语言模型的持续发展提供了理论指导，阐明了增加资源投入能够系统性提升模型性能的底层逻辑。

研究发现，在对数-对数坐标系下，模型的性能（通常用损失 Loss 来衡量）与参数量、数据量和计算量这三个因素都呈现出平滑的幂律关系。简单来说，只要我们持续、按比例地增加这三个要素，模型的性能就会可预测地、平滑地提升，而不会出现明显的瓶颈。这一发现为大模型的设计和训练提供了清晰的指导：**在资源允许的范围内，尽可能地扩大模型规模和训练数据量**。

早期的研究更侧重于增加模型参数量，但 DeepMind 在 2022 年提出的“Chinchilla 定律”对此进行了重要修正。该定律指出，在给定的计算预算下，为了达到最优性能，**模型参数量和训练数据量之间存在一个最优配比**。具体来说，最优的模型应该比之前普遍认为的要小，但需要用多得多的数据进行训练。例如，一个 700 亿参数的 Chinchilla 模型，由于使用了比 GPT-3（1750 亿参数）多 4 倍的数据进行训练，其性能反而超越了后者。这一发现纠正了“越大越好”的片面认知，强调了数据效率的重要性，并指导了后续许多高效大模型的设计。

缩放法则最令人惊奇的产物是“**能力的涌现**”。所谓能力涌现，是指当模型规模达到一定阈值后，会突然展现出在小规模模型中完全不存在或表现不佳的全新能力。例如，**链式思考 (Chain-of-Thought)** 、**指令遵循 (Instruction Following)** 、多步推理、代码生成等能力，都是在模型参数量达到数百亿甚至千亿级别后才显著出现的。这种现象表明，大语言模型不仅仅是简单地记忆和复述，它们在学习过程中可能形成了某种更深层次的抽象和推理能力。对于智能体开发者而言，能力的涌现意味着选择一个足够大规模的模型，是实现复杂自主决策和规划能力的前提。

#### 3.3.2模型幻觉

**模型幻觉（Hallucination）**通常指的是大语言模型生成的内容与客观事实、用户输入或上下文信息相矛盾，或者生成了不存在的事实、实体或事件。幻觉的本质是模型在生成过程中，过度自信地“编造”了信息，而非准确地检索或推理。根据其表现形式，幻觉可以被分为多种类型，例如：

- **事实性幻觉 (Factual Hallucinations)** ： 模型生成与现实世界事实不符的信息。
- **忠实性幻觉 (Faithfulness Hallucinations)** ： 在文本摘要、翻译等任务中，生成的内容未能忠实地反映源文本的含义。
- **内在幻觉 (Intrinsic Hallucinations)** ： 模型生成的内容与输入信息直接矛盾。

幻觉的产生是多方面因素共同作用的结果。首先，训练数据中可能包含错误或矛盾的信息。其次，模型的自回归生成机制决定了它只是在预测下一个最可能的词元，而没有内置的事实核查模块。最后，在面对需要复杂推理的任务时，模型可能会在逻辑链条中出错，从而“编造”出错误的结论。

此外，大语言模型还面临着知识时效性不足和训练数据中存在的偏见等挑战。大语言模型的能力来源于其训练数据。这意味着模型所掌握的知识是其训练数据收集时的最新材料。对于在此日期之后发生的事件、新出现的概念或最新的事实，模型将无法感知或正确回答。与此同时训练数据往往包含了人类社会的各种偏见和刻板印象。当模型在这些数据上学习时，它不可避免地会吸收并反映出这些偏见。

为了提高大语言模型的可靠性，研究人员和开发者正在积极探索多种检测和缓解幻觉的方法：

1. **数据层面**： 通过高质量**数据清洗**、**引入事实性知识**以及**强化学习**与**人类反馈 (RLHF)** 等方式，从源头减少幻觉。
2. **模型层面**： 探索新的模型架构，或让模型能够表达其对生成内容的不确定性。
3. **推理与生成层面**：
   1. **检索增强生成 (Retrieval-Augmented Generation, RAG)** ： 这是目前缓解幻觉的有效方法之一。RAG 系统通过在生成之前从外部知识库（如文档数据库、网页）中检索相关信息，然后将检索到的信息作为上下文，引导模型生成基于事实的回答。
   2. **多步推理与验证**： 引导模型进行多步推理，并在每一步进行自我检查或外部验证。
   3. **引入外部工具**： 允许模型调用外部工具（如搜索引擎、计算器、代码解释器）来获取实时信息或进行精确计算。

尽管幻觉问题短期内难以完全消除，但通过上述的策略，可以显著降低其发生频率和影响，提高大语言模型在实际应用中的可靠性和实用性。

# AutoGen 多智能体技术调研助手

这是一个面向课程教学的 LLM-based 多智能体案例。学生输入一个技术主题，多个 Agent 协作检索资料、写作，并在提高版中审阅和修订报告。

案例基于 Microsoft AutoGen 的 AgentChat API，使用 OpenAI 兼容接口。本项目默认适配火山引擎 Ark Coding API，也可以替换为其他兼容服务。

## 三个版本

### 基础版 `basic.py`

流程固定为：

```text
PlannerAgent → SearchAgent → WriterAgent
```

学生可以观察固定顺序的消息传递、工具调用和报告生成。

### 提高版 `advanced.py`

流程增加审阅和一次修订：

```text
PlannerAgent → SearchAgent → WriterAgent → ReviewerAgent
                                                 ↘ 需要修改时回到 WriterAgent
```

提高版使用 `SelectorGroupChat`，由一个简单的路由函数根据消息来源和审阅结果选择下一位 Agent。审阅 Agent 还会检查报告中的 `[S1]` 引用是否来自本次检索结果。

### 改进版 `modified.py`

本版本在原有 `advanced.py` 的基础上进行了进一步改进，提高了多智能体调研系统的证据质量、覆盖范围和引用可靠性。

为此，新增了证据评估 Agent 来评估检索到的资料是否相关、覆盖的方法、技术及观点是否充足。如果证据不足，检索 Agent 会进行一次补充检索。

额外，在 `advanced.py` 的基础上，审核 Agent 还会检查引用编号时候存在、报告 Agent 是否使用了证据评估 Agent 认可的资料及证据引用是否支持报告中的论点。


```text
PlannerAgent → SearchAgent → EvidenceEvaluatorAgent → WriterAgent → ReviewerAgent
                                      ↘ 证据不足回到 SearchAgent           ↘ 需要修改时回到 WriterAgent
```

证据评估 Agent 和审核 Agent 只会运行一次，调整了之后会直接跳到下一步，不会重新检查。


## 快速开始

需要 Python 3.11 或更高版本。

### 1. 创建虚拟环境并安装依赖

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS 或 Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置接口

复制 `.env.example` 为 `.env`，填写自己的 `ARK_API_KEY`。本项目不会把密钥写入运行结果。

默认配置为：

```text
LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3
LLM_MODEL=doubao-seed-2-1-turbo-260628
```

先检查接口：

```powershell
python check_api.py
```

### 3. 运行基础版

```powershell
python basic.py --topic "Application of data-driven fault diagnosis in industrial processes"
```

### 4. 运行提高版

```powershell
python advanced.py --topic "Application of data-driven fault diagnosis in industrial processes"
```

### 5. 运行改进版

```powershell
python modified.py --topic "Application of data-driven fault diagnosis in industrial processes"
```

每次运行会在 `outputs/` 下生成独立目录，包含：

- `report.md`：最终中文技术报告
- `trace.jsonl`：完整消息和工具调用轨迹
- `evidence_registry.json`：本次运行使用的证据编号
- `run_summary.json`：不含密钥的运行配置
- 提高版另有 `review.md`：审阅意见和引用复核结果
- 改进版另有 `evidence_evaluation.md`：证据评估结果

## 工具和资料来源

`research_tools.py` 提供两个来源：

1. 项目内的 `data/course_sources.json`，保证课堂演示至少有一组稳定资料。
2. Crossref 实时检索，不需要额外密钥。Crossref 可能没有摘要，代码会如实标注，不会把标题当成摘要。

如果实时检索失败，报告仍可以使用课程资料库；完整结果会在 `trace.jsonl` 中显示。

## 三个版本的主要区别

| 功能 | 基础版 `basic.py` | 提高版 `advanced.py` | 改进版 `modified.py` |
|---|---|-------------------|---|
| Agent 数量 | 3 | 4                 | 5 |
| Agent 角色 | Planner、Search、Writer | 增加 ReviewerAgent  | 增加 EvidenceEvaluatorAgent |
| MAS 架构 | 固定顺序执行 | SelectorGroupChat | SelectorGroupChat + 自适应检索 |
| 文献检索 | 单轮研究 | 支持一次报告修改          | 支持一次补充检索 |
| 证据评估 | 无 | 无独立评估阶段           | EvidenceEvaluatorAgent |
| 文献数量检查 | 无 | 无                 | 每个问题至少两篇相关文献 |
| 文献多样性检查 | 无 | 无                 | 检查不同方法、技术和观点 |
| 引用审核 | 无独立审核 | 检查引用编号            | 检查引用编号及证据认可状态 |
| 报告修订 | 无 | 最多一次              | 最多一次 |
| 工具调用 | search_literature | search_literature | search_literature、evaluate_evidence、modified_audit_citations |

## 与课程内容的对应关系

| 课程概念 | 案例中的位置 |
|---|---|
| Agent | Planner、Searcher、Writer、Reviewer 各自负责一个目标 |
| Harness | AutoGen 的消息循环、工具调用、终止条件和重试 |
| MAS | 多个有明确角色的 Agent 共享任务上下文 |
| 工具调用 | `search_literature` 和 `audit_citations` |
| 工作流 | 基础版固定顺序，提高版带条件回路 |
| 可靠性 | 引用编号、审阅、最大轮数和人工复核 |

本案例不要求学生自己实现 MCP 或 A2A。协议可以作为课堂讨论的工程扩展，基础作业先把 Agent 分工、消息路由、工具和验证做清楚。

## 官方参考

- [AutoGen AgentChat 文档](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/)
- [AutoGen Literature Review 示例](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/examples/literature-review.html)
- [AutoGen Selector Group Chat 示例](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html)
- [Crossref REST API](https://api.crossref.org/swagger-ui/index.html)

## 注意事项

- 不要把真实 API 密钥写进代码、README 或 Git 仓库。
- 建议先用一个具体主题测试，再扩大检索范围。
- 提高版最多允许一次修订，并设置最大轮数，防止 Agent 无限对话。
- LLM 生成的报告仍需要人工检查，尤其是引用、数字和因果结论。
- AutoGen 和模型服务都会更新，课程发布时请固定依赖版本并保留本 README。

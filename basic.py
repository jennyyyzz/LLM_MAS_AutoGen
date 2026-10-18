"""基础版：顺序式多智能体技术调研助手。

运行：
    python basic.py --topic "LLM-based agents for industrial control"

三个 Agent 按固定顺序工作：规划 → 检索 → 写作。学生可以先运行这一版，
再观察 outputs 目录中的 trace.jsonl，理解消息如何在 Agent 之间传递。
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence
import time
from collections import Counter

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import TextMessage, ToolCallSummaryMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from common import (
    build_model_client,
    last_text,
    load_settings,
    make_run_dir,
    parse_args,
    remove_markers,
    run_team,
    save_run_summary,
)
from research_tools import get_registry, reset_registry

# Replace these prices with the provider's actual rates
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 3.75


async def main() -> None:
    args = parse_args("基础版：顺序式多智能体技术调研助手")
    settings = load_settings()
    reset_registry()
    out_dir = make_run_dir("basic")
    model_client = build_model_client(settings)

    # 单独创建带检索工具的 SearchAgent，避免把工具逻辑藏在框架内部。
    from research_tools import search_literature

    planner = AssistantAgent(
        "PlannerAgent",
        model_client=model_client,
        description="负责把调研主题拆成可检索的问题，并安排检索步骤。",
        system_message=(
            "你是规划 Agent。请把用户主题拆成 2 到 3 个具体检索问题，"
            "并在结尾明确告诉 SearchAgent 要检索哪些关键词。你只负责规划，不要写最终报告。"
        ),
    )
    searcher = AssistantAgent(
        "SearchAgent",
        model_client=model_client,
        tools=[search_literature],
        description="负责检索课程资料库和 Crossref 实时文献。",
        system_message=(
            "你是资料检索 Agent。根据 PlannerAgent 的关键词，至少调用一次 search_literature 工具。"
            "只汇报工具返回的资料，不要凭空补充论文内容。请保留每条资料的 [S#] 编号。"
        ),
        # Ark Coding 接口的部分模型在工具调用后的二次反思可能不返回文本。
        # 关闭反思后，AutoGen 会把工具结果直接作为摘要消息交给 WriterAgent，
        # 课堂上也更容易观察“调用工具 → 传递证据”的过程。
        reflect_on_tool_use=False,
        max_tool_iterations=1,
    )
    writer = AssistantAgent(
        "WriterAgent",
        model_client=model_client,
        description="负责根据检索结果生成带引用的中文技术小报告。",
        system_message=(
            "你是报告 Agent。请根据前面 Agent 提供的证据，写一份 500 到 700 字的中文技术小报告。"
            "报告包含：问题定义、主要发现、不同方法的比较、局限性和结论。"
            "事实性陈述尽量在句末使用 [S1] 这样的编号引用，只能使用已经出现的编号。"
            "最后单独一行写 BASIC_REPORT_DONE。"
        ),
    )
    team = RoundRobinGroupChat(
        [planner, searcher, writer],
        termination_condition=TextMentionTermination("BASIC_REPORT_DONE", sources=["WriterAgent"])
        | MaxMessageTermination(6),
        max_turns=6,
    )
    task = (
        f"请围绕以下主题完成一次小型技术调研：{args.topic}\n"
        "要求：优先使用课程资料库和实时检索结果；报告面向本科生，避免堆砌术语。"
    )
    try:
        start_time = time.perf_counter()
        events = await run_team(team, task, out_dir)
        execution_time = time.perf_counter() - start_time
        report = remove_markers(last_text(events, "WriterAgent"))
        agent_names = {"PlannerAgent", "SearchAgent", "WriterAgent"}
        agent_messages = [event for event in events if
                          isinstance(event, (TextMessage, ToolCallSummaryMessage)) and event.source in agent_names]
        agent_turns = len(agent_messages)
        turns_per_agent = dict(Counter(event.source for event in agent_messages))
        prompt_tokens = 0
        completion_tokens = 0
        for event in events:
            usage = getattr(event, "models_usage", None)
            if usage is not None:
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (
                prompt_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
                + completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
        )
        save_run_summary(
            out_dir,
            kind="basic",
            topic=args.topic,
            report=report,
            evidence=get_registry(),
            settings=settings,
        )
        print(f"\n报告已保存：{out_dir / 'report.md'}")
        print(f"消息轨迹已保存：{out_dir / 'trace.jsonl'}")
        print(f"运行时间：{execution_time:.2f} 秒")
        print(f"Agent 总轮数：{agent_turns}")
        print(f"各 Agent 轮数：{turns_per_agent}")
        print(f"输入 Token 数：{prompt_tokens}")
        print(f"输出 Token 数：{completion_tokens}")
        print(f"Token 总数：{total_tokens}")
        print(f"预估 API 费用：${estimated_cost:.6f}")
    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

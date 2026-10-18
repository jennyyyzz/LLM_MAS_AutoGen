"""提高版：带审阅和一次修订回路的多智能体调研助手。

运行：
    python advanced.py --topic "multi-agent coordination for LLM systems"

与基础版的区别：
1. 增加 ReviewerAgent，检查引用编号和结论是否超出证据。
2. 使用 AutoGen SelectorGroupChat，根据上一条消息把任务路由到合适角色。
3. 审阅不通过时只允许 WriterAgent 修订一次，避免对话无限循环。
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence
import time
from collections import Counter

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage, ToolCallSummaryMessage
from autogen_agentchat.teams import SelectorGroupChat

# Replace these prices with the provider's actual rates
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 3.75


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
from research_tools import audit_citations, get_registry, reset_registry, search_literature


def choose_next(
    messages: Sequence[BaseAgentEvent | BaseChatMessage], state: dict[str, int]
) -> str | None:
    """根据最近一条 Agent 消息选择下一位角色。

    这是提高版的核心：流程不是简单轮流发言，而是由任务状态决定是否
    进入修订。返回 None 时交给 AutoGen 的模型选择器处理。
    """

    names = {"PlannerAgent", "SearchAgent", "WriterAgent", "ReviewerAgent"}
    for message in reversed(messages):
        source = getattr(message, "source", None)
        if source not in names:
            continue
        content = str(getattr(message, "content", ""))
        if source == "PlannerAgent":
            return "SearchAgent"
        if source == "SearchAgent":
            return "WriterAgent"
        if source == "WriterAgent":
            return "ReviewerAgent"
        if source == "ReviewerAgent":
            if "REVISION_REQUIRED" in content and state["revisions"] < 1:
                state["revisions"] += 1
                return "WriterAgent"
            return None
    return "PlannerAgent"


def build_team(model_client: Any) -> tuple[SelectorGroupChat, dict[str, int]]:
    """构造提高版 Team，并返回可观察的修订次数状态。"""

    planner = AssistantAgent(
        "PlannerAgent",
        model_client=model_client,
        description="拆解主题并为检索 Agent 制定问题。",
        system_message=(
            "你是规划 Agent。把用户主题拆成 2 到 3 个具体问题，"
            "最后明确给 SearchAgent 的检索关键词。不要写最终报告。"
        ),
    )
    searcher = AssistantAgent(
        "SearchAgent",
        model_client=model_client,
        tools=[search_literature],
        description="从课程资料库和 Crossref 检索证据。",
        system_message=(
            "你是检索 Agent。根据 PlannerAgent 的关键词调用 search_literature。"
            "至少调用一次工具，只汇报工具返回的资料，并保留 [S#] 编号。"
        ),
        # 与基础版相同，直接传递工具摘要，避免兼容性较弱的二次反思调用。
        reflect_on_tool_use=False,
        max_tool_iterations=1,
    )
    writer = AssistantAgent(
        "WriterAgent",
        model_client=model_client,
        description="根据证据写作并按审阅意见修订报告。",
        system_message=(
            "你是报告 Agent。根据检索证据写 500 到 700 字中文技术小报告，"
            "包含问题定义、主要发现、比较、局限性和结论。事实性陈述使用 [S#] 引用，"
            "只能使用已出现的编号。若收到 ReviewerAgent 的 REVISION_REQUIRED，"
            "请直接输出完整修订稿。首次或修订完成后在末尾写 REPORT_DRAFT_DONE。"
        ),
    )
    reviewer = AssistantAgent(
        "ReviewerAgent",
        model_client=model_client,
        description="检查报告的证据引用、范围和可核查性。",
        system_message=(
            "你是审阅 Agent。阅读 WriterAgent 的完整报告，检查 [S#] 是否来自前面的检索结果。"
            "检查是否存在无证据的事实、过度外推、重复和缺失引用。"
            "若需要修改，先列出最多 3 条明确修改意见，最后单独一行写 REVISION_REQUIRED。"
            "若报告可以提交，最后单独一行写 REVIEW_APPROVED。"
        ),
    )

    state = {"revisions": 0}
    termination = TextMentionTermination("REVIEW_APPROVED", sources=["ReviewerAgent"]) | MaxMessageTermination(20)
    team = SelectorGroupChat(
        [planner, searcher, writer, reviewer],
        model_client=model_client,
        selector_func=lambda messages: choose_next(messages, state),
        allow_repeated_speaker=True,
        termination_condition=termination,
        max_turns=6,
    )
    return team, state


async def main() -> None:
    args = parse_args("提高版：带审阅和修订回路的多智能体调研助手")
    settings = load_settings()
    reset_registry()
    out_dir = make_run_dir("advanced")
    model_client = build_model_client(settings)
    team, state = build_team(model_client)
    task = (
        f"请围绕以下主题完成一次小型技术调研：{args.topic}\n"
        "要求：只根据检索到的证据写作，面向本科生，保留引用编号。"
    )
    try:
        start_time = time.perf_counter()
        events = await run_team(team, task, out_dir)
        execution_time = time.perf_counter() - start_time
        report = remove_markers(last_text(events, "WriterAgent"))
        review = last_text(events, "ReviewerAgent")
        # 审阅 Agent 可能忘记调用工具，课后仍可以查看这一条确定性的检查结果。
        post_audit = audit_citations(report)
        review = review + "\n\n课后引用复核：" + post_audit
        agent_names = {"PlannerAgent", "SearchAgent", "WriterAgent", "ReviewerAgent"}
        agent_messages = [event for event in events if isinstance(event, (TextMessage, ToolCallSummaryMessage)) and event.source in agent_names]
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
            kind="advanced",
            topic=args.topic,
            report=report,
            evidence=get_registry(),
            settings=settings,
            review=review,
        )
        (out_dir / "review.md").write_text(review + "\n", encoding="utf-8")
        print(f"\n报告已保存：{out_dir / 'report.md'}")
        print(f"审阅意见已保存：{out_dir / 'review.md'}")
        print(f"修订次数：{state['revisions']}")
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

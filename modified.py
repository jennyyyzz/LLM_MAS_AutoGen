""" Modified version: Multi-Agent Research Assistant with evidence evaluation,
adaptive literature retrieval, and enhanced citation verification.

Run:
    python modified.py --topic "multi-agent coordination for LLM systems"

Improvements over the advanced version:
1. Adds EvidenceEvaluatorAgent to evaluate source relevance, quantity,
   and diversity before report generation.
2. Introduces adaptive literature retrieval, allowing SearchAgent to
   perform one additional search when insufficient evidence is found.
3. Uses evaluate_evidence to verify that each research question has
   at least two distinct relevant sources.
4. Improves ReviewerAgent with modified_audit_citations to verify that
   citations exist and have been approved by EvidenceEvaluatorAgent,
   and evaluates if the citations correspond to the points made.
5. Retains the single revision loop while recording evidence evaluations
   and additional search attempts for analysis.
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
from research_tools import (
    get_registry,
    reset_registry,
    search_literature,
    evaluate_evidence,
    modified_audit_citations
)

# Replace these prices with the provider's actual rates
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 3.75


def choose_next(
    messages: Sequence[BaseAgentEvent | BaseChatMessage], state: dict[str, int]
) -> str | None:
    """ Included the evidence-evaluation mechanism so that another round of
    search can be made if the evidence is insufficient. """

    names = {"PlannerAgent", "SearchAgent", "EvidenceEvaluatorAgent", "WriterAgent", "ReviewerAgent"}
    for message in reversed(messages):
        source = getattr(message, "source", None)
        if source not in names:
            continue
        content = str(getattr(message, "content", ""))
        if source == "PlannerAgent":
            return "SearchAgent"
        if source == "SearchAgent":
            return "EvidenceEvaluatorAgent"
        if source == "EvidenceEvaluatorAgent":
            if "EVIDENCE_INSUFFICIENT" in content and state["search_retries"] < 1:
                state["search_retries"] += 1
                return "SearchAgent"
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

    planner = AssistantAgent(
        "PlannerAgent",
        model_client=model_client,
        description="拆解主题并为检索 Agent 制定问题。",
        system_message=(
            "你是规划 Agent。把用户主题拆成 2 到 3 个具体问题，"
            "最后明确给 SearchAgent 的检索关键词。不要写最终报告。"
        ),
    )
    """ Modified SearchAgent to search for additional supporting evidence if needed, 
    while trying to avoid repeated sources."""
    searcher = AssistantAgent(
        "SearchAgent",
        model_client=model_client,
        tools=[search_literature],
        description="从课程资料库和 Crossref 检索证据。",
        system_message=(
            "你是检索 Agent。首次检索时，根据 PlannerAgent 的关键词调用 search_literature。"
            "若收到 EvidenceEvaluatorAgent 的 EVIDENCE_INSUFFICIENT，"
            "根据其指出的证据缺口和建议关键词进行针对性补充检索。"
            "每轮执行时，至少调用一次 search_literature 工具。"
            "补充检索时，尽量避免重复使用之前的检索关键词及资料。"
            "只汇报工具返回的资料，并保留 [S#] 编号。"
        ),
        reflect_on_tool_use=False,
        max_tool_iterations=1,
    )
    """ New EvidenceEvaluatorAgent in charge of checking the relevance, quantity
    and diversity of the retrieved sources, ensuring that each research question
    has at least two relevant sources and that different methods or perspectives 
    are adequately covered.
    If evidence is insufficient, it identifies the missing gaps and suggests
    additional search keywords to SearchAgent. """
    evidence_evaluator = AssistantAgent(
        "EvidenceEvaluatorAgent",
        model_client=model_client,
        tools=[evaluate_evidence],
        description="检查检索到的证据与具体问题的相关性，并检查证据数量",
        system_message=(
            "你是证据评估 Agent。阅读 PlannerAgent 提出的具体问题，"
            "以及 SearchAgent 检索到证据标题与摘要。"
            "对于每个调研问题，根据文献的实际内容判断哪些资料与其相关，"
            "不要仅根据关键词匹配判断相关性。"
            "使用 evaluate_evidence，传入每个调研问题及其相关资料编号。"
            "每个研究问题必须至少有两篇不同的相关资料。"
            "同时评估文献是否涵盖研究主题所需的不同方法、技术或观点。"
            "如果研究问题涉及不同方法的比较，确保各主要方法都有相关证据支持。"
            "不要为了增加多样性而加入无关文献。"
            "若有任何问题缺少足够资料或者重要研究方法缺乏证据支持，"
            "先指出缺少证据的问题并提出新的检索关键词，最后单独一行写 EVIDENCE_INSUFFICIENT。"
            "若证据充分并且文献覆盖了研究主题所需的主要方法或观点，"
            "最后单独一行写 EVIDENCE_SUFFICIENT。"
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=1,
    )
    """ Modified WriterAgent to use only the sources approved by the 
    EvidenceEvaluatorAgent. """
    writer = AssistantAgent(
        "WriterAgent",
        model_client=model_client,
        description="根据证据写作并按审阅意见修订报告。",
        system_message=(
            "你是报告 Agent。根据检索证据写 500 到 700 字中文技术小报告，"
            "包含问题定义、主要发现、比较、局限性和结论。事实性陈述使用 [S#] 引用，"
            "只能使用 EvidenceEvaluatorAgent 认可过的证据，不得使用未经批准的文献编号。"
            "若收到 ReviewerAgent 的 REVISION_REQUIRED，"
            "请直接输出完整修订稿。首次或修订完成后在末尾写 REPORT_DRAFT_DONE。"
        ),
    )
    """ Modified ReviewerAgent to verify that all cited sources are approved by 
    the EvidenceEvaluatorAgent using the modified_audit_citations tool, and that
    the citations actually support the corresponding claims based on the titles and
    abstracts of the papers. """
    reviewer = AssistantAgent(
        "ReviewerAgent",
        model_client=model_client,
        tools=[modified_audit_citations],
        description="检查报告的证据引用、范围和可核查性。",
        system_message=(
            "你是审阅 Agent。阅读 WriterAgent 的完整报告。"
            "首先调用 modified_audit_citations 检查报告中的引用编号，"
            "确认所有引用均存在且经过 EvidenceEvaluatorAgent 认可。"
            "然后根据文献标题和摘要判断引用是否真正支持对应论点。"
            "不要仅因为引用编号有效就认为其内容正确。"
            "同时检查是否存在无证据的事实、过度外推、重复、缺失引用和主张与证据不匹配。"
            "若需要修改，先列出最多 3 条明确修改意见，最后单独一行写 REVISION_REQUIRED。"
            "若报告可以提交，最后单独一行写 REVIEW_APPROVED。"
        ),
        reflect_on_tool_use=True,
        max_tool_iterations=1,
    )

    state = {"revisions": 0, "search_retries": 0}
    termination = TextMentionTermination("REVIEW_APPROVED", sources=["ReviewerAgent"]) | MaxMessageTermination(40)
    team = SelectorGroupChat(
        [planner, searcher, evidence_evaluator, writer, reviewer],
        model_client=model_client,
        selector_func=lambda messages: choose_next(messages, state),
        allow_repeated_speaker=True,
        termination_condition=termination,
        max_turns=9,
    )
    return team, state


async def main() -> None:
    args = parse_args("改进版：带证据评估、自适应资料检索、增强引用审核修订回路的多智能体调研助手")
    settings = load_settings()
    reset_registry()
    out_dir = make_run_dir("modified")
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
        evidence_evaluation = last_text(events, "EvidenceEvaluatorAgent")
        review = last_text(events, "ReviewerAgent")
        agent_names = {"PlannerAgent", "SearchAgent", "EvidenceEvaluatorAgent", "WriterAgent", "ReviewerAgent"}
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
            kind="modified",
            topic=args.topic,
            report=report,
            evidence=get_registry(),
            settings=settings,
            evidence_evaluation=evidence_evaluation,
            review=review,
        )
        (out_dir / "evidence_evaluation.md").write_text(evidence_evaluation + "\n", encoding="utf-8")
        (out_dir / "review.md").write_text(review + "\n", encoding="utf-8")
        print(f"\n报告已保存：{out_dir / 'report.md'}")
        print(f"证据审核已保存：{out_dir / 'evidence_evaluation.md'}")
        print(f"审阅意见已保存：{out_dir / 'review.md'}")
        print(f"补充检索次数：{state['search_retries']}")
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

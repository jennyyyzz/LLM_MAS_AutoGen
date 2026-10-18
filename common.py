"""课程案例的公共工具。

本文件只负责配置、运行记录和结果保存；具体的 Agent 角色写在
basic.py 与 advanced.py 中，方便学生对照修改。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


PROJECT_ROOT = Path(__file__).resolve().parent


def load_settings() -> dict[str, Any]:
    """读取 .env 配置，并返回给 AutoGen 使用的模型参数。

    支持 ARK_API_KEY 和 OPENAI_API_KEY 两种变量名，便于学生迁移到
    其他 OpenAI 兼容服务。密钥只从环境变量读取，不写入运行结果。
    """

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("在此填入"):
        raise RuntimeError(
            "未找到 API 密钥。请复制 .env.example 为 .env，并填写 ARK_API_KEY。"
        )

    return {
        "api_key": api_key,
        "base_url": os.getenv(
            "LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"
        ),
        "model": os.getenv("LLM_MODEL", "doubao-seed-2-1-turbo-260628"),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1800")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    }


def build_model_client(settings: dict[str, Any] | None = None) -> OpenAIChatCompletionClient:
    """创建 OpenAI 兼容模型客户端。

    Ark 的模型不一定出现在 AutoGen 的内置模型映射表中，因此显式提供
    model_info，告诉 AutoGen 该模型支持函数调用，但本案例不使用视觉输入。
    """

    settings = settings or load_settings()
    return OpenAIChatCompletionClient(
        model=settings["model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
        timeout=120,
        max_retries=2,
        parallel_tool_calls=False,
        add_name_prefixes=True,
    )


def parse_args(description: str) -> argparse.Namespace:
    """统一命令行参数，默认主题适合课堂演示。"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--topic",
        default="LLM-based multi-agent systems in industrial control",
        help="要调研的主题；建议使用一个具体、可在 5 分钟内检索的主题。",
    )
    return parser.parse_args()


def make_run_dir(kind: str) -> Path:
    """为一次运行建立独立输出目录。"""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / f"{kind}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def jsonable(value: Any) -> Any:
    """把 AutoGen 事件尽量转换为可写入 JSON 的对象。"""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump())
        except Exception:
            pass
    return str(value)


def event_record(event: Any) -> dict[str, Any]:
    """提取消息来源和内容，保留完整事件的可读表示。"""

    record = {
        "type": event.__class__.__name__,
        "source": getattr(event, "source", None),
        "content": jsonable(getattr(event, "content", None)),
    }
    if hasattr(event, "model_dump"):
        try:
            record["event"] = jsonable(event.model_dump())
        except Exception:
            record["event"] = str(event)
    return record


async def run_team(team: Any, task: str, out_dir: Path) -> list[Any]:
    """运行 AutoGen Team，并同时保存消息轨迹。

    终端只显示简短进度；完整消息会写入 trace.jsonl，便于学生复盘
    每个 Agent 看到的上下文和工具调用。
    """

    events: list[Any] = []
    trace_path = out_dir / "trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace_file:
        async for event in team.run_stream(task=task):
            events.append(event)
            trace_file.write(json.dumps(event_record(event), ensure_ascii=False) + "\n")
            if isinstance(event, TextMessage):
                text = str(event.content).replace("\n", " ").strip()
                print(f"[{event.source}] {text[:260]}")
    return events


def last_text(events: Iterable[Any], source: str) -> str:
    """取某个 Agent 最后一次文本消息。"""

    for event in reversed(list(events)):
        if isinstance(event, TextMessage) and getattr(event, "source", None) == source:
            return str(event.content)
    return ""


def remove_markers(text: str) -> str:
    """删除控制工作流的哨兵词，不把它们放进最终报告。"""

    for marker in (
        "BASIC_REPORT_DONE",
        "REPORT_DRAFT_DONE",
        "REVIEW_APPROVED",
        "REVISION_REQUIRED",
    ):
        text = text.replace(marker, "")
    return text.strip()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_run_summary(
    out_dir: Path,
    *,
    kind: str,
    topic: str,
    report: str,
    evidence: dict[str, Any],
    settings: dict[str, Any],
    evidence_evaluation: str | None = None,
    review: str | None = None,
) -> None:
    """保存报告、证据注册表和不含密钥的运行摘要。"""

    (out_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    save_json(out_dir / "evidence_registry.json", evidence)
    safe_settings = {
        "base_url": settings["base_url"],
        "model": settings["model"],
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
    }
    summary: dict[str, Any] = {"kind": kind, "topic": topic, "settings": safe_settings}
    if evidence_evaluation is not None:
        summary["evidence_evaluation"] = evidence_evaluation
    if review is not None:
        summary["review"] = review
    save_json(out_dir / "run_summary.json", summary)

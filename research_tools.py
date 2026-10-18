"""给调研 Agent 使用的轻量工具。

工具包含一个随项目提供的课程资料库，以及一个不需要额外密钥的
Crossref 实时检索。实时检索失败时会明确返回错误，不会伪造资料。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_ROOT / "data" / "course_sources.json"

_registry: dict[str, dict[str, Any]] = {}
_url_to_id: dict[str, str] = {}
""" Storage of the approved sources. """
_approved_sources: set[str] = set()


def _load_course_sources() -> list[dict[str, Any]]:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def reset_registry() -> None:
    """每次运行前清空证据编号，保证报告中的 [S1] 可复现。"""

    _registry.clear()
    _url_to_id.clear()
    _approved_sources.clear()


def get_registry() -> dict[str, dict[str, Any]]:
    return dict(_registry)


def _register(item: dict[str, Any]) -> str:
    url = item["url"]
    if url in _url_to_id:
        return _url_to_id[url]
    source_id = f"S{len(_registry) + 1}"
    record = dict(item)
    record["citation_id"] = source_id
    _registry[source_id] = record
    _url_to_id[url] = source_id
    return source_id


def _clean_html(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "").replace("\n", " ").strip()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+|[\u4e00-\u9fff]", text)}


def search_course_sources(query: str, limit: int = 5) -> str:
    """从课程资料库中按关键词返回最相关的资料。"""

    query_tokens = _tokens(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in _load_course_sources():
        haystack = " ".join(
            [item.get("title", ""), item.get("summary_cn", ""), item.get("keywords", "")]
        )
        score = len(query_tokens & _tokens(haystack))
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for score, item in scored[:limit] if score > 0]
    if not selected:
        selected = [item for _, item in scored[: min(limit, 3)]]
    return _format_records(selected, "课程资料库")


def search_crossref(query: str, limit: int = 4) -> str:
    """调用 Crossref 实时检索，返回标题、作者、年份、摘要和 DOI。

    Crossref 的摘要并不总是存在；缺少摘要时会如实标注。
    """

    try:
        response = httpx.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": limit, "select": "DOI,title,author,published,abstract,URL"},
            headers={"User-Agent": "ZJU-LLM-MAS-course/1.0 (educational use)"},
            timeout=20,
            follow_redirects=True,
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        records: list[dict[str, Any]] = []
        for item in items[:limit]:
            title = (item.get("title") or ["未提供标题"])[0]
            date_parts = (item.get("published") or {}).get("date-parts") or [[None]]
            year = date_parts[0][0] if date_parts[0] else None
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])
            )
            doi = item.get("DOI", "")
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            if not url:
                continue
            records.append(
                {
                    "title": title,
                    "year": year,
                    "authors": authors,
                    "summary_cn": _clean_html(item.get("abstract", ""))
                    or "Crossref 未提供摘要，请打开原文核对。",
                    "url": url,
                    "doi": doi,
                    "source_type": "Crossref 实时检索",
                    "keywords": query,
                }
            )
        return _format_records(records, "Crossref 实时检索") if records else "Crossref 未返回结果。"
    except Exception as exc:  # 实时检索是可选增强，不应让主流程失控
        return f"Crossref 实时检索失败：{type(exc).__name__}: {exc}。请使用课程资料库结果。"


def search_literature(query: str, limit: int = 5) -> str:
    """调研 Agent 的主工具：课程资料库 + Crossref 实时检索。"""

    local = search_course_sources(query, limit=limit)
    live = search_crossref(query, limit=min(limit, 4))
    return f"{local}\n\n{live}"

def evaluate_evidence(relevant_sources: dict[str, list[str]]) -> str:
    """ Checks if each research question has at least two relevant sources and stores them. """

    sources = get_registry()
    results = []
    approved = set()

    for question, source_ids in relevant_sources.items():
        valid_sources = [source_id for source_id in set(source_ids) if source_id in sources]

        approved.update(valid_sources)

        if len(valid_sources) >= 2:
            status = "SUFFICIENT"
        else:
            status = "INSUFFICIENT"

        results.append(
            f"研究问题：{question}\n"
            f"相关文献：{valid_sources}\n"
            f"文献数量：{len(valid_sources)}\n"
            f"状态：{status}"
        )

    _approved_sources.clear()
    _approved_sources.update(approved)

    return "\n\n".join(results)


def audit_citations(report: str) -> str:
    """检查报告是否引用了本次运行中实际返回的证据编号。"""

    cited = sorted(set(re.findall(r"\[(S\d+)\]", report)))
    unknown = [source_id for source_id in cited if source_id not in _registry]
    if not cited:
        return "未发现 [S#] 引用。请在事实性段落后添加本次检索返回的证据编号。"
    if unknown:
        return f"发现未知引用 {unknown}。可用编号：{sorted(_registry)}。"
    return (
        f"引用检查通过：报告使用 {len(cited)} 个已注册来源 {cited}。"
        "请继续人工核对原文与结论是否一致。"
    )

def modified_audit_citations(report: str) -> str:
    """ Check whether the report cites only approved evidence. """

    cited = sorted(set(re.findall(r"\[(S\d+)\]", report)))
    unknown = [source_id for source_id in cited if source_id not in _registry]
    unapproved = [source_id for source_id in cited if source_id not in _approved_sources]
    if not cited:
        return "未发现 [S#] 引用。请在事实性段落后添加本次检索返回的证据编号。"
    if unknown:
        return f"发现未知引用 {unknown}。可用编号：{sorted(_registry)}。"
    if unapproved:
        return f"发现未经证据评估认可的引用: {unapproved}。可用证据：{sorted(_approved_sources)}"
    return (
        f"引用检查通过：报告使用 {len(cited)} 个已注册来源 {cited}。"
    )


def _format_records(items: list[dict[str, Any]], heading: str) -> str:
    if not items:
        return f"{heading}没有可用结果。"
    lines = [f"{heading}："]
    for item in items:
        source_id = _register(item)
        lines.append(
            f"[{source_id}] {item.get('title', '未提供标题')} ({item.get('year', '年份未知')})\n"
            f"作者：{item.get('authors', '作者未知')}\n"
            f"摘要/说明：{item.get('summary_cn', '无')}\n"
            f"链接：{item.get('url', '')}"
        )
    return "\n".join(lines)


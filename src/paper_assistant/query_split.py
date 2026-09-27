"""Split wide research questions before evidence retrieval."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from src.libs.llm.base_llm import ChatResponse, Message

DEFAULT_MAX_SUBQUESTIONS = 3

_ENUMERATION_SEPARATORS = ("以及", "、", "，", ",", "；", ";")
_CONJUNCTION_SEPARATORS = ("和", "与")
_LIST_CUES = ("哪些", "哪几个", "有哪", "是什么", "什么", "包括", "列举")
_TRAILING_CUES = (
    "分别是什么",
    "分别有哪些",
    "分别包括哪些",
    "是什么",
    "有哪些",
    "有什么",
    "包括哪些",
)
_TRAILING_PUNCTUATION = "？?。.!！"
_MAX_ENUMERATION_ITEM_CHARS = 24
_WIDE_MARKERS = ("综述", "总结", "概述", "调研", "梳理", "归纳", "全景")
_WIDE_MIN_CHARS = 30
_JSON_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class ChatModel(Protocol):
    def chat(self, messages: list[Message], **kwargs: Any) -> ChatResponse: ...


class HasChunkId(Protocol):
    chunk_id: str


ItemT = TypeVar("ItemT", bound=HasChunkId)


def looks_wide(query: str) -> bool:
    """Return whether an unresolved question is worth a splitter-model call."""
    text = query.strip()
    return len(text) >= _WIDE_MIN_CHARS or any(marker in text for marker in _WIDE_MARKERS)


def rule_split_query(
    query: str, *, max_subquestions: int = DEFAULT_MAX_SUBQUESTIONS
) -> tuple[str, ...]:
    """Split an explicit list question, or return an empty tuple when uncertain."""
    if max_subquestions < 2:
        raise ValueError("max_subquestions must be at least two")
    text = " ".join(query.split())
    if not text:
        raise ValueError("query cannot be empty")

    trailing = ""
    if text[-1] in _TRAILING_PUNCTUATION:
        trailing, text = text[-1], text[:-1].strip()
    for cue in _TRAILING_CUES:
        if text.endswith(cue):
            text = text[: -len(cue)].strip()
            break

    cue_end = -1
    for cue in _LIST_CUES:
        position = text.rfind(cue)
        if position >= 0:
            cue_end = max(cue_end, position + len(cue))
    if cue_end <= 0:
        return ()

    items = _split_enumeration(text[cue_end:])
    if len(items) < 2:
        return ()
    if any(len(item) > _MAX_ENUMERATION_ITEM_CHARS for item in items):
        return ()
    if any(cue in item for item in items for cue in _LIST_CUES):
        return ()
    return tuple(
        f"{text[:cue_end]}{item}{trailing}"
        for item in _cap_items(items, max_subquestions)
    )


def parse_model_subquestions(
    content: Any, *, max_subquestions: int = DEFAULT_MAX_SUBQUESTIONS
) -> tuple[str, ...]:
    """Parse a JSON string array from the splitter model."""
    if max_subquestions < 2 or not isinstance(content, str):
        return ()
    text = _JSON_FENCE.sub("", content.strip()).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return ()
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list):
        return ()
    items = [item.strip() for item in payload if isinstance(item, str) and item.strip()]
    unique = list(dict.fromkeys(items))
    if len(unique) < 2:
        return ()
    return tuple(_cap_items(unique, max_subquestions))


def model_split_query(
    llm: ChatModel,
    query: str,
    *,
    max_subquestions: int = DEFAULT_MAX_SUBQUESTIONS,
) -> tuple[str, ...]:
    """Ask a model to split a question that deterministic rules could not split."""
    if max_subquestions < 2:
        raise ValueError("max_subquestions must be at least two")
    text = query.strip()
    if not text:
        raise ValueError("query cannot be empty")
    messages = [
        Message(
            role="system",
            content=_SPLIT_SYSTEM_PROMPT.format(limit=max_subquestions),
        ),
        Message(role="user", content=text),
    ]
    try:
        response = llm.chat(messages, temperature=0.0)
    except Exception:
        return ()
    return parse_model_subquestions(
        getattr(response, "content", ""), max_subquestions=max_subquestions
    )


def split_query(
    query: str,
    *,
    llm: ChatModel | None = None,
    max_subquestions: int = DEFAULT_MAX_SUBQUESTIONS,
) -> tuple[str, ...]:
    """Return focused sub-questions, or the original question as a safe fallback."""
    if max_subquestions < 2:
        raise ValueError("max_subquestions must be at least two")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query cannot be empty")
    text = query.strip()

    ruled = rule_split_query(text, max_subquestions=max_subquestions)
    if ruled:
        return ruled
    if llm is not None and looks_wide(text):
        modelled = model_split_query(llm, text, max_subquestions=max_subquestions)
        if modelled:
            return modelled
    return (text,)


def merge_topic_groups(groups: Sequence[Sequence[ItemT]], *, top_k: int) -> list[ItemT]:
    """Interleave topic rankings so one topic cannot consume the whole budget."""
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    populated = [list(group) for group in groups if group]
    if not populated:
        return []

    merged: list[ItemT] = []
    seen: set[str] = set()
    deepest = max(len(group) for group in populated)
    for position in range(deepest):
        for group in populated:
            if position >= len(group):
                continue
            item = group[position]
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            merged.append(item)
            if len(merged) == top_k:
                return merged
    return merged


def _split_enumeration(text: str) -> list[str]:
    pattern = "|".join(
        re.escape(separator)
        for separator in _ENUMERATION_SEPARATORS + _CONJUNCTION_SEPARATORS
    )
    return [part.strip() for part in re.split(pattern, text) if part.strip()]


def _cap_items(items: Sequence[str], max_subquestions: int) -> list[str]:
    if len(items) <= max_subquestions:
        return list(items)
    head = list(items[: max_subquestions - 1])
    return head + ["；".join(items[max_subquestions - 1 :])]


_SPLIT_SYSTEM_PROMPT = (
    "You rewrite one research question as at most {limit} focused search queries.\n"
    "The indexed papers are English research prose, so every query must be in "
    "English.\n"
    "Rules:\n"
    "- Split only when the question really asks several distinct things at once.\n"
    "- Add the aspect's own research vocabulary (for example \"datasets\" or "
    "\"open research problems\") so each query names the facet it looks for.\n"
    "- Write a short phrase, not a sentence, and never a literal translation.\n"
    "- Repeat the shared topic inside every query so it stands alone.\n"
    "- Do not answer the question or add topics that the user did not ask about.\n"
    "- Reply with a JSON array of strings and nothing else; reply [] when the "
    "question is already focused."
)

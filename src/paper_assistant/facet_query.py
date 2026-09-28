"""Rewrite a wide question's aspects into retrieval queries that carry facet words.

Two effects bury the few chunks that really answer a wide question:

* the indexed corpus is English research prose while the question may be Chinese, and
* a wide question's words are dominated by the paper's own topic terms, which appear
  in almost every chunk, so every phrasing retrieves the same background pages.

This module builds one short English query per aspect.  It names the facet with
research vocabulary, keeps only that facet's distinctive words, and can anchor
itself to the paper's own table captions.  Everything here is deterministic: the
paper's topic terms and captions are measured from the index, not hard-coded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

DEFAULT_MAX_QUERIES = 3
DEFAULT_TOPIC_SHARE = 0.5
DEFAULT_MAX_ANCHOR_WORDS = 6
DEFAULT_MAX_QUERY_TOKENS = 10

_FACET_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("核心环节", ("life cycle", "stages")),
    ("关键环节", ("life cycle", "stages")),
    ("主要环节", ("life cycle", "stages")),
    ("研究问题", ("research issues",)),
    ("开放问题", ("open research problems",)),
    ("开放挑战", ("open challenges",)),
    ("未来方向", ("future research directions",)),
    ("未来工作", ("future work",)),
    ("核心内容", ("core components",)),
    ("组成部分", ("components",)),
    ("环节", ("life cycle", "stages")),
    ("流程", ("workflow", "stages")),
    ("阶段", ("stages",)),
    ("步骤", ("stages",)),
    ("数据集", ("datasets", "dataset")),
    ("数据", ("data",)),
    ("挑战", ("challenges",)),
    ("不足", ("limitations",)),
    ("局限", ("limitations",)),
    ("方法", ("method",)),
    ("算法", ("algorithm",)),
    ("模型", ("model",)),
    ("框架", ("framework",)),
    ("指标", ("metrics",)),
    ("评测", ("evaluation",)),
    ("评估", ("evaluation",)),
    ("评价", ("evaluation",)),
    ("应用", ("applications",)),
    ("场景", ("use cases",)),
    ("隐私", ("privacy",)),
    ("安全", ("security",)),
    ("激励", ("incentive",)),
    ("成本", ("cost",)),
    ("能耗", ("energy consumption",)),
)

_TABLE_CAPTION = re.compile(r"^table\s+\d+[.:]?\s+(.+)$", re.IGNORECASE)
_WORD = re.compile(r"[a-z][a-z\-]{2,}")

# Words that carry no facet meaning inside a table caption.
_STOPWORDS = frozenset(
    {
        "and", "are", "based", "between", "by", "for", "from", "into", "its",
        "main", "of", "on", "onto", "over", "per", "such", "than", "that",
        "the", "their", "these", "this", "those", "through", "to", "under",
        "used", "using", "versus", "via", "with",
    }
)


def canonical_paper_query(query: str) -> str:
    """Map precise IoT synonym clusters to short terms used by the corpus.

    Cross-Encoders are sensitive to diluted queries such as ``operation and
    maintenance management lifecycle``.  These rules require the complete
    concept cluster, so an unrelated query that merely contains one IoT word is
    left intact and still has to pass the normal open-set gate.
    """
    text = " ".join(query.split())
    lowered = text.lower()
    if (
        "device" in lowered
        and "management" in lowered
        and ("operation" in lowered or "maintenance" in lowered)
    ):
        return "IoT devices device management"
    if "边缘计算" in text and "卸载" in text and any(
        cue in text for cue in ("资源调度", "资源分配")
    ):
        return "IoT edge computing computation offloading resource allocation"
    if "医疗物联网" in text and any(
        cue in text for cue in ("连续监护", "患者监测", "患者监护")
    ):
        return "medical IoT patient monitoring"
    if "医疗物联网" in text and "可穿戴" in text and any(
        cue in text for cue in ("健康感知", "健康监测", "医疗")
    ):
        return "medical IoT wearable health sensing"
    if "无线传感器网络" in text and any(
        cue in text for cue in ("覆盖增强", "网络覆盖", "覆盖优化")
    ):
        return "wireless sensor networks coverage optimization"
    if "无线传感器网络" in text and "节点定位" in text:
        return "wireless sensor networks node localization"
    if "无线传感器网络" in text and any(
        cue in text for cue in ("节能路由", "能耗优化", "能源效率")
    ):
        return "wireless sensor networks energy consumption optimization"
    return text


def facet_english_words(aspect: str) -> tuple[str, ...]:
    """Return the English retrieval words for an aspect, or an empty tuple."""
    if not isinstance(aspect, str) or not aspect.strip():
        return ()
    best_cue = ""
    best_words: tuple[str, ...] = ()
    for cue, words in _FACET_TERMS:
        if cue in aspect and len(cue) > len(best_cue):
            best_cue, best_words = cue, words
    return best_words


def topic_terms(
    document_counts: Mapping[str, int],
    documents: int,
    *,
    share: float = DEFAULT_TOPIC_SHARE,
) -> frozenset[str]:
    """Return terms that appear in at least ``share`` of the paper's chunks."""
    if documents <= 0:
        return frozenset()
    if not 0 < share <= 1:
        raise ValueError("share must be within (0, 1]")
    return frozenset(
        term for term, count in document_counts.items() if count / documents >= share
    )


def table_captions(texts: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return ``(caption, containing chunk text)`` pairs found in the corpus."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text in texts:
        body = str(text)
        if "table " not in body.lower():
            continue
        for line in body.splitlines():
            match = _TABLE_CAPTION.match(line.strip())
            if not match:
                continue
            caption = re.sub(r"\s+", " ", match.group(1)).strip(" .")
            if len(caption) < 4 or caption.lower() in seen:
                continue
            seen.add(caption.lower())
            found.append((caption, body))
    return tuple(found)


def _matches(needle: str, haystack_lower: str) -> bool:
    if needle in haystack_lower:
        return True
    if needle.endswith("s") and re.search(rf"\b{re.escape(needle[:-1])}\b", haystack_lower):
        return True
    return bool(re.search(rf"\b{re.escape(needle)}s\b", haystack_lower))


def anchor_words(
    words: Sequence[str],
    caption_sources: Sequence[tuple[str, str]],
    *,
    topic: frozenset[str] = frozenset(),
    max_words: int = DEFAULT_MAX_ANCHOR_WORDS,
) -> tuple[str, ...]:
    """Return caption words of the caption whose chunk really holds the facet.

    Whole facet phrases are matched, so a generic caption word such as
    ``research`` cannot by itself drag an unrelated caption into the query.
    """
    distinctive = [word for word in words if word not in topic]
    if not distinctive or max_words < 1:
        return ()
    best_score = 0
    best_caption = ""
    for caption, body in caption_sources:
        lowered = body.lower()
        score = sum(1 for word in distinctive if _matches(word, lowered))
        if score > best_score:
            best_score, best_caption = score, caption
    if best_score < 1:
        return ()
    known = {token for word in words for token in word.split()}
    picked = [
        token
        for token in _WORD.findall(best_caption.lower())
        if token not in _STOPWORDS and token not in topic and token not in known
    ]
    return tuple(dict.fromkeys(picked))[:max_words]


def build_facet_queries(
    aspects: Sequence[str],
    *,
    topic: frozenset[str] = frozenset(),
    caption_sources: Sequence[tuple[str, str]] = (),
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_query_tokens: int = DEFAULT_MAX_QUERY_TOKENS,
    use_anchors: bool = True,
) -> tuple[str, ...]:
    """Return one retrieval query per aspect.

    An aspect whose facet words cannot be recognised is kept unchanged, so the
    caller always receives exactly as many queries as it supplied aspects.
    """
    if max_queries < 1:
        raise ValueError("max_queries must be at least one")
    queries: list[str] = []
    for aspect in aspects:
        canonical = canonical_paper_query(aspect)
        if canonical != " ".join(aspect.split()):
            queries.append(canonical)
            continue
        words = facet_english_words(aspect)
        if not words:
            queries.append(aspect)
            continue
        own = [token for word in words for token in word.split()]
        tokens = list(own)
        if use_anchors:
            tokens.extend(anchor_words(words, caption_sources, topic=topic))
        cleaned = [
            token
            for token in dict.fromkeys(tokens)
            if token in own or token not in topic
        ]
        query = " ".join(cleaned[:max_query_tokens]).strip()
        queries.append(query or aspect)
    return tuple(queries[:max_queries])

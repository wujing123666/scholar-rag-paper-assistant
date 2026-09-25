"""An explainable paper-level BM25 retrieval baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import jieba

from src.paper_assistant.catalog import PaperCatalog, PaperProfile

_SPAN_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[._-][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*|[\u3400-\u9fff]+"
)
_CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]+")
_STOP_WORDS = {
    "一篇",
    "一下",
    "哪个",
    "哪篇",
    "帮我",
    "我记得",
    "有篇",
    "论文",
    "文章",
    "工作",
    "方法",
    "研究",
    "the",
    "and",
    "for",
    "with",
    "from",
}


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and English text consistently."""
    tokens: list[str] = []
    for span in _SPAN_PATTERN.findall(text):
        if _CHINESE_PATTERN.fullmatch(span):
            tokens.extend(token.strip() for token in jieba.lcut(span) if token.strip())
        else:
            tokens.append(span.lower())
    return [
        token.lower()
        for token in tokens
        if (len(token) >= 2 or token.isdigit()) and token.lower() not in _STOP_WORDS
    ]


@dataclass(frozen=True)
class PaperSearchResult:
    paper: PaperProfile
    score: float
    matched_terms: tuple[str, ...]


class PaperBM25Retriever:
    """Rank one profile per paper and collapse duplicate PDF versions."""

    def __init__(
        self,
        catalog: PaperCatalog,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.catalog = catalog
        self.k1 = k1
        self.b = b
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_lengths: dict[str, int] = {}
        self._document_frequency: Counter[str] = Counter()
        self._build_index()

    def _build_index(self) -> None:
        for profile in self.catalog.profiles:
            frequencies: Counter[str] = Counter()
            for text, weight in profile.weighted_fields():
                if text:
                    frequencies.update(
                        {term: count * weight for term, count in Counter(tokenize(text)).items()}
                    )
            self._term_frequencies[profile.paper_id] = frequencies
            self._document_lengths[profile.paper_id] = sum(frequencies.values())
            self._document_frequency.update(frequencies.keys())

        self._average_document_length = sum(self._document_lengths.values()) / len(
            self._document_lengths
        )

    def _idf(self, term: str) -> float:
        document_count = len(self.catalog)
        frequency = self._document_frequency.get(term, 0)
        # Positive Robertson/Sparck Jones IDF avoids negative scores for
        # common domain terms in a small personal library.
        return math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[PaperSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")

        query_terms = Counter(tokenize(query))
        scored: list[PaperSearchResult] = []
        for profile in self.catalog.profiles:
            frequencies = self._term_frequencies[profile.paper_id]
            document_length = self._document_lengths[profile.paper_id]
            score = 0.0
            matched: list[str] = []
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                matched.append(term)
                denominator = term_frequency + self.k1 * (
                    1
                    - self.b
                    + self.b * document_length / self._average_document_length
                )
                score += (
                    self._idf(term)
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                    * query_frequency
                )
            if score > 0:
                scored.append(
                    PaperSearchResult(
                        paper=profile,
                        score=score,
                        matched_terms=tuple(
                            sorted(set(matched), key=lambda term: (-self._idf(term), term))
                        ),
                    )
                )

        scored.sort(key=lambda result: (-result.score, result.paper.paper_id))
        return scored[: min(top_k, len(scored))]

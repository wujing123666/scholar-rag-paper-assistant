"""Page-aware PDF chunk indexing and retrieval for the paper assistant."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.paper_assistant.catalog import PaperCatalog, PaperProfile
from src.paper_assistant.inventory import sha256_file
from src.paper_assistant.retriever import tokenize

CHUNK_RECORD_PREFIX = "paper_chunk:"
CHUNK_SCHEMA_VERSION = 4

_QUERY_ALIASES = {
    "伪历史数据": "pseudo historical data",
    "切比雪夫图滤波": "Chebyshev graph filtering topology-aware",
    "卡尔曼滤波": "Kalman filter Kalman filtering Kalman fusion uncertainty estimates",
    "不确定性": "uncertainty estimate",
    "专用模型": "dedicated model dedicated prediction sigma D",
    "泛化模型": "generalized model generalized prediction sigma G",
    "双路径": "dual-path",
    "冷启动": "cold-start",
    "目标域": "target domain",
    "源域": "source domain",
    "强化学习": "reinforcement learning",
    "用户招募": "user recruitment",
    "预算保留": "budget retention",
    "期望最大化": "expectation-maximization EM",
    "观测数据": "observed data",
    "缺失数据": "missing data",
    "缺失值": "missing values",
    "潜变量": "latent variable",
    "历史数据": "historical data",
    "最相似": "matching similarity Frobenius norm",
    "主动学习": "active learning",
    "贝叶斯推断": "Bayesian inference",
    "感知区域": "sensing area selection",
    "隐私保护": "privacy-preserving",
    "确定性": "deterministic",
    "初始插补": "initial imputation preliminary estimate",
    "局部相关性": "local correlation",
    "演化": "evolutionary inference",
    "自适应系数": "adaptive coefficient",
    "已感知数据": "sensed data observed data",
    "全局": "global",
    "双流": "dual-stream",
    "时域": "temporal domain",
    "频域": "frequency domain",
    "交叉注意力": "cross-attention",
    "数据补全": "data completion",
    "数据插补": "data imputation",
    "群智感知": "crowdsensing",
    "扩散模型": "diffusion model",
    "残差": "residual",
    "融合": "fusion",
    "时空": "spatiotemporal",
    "预测": "prediction",
}

_NAMED_SECTION_PATTERN = re.compile(
    r"^(?:abstract|references|acknowledg(?:e)?ments?|appendix)$",
    re.IGNORECASE,
)
_ROMAN_SECTION_PATTERN = re.compile(r"^[IVXLC]+\.\s+[A-Z][A-Z0-9 ,&:/()\-]{2,100}$")
_LETTERED_SECTION_PATTERN = re.compile(r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,&:/()\-]{2,100}$")
_NUMBERED_SECTION_PATTERN = re.compile(
    r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z ,&:/()\-]{2,100}$"
)
_NOISE_PATTERNS = (
    re.compile(r"^IEEE TRANSACTIONS ON MOBILE COMPUTING\s+\d+$", re.IGNORECASE),
    re.compile(r"^authorized licensed use limited to", re.IGNORECASE),
    re.compile(r"^downloaded on .* from ieee xplore", re.IGNORECASE),
    re.compile(r"^see https?://", re.IGNORECASE),
    re.compile(r"^digital object identifier\b", re.IGNORECASE),
    re.compile(r"^©\s*\d{4}\s+IEEE\b", re.IGNORECASE),
    re.compile(r"^\d+$"),
)
_METHOD_DETAIL_PATTERN = re.compile(
    r"\b(algorithm|method(?:ology)?|defined as|calculation is as follows|"
    r"computed as follows|we formulate|算法|方法|定义为|计算如下|公式)\b",
    re.IGNORECASE,
)
_FORMULA_DETAIL_PATTERN = re.compile(
    r"\b(algorithm\s+\d+|defined as|calculation is as follows|computed as follows|"
    r"we formulate|算法\s*\d+|定义为|计算如下|公式)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PaperChunk:
    """A retrievable piece of one PDF page with stable source metadata."""

    chunk_id: str
    paper_id: str
    pdf_file: str
    page_number: int
    chunk_index: int
    section: str
    text: str
    source_hash: str

    def embedding_text(self, profile: PaperProfile) -> str:
        del profile
        fields = []
        if self.section:
            fields.append(f"章节：{self.section}")
        fields.append(f"正文：{self.text}")
        return "\n".join(fields)


@dataclass(frozen=True)
class ChunkSearchResult:
    chunk_id: str
    paper: PaperProfile
    pdf_file: str
    page_number: int
    chunk_index: int
    section: str
    text: str
    score: float
    dense_score: float
    sparse_score: float = 0.0
    routing_rank: int = 0
    rerank_score: float | None = None
    focus_rank: int = 0


@dataclass(frozen=True)
class ChunkIndexSync:
    papers: int
    chunks: int
    embedded: int
    reused: int
    deleted: int
    rebuilt: bool


def is_method_detail(result: ChunkSearchResult) -> bool:
    """Return whether a chunk contains method or formula-level detail."""
    return bool(_METHOD_DETAIL_PATTERN.search(f"{result.section}\n{result.text}"))


def is_formula_detail(result: ChunkSearchResult) -> bool:
    """Return whether a chunk contains algorithm or formula-level evidence."""
    return bool(_FORMULA_DETAIL_PATTERN.search(f"{result.section}\n{result.text}"))


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _is_noise_line(line: str) -> bool:
    if not line or any(pattern.search(line) for pattern in _NOISE_PATTERNS):
        return True
    visible = [character for character in line if not character.isspace()]
    letters = sum(character.isalpha() for character in visible)
    return len(visible) >= 12 and letters / len(visible) < 0.3


def _is_section_heading(line: str) -> bool:
    if len(line) > 110 or len(line.split()) > 14:
        return False
    return bool(
        _NAMED_SECTION_PATTERN.fullmatch(line)
        or _ROMAN_SECTION_PATTERN.fullmatch(line)
        or _LETTERED_SECTION_PATTERN.fullmatch(line)
        or _NUMBERED_SECTION_PATTERN.fullmatch(line)
    )


def _ordered_page_blocks(page: fitz.Page) -> list[str]:
    """Read ordinary two-column papers left column before right column."""
    blocks = [block for block in page.get_text("blocks") if int(block[6]) == 0]
    midpoint = page.rect.width / 2
    normalized = []
    for block in blocks:
        x0, y0, x1, _y1, raw_text = block[:5]
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", str(raw_text))
        text = _clean_line(text)
        if text:
            normalized.append((x0, y0, x1, text))

    narrow = [item for item in normalized if item[2] - item[0] < page.rect.width * 0.65]
    left_count = sum((item[0] + item[2]) / 2 < midpoint for item in narrow)
    right_count = len(narrow) - left_count
    if left_count < 2 or right_count < 2:
        return [item[3] for item in sorted(normalized, key=lambda item: (item[1], item[0]))]

    top_spanning = []
    left = []
    right = []
    bottom_spanning = []
    for x0, y0, x1, text in normalized:
        width = x1 - x0
        if width >= page.rect.width * 0.65:
            target = top_spanning if y0 < page.rect.height / 2 else bottom_spanning
        elif (x0 + x1) / 2 < midpoint:
            target = left
        else:
            target = right
        target.append((y0, x0, text))
    ordered = []
    for group in (top_spanning, left, right, bottom_spanning):
        ordered.extend(item[2] for item in sorted(group))
    return ordered


def _has_prose_quality(text: str) -> bool:
    visible = [character for character in text if not character.isspace()]
    letters = sum(character.isalpha() for character in visible)
    return len(text) >= 80 and letters >= 45 and letters / max(len(visible), 1) >= 0.45


def expand_chunk_query(query: str) -> str:
    """Append transparent English aliases for common Chinese research terms."""
    aliases = [alias for term, alias in _QUERY_ALIASES.items() if term in query]
    return f"{query}\nEnglish terminology: {'; '.join(aliases)}" if aliases else query


def evidence_query_variants(query: str, *, max_variants: int = 4) -> tuple[str, ...]:
    """Build focused query variants for multi-aspect method questions."""
    if max_variants < 1:
        raise ValueError("max_variants must be at least one")
    variants = [query]
    for term, alias in _QUERY_ALIASES.items():
        if term not in query:
            continue
        variants.append(
            f"{term} {alias}\nmethod details algorithm formula architecture process"
        )
        if len(variants) >= max_variants:
            break
    return tuple(dict.fromkeys(variants))


def select_focused_evidence(
    variant_rankings: list[list[ChunkSearchResult]],
) -> tuple[ChunkSearchResult, ...]:
    """Pick distinct method-heavy pages that represent focused subqueries."""
    selected: list[ChunkSearchResult] = []
    selected_ids: set[str] = set()
    selected_pages: set[tuple[str, int]] = set()
    focused_rankings = variant_rankings[1:]
    target_count = len(focused_rankings)

    def add_candidate(candidate: ChunkSearchResult) -> None:
        selected.append(candidate)
        selected_ids.add(candidate.chunk_id)
        selected_pages.add((candidate.paper.paper_id, candidate.page_number))

    for ranking in focused_rankings:
        candidate = next(
            (
                result
                for result in ranking
                if is_formula_detail(result)
                and result.chunk_id not in selected_ids
                and (result.paper.paper_id, result.page_number) not in selected_pages
            ),
            None,
        )
        if candidate is not None:
            add_candidate(candidate)

    for predicate in (is_formula_detail, is_method_detail, lambda _result: True):
        for rank in range(max((len(items) for items in focused_rankings), default=0)):
            for ranking in focused_rankings:
                if len(selected) >= target_count or rank >= len(ranking):
                    continue
                candidate = ranking[rank]
                if (
                    predicate(candidate)
                    and candidate.chunk_id not in selected_ids
                    and (candidate.paper.paper_id, candidate.page_number)
                    not in selected_pages
                ):
                    add_candidate(candidate)
            if len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break
    return tuple(selected)


def apply_paper_routing_prior(
    results: list[ChunkSearchResult],
    candidate_paper_ids: tuple[str, ...],
    *,
    routing_weight: float = 0.2,
) -> list[ChunkSearchResult]:
    """Fuse chunk similarity with the first-stage paper rank."""
    if not 0 <= routing_weight <= 1:
        raise ValueError("routing_weight must be between zero and one")
    ranks = {paper_id: rank for rank, paper_id in enumerate(candidate_paper_ids, start=1)}
    fused = []
    for result in results:
        routing_rank = ranks.get(result.paper.paper_id, len(ranks) + 1)
        score = (1 - routing_weight) * result.score + routing_weight / routing_rank
        fused.append(replace(result, score=score, routing_rank=routing_rank))
    return sorted(fused, key=lambda result: (-result.score, result.chunk_id))


def _repeated_lines(page_lines: list[list[str]]) -> set[str]:
    """Identify short headers or footers repeated across many pages."""
    page_count = len(page_lines)
    if page_count < 3:
        return set()
    counts: Counter[str] = Counter()
    for lines in page_lines:
        counts.update(set(line for line in lines if 4 <= len(line) <= 160))
    threshold = max(3, math.ceil(page_count * 0.6))
    return {line for line, count in counts.items() if count >= threshold}


def _page_sections(
    lines: list[str], current_section: str
) -> tuple[list[tuple[str, str]], str]:
    sections: list[tuple[str, str]] = []
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((current_section, text))
        body.clear()

    for line in lines:
        if _is_section_heading(line):
            flush()
            current_section = line
        else:
            body.append(line)
    flush()
    return sections, current_section


def extract_paper_chunks(
    profile: PaperProfile,
    pdf_path: str | Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
) -> list[PaperChunk]:
    """Extract page-bounded chunks while carrying the latest section heading."""
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Paper PDF does not exist: {path}")

    source_hash = sha256_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "。", "; ", "；", " ", ""],
        length_function=len,
    )
    with fitz.open(path) as document:
        raw_pages = [_ordered_page_blocks(page) for page in document]

    repeated = _repeated_lines(raw_pages)
    chunks: list[PaperChunk] = []
    current_section = ""
    global_index = 0
    for page_number, raw_lines in enumerate(raw_pages, start=1):
        lines = [
            line
            for line in raw_lines
            if line not in repeated and not _is_noise_line(line)
        ]
        sections, current_section = _page_sections(lines, current_section)
        page_chunk_index = 0
        for section, text in sections:
            if section.strip().lower() == "references":
                continue
            for fragment in splitter.split_text(text):
                clean_text = fragment.strip()
                if not _has_prose_quality(clean_text):
                    continue
                digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:12]
                chunk_id = (
                    f"{CHUNK_RECORD_PREFIX}{profile.paper_id}:"
                    f"p{page_number:04d}:c{page_chunk_index:03d}:{digest}"
                )
                chunks.append(
                    PaperChunk(
                        chunk_id=chunk_id,
                        paper_id=profile.paper_id,
                        pdf_file=path.name,
                        page_number=page_number,
                        chunk_index=global_index,
                        section=section,
                        text=clean_text,
                        source_hash=source_hash,
                    )
                )
                global_index += 1
                page_chunk_index += 1
    if not chunks:
        raise ValueError(f"No usable text chunks were extracted from {path.name}")
    return chunks


class PaperChunkRetriever:
    """Synchronize page-aware chunks to Chroma and retrieve source evidence."""

    name = "paper_chunk_dense"

    def __init__(
        self,
        catalog: PaperCatalog,
        inbox: str | Path,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
    ) -> None:
        self.catalog = catalog
        self.inbox = Path(inbox)
        self.embedding = embedding
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = str(
            getattr(embedding, "model", embedding.__class__.__qualname__)
        )
        self.embedding_dimension = embedding.get_dimension()
        self._query_embeddings: OrderedDict[str, list[float]] = OrderedDict()
        self.index_sync = self._sync_index()
        self._build_sparse_index()

    def _build_sparse_index(self) -> None:
        ids = self.vector_store.list_ids()
        records = self.vector_store.get_by_ids(ids) if ids else []
        self._search_records = {
            record["id"]: record for record in records if record and record.get("id")
        }
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_lengths: dict[str, int] = {}
        self._document_frequency: Counter[str] = Counter()
        for record_id, record in self._search_records.items():
            frequencies = Counter(tokenize(str(record.get("text", ""))))
            self._term_frequencies[record_id] = frequencies
            self._document_lengths[record_id] = sum(frequencies.values())
            self._document_frequency.update(frequencies.keys())
        self._average_document_length = (
            sum(self._document_lengths.values()) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def _sparse_search(
        self,
        query: str,
        paper_ids: tuple[str, ...],
        top_k: int,
    ) -> list[tuple[dict, float]]:
        query_terms = Counter(tokenize(query))
        allowed = set(paper_ids)
        document_count = len(self._search_records)
        scored: list[tuple[dict, float]] = []
        for record_id, record in self._search_records.items():
            metadata = record.get("metadata", {})
            if allowed and str(metadata.get("paper_id", "")) not in allowed:
                continue
            frequencies = self._term_frequencies[record_id]
            document_length = self._document_lengths[record_id]
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                frequency = self._document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1 + (document_count - frequency + 0.5) / (frequency + 0.5)
                )
                denominator = term_frequency + 1.5 * (
                    0.25 + 0.75 * document_length / self._average_document_length
                )
                score += (
                    inverse_frequency
                    * term_frequency
                    * 2.5
                    / denominator
                    * query_frequency
                )
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], item[0]["id"]))
        return scored[:top_k]

    def _primary_sources(self) -> dict[str, Path]:
        sources: dict[str, Path] = {}
        for profile in self.catalog.profiles:
            existing = [self.inbox / name for name in profile.pdf_files if (self.inbox / name).is_file()]
            if not existing:
                raise FileNotFoundError(
                    f"No registered PDF exists for paper_id={profile.paper_id!r}"
                )
            sources[profile.paper_id] = existing[0]
        return sources

    def _sync_index(self) -> ChunkIndexSync:
        sources = self._primary_sources()
        source_hashes = {paper_id: sha256_file(path) for paper_id, path in sources.items()}
        current_ids = set(self.vector_store.list_ids())
        records = self.vector_store.get_by_ids(sorted(current_ids)) if current_ids else []
        current = {record["id"]: record for record in records if record and record.get("id")}
        by_paper: dict[str, list[dict]] = {}
        for record in current.values():
            metadata = record.get("metadata", {})
            by_paper.setdefault(str(metadata.get("paper_id", "")), []).append(record)

        rebuilt = any(
            record.get("metadata", {}).get("embedding_model") != self.embedding_model
            or record.get("metadata", {}).get("embedding_dimension")
            != self.embedding_dimension
            for record in current.values()
        )
        changed_papers: list[str] = []
        reused = 0
        for paper_id in sources:
            paper_records = by_paper.get(paper_id, [])
            metadata = paper_records[0].get("metadata", {}) if paper_records else {}
            expected_count = int(metadata.get("paper_chunk_count", 0) or 0)
            unchanged = (
                not rebuilt
                and bool(paper_records)
                and metadata.get("source_hash") == source_hashes[paper_id]
                and metadata.get("chunk_size") == self.chunk_size
                and metadata.get("chunk_overlap") == self.chunk_overlap
                and metadata.get("chunk_schema_version") == CHUNK_SCHEMA_VERSION
                and expected_count == len(paper_records)
            )
            if unchanged:
                reused += len(paper_records)
            else:
                changed_papers.append(paper_id)

        stale_ids = [
            record_id
            for record_id, record in current.items()
            if rebuilt
            or str(record.get("metadata", {}).get("paper_id", "")) not in sources
            or str(record.get("metadata", {}).get("paper_id", "")) in changed_papers
        ]

        pending: list[tuple[PaperChunk, PaperProfile, str]] = []
        for paper_id in changed_papers:
            profile = self.catalog.get(paper_id)
            for chunk in extract_paper_chunks(
                profile,
                sources[paper_id],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            ):
                pending.append((chunk, profile, chunk.embedding_text(profile)))

        vectors = self.embedding.embed([item[2] for item in pending]) if pending else []
        if len(vectors) != len(pending):
            raise ValueError("Embedding provider returned an invalid chunk matrix")

        counts = Counter(chunk.paper_id for chunk, _, _ in pending)
        new_records = []
        for (chunk, _profile, _embedding_text), vector in zip(pending, vectors):
            if len(vector) != self.embedding_dimension:
                raise ValueError("Embedding provider returned a chunk vector with the wrong dimension")
            new_records.append(
                {
                    "id": chunk.chunk_id,
                    "vector": vector,
                    "document": chunk.text,
                    "metadata": {
                        "record_type": "paper_chunk",
                        "paper_id": chunk.paper_id,
                        "pdf_file": chunk.pdf_file,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "section": chunk.section,
                        "source_hash": chunk.source_hash,
                        "content_hash": hashlib.sha256(
                            chunk.text.encode("utf-8")
                        ).hexdigest(),
                        "paper_chunk_count": counts[chunk.paper_id],
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                        "embedding_model": self.embedding_model,
                        "embedding_dimension": self.embedding_dimension,
                    },
                }
            )

        new_ids = {record["id"] for record in new_records}
        obsolete_ids = sorted(set(stale_ids) - new_ids)
        if new_records:
            self.vector_store.upsert(new_records)
        if obsolete_ids:
            self.vector_store.delete(obsolete_ids)
        return ChunkIndexSync(
            papers=len(sources),
            chunks=reused + len(new_records),
            embedded=len(new_records),
            reused=reused,
            deleted=len(obsolete_ids),
            rebuilt=rebuilt,
        )

    def _search_single(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_ids: tuple[str, ...] = (),
    ) -> list[ChunkSearchResult]:
        expanded_query = expand_chunk_query(query)
        if expanded_query in self._query_embeddings:
            query_vector = self._query_embeddings.pop(expanded_query)
            self._query_embeddings[expanded_query] = query_vector
        else:
            vectors = self.embedding.embed([expanded_query], is_query=True)
            if len(vectors) != 1 or len(vectors[0]) != self.embedding_dimension:
                raise ValueError("Embedding provider returned an invalid query vector")
            query_vector = vectors[0]
            self._query_embeddings[expanded_query] = query_vector
            if len(self._query_embeddings) > 128:
                self._query_embeddings.popitem(last=False)
        filters = None
        if len(paper_ids) == 1:
            filters = {"paper_id": paper_ids[0]}
        elif paper_ids:
            filters = {"paper_id": {"$in": list(paper_ids)}}
        candidate_pool = min(max(top_k * 4, 20), self.index_sync.chunks)
        dense_matches = self.vector_store.query(
            query_vector, top_k=candidate_pool, filters=filters
        )
        sparse_matches = self._sparse_search(
            expanded_query, paper_ids, candidate_pool
        )
        dense_by_id = {str(match.get("id", "")): match for match in dense_matches}
        sparse_by_id = {
            str(record.get("id", "")): (record, score)
            for record, score in sparse_matches
        }
        max_sparse = max((score for _, score in sparse_matches), default=0.0)
        ranked_matches = []
        for record_id in set(dense_by_id) | set(sparse_by_id):
            dense = dense_by_id.get(record_id, {})
            sparse_record, sparse_score = sparse_by_id.get(record_id, ({}, 0.0))
            record = dense or sparse_record
            dense_score = float(dense.get("score", 0.0))
            normalized_sparse = sparse_score / max_sparse if max_sparse else 0.0
            combined_score = 0.72 * dense_score + 0.28 * normalized_sparse
            ranked_matches.append(
                (record, combined_score, dense_score, sparse_score)
            )
        ranked_matches.sort(key=lambda item: (-item[1], str(item[0].get("id", ""))))
        results: list[ChunkSearchResult] = []
        for match, combined_score, dense_score, sparse_score in ranked_matches[:top_k]:
            metadata = match.get("metadata", {})
            paper_id = str(metadata.get("paper_id", ""))
            if not paper_id:
                continue
            try:
                paper = self.catalog.get(paper_id)
            except KeyError:
                continue
            results.append(
                ChunkSearchResult(
                    chunk_id=str(match.get("id", "")),
                    paper=paper,
                    pdf_file=str(metadata.get("pdf_file", "")),
                    page_number=int(metadata.get("page_number", 0)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    section=str(metadata.get("section", "")),
                    text=str(match.get("text", "")),
                    score=combined_score,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                )
            )
        return results

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        paper_ids: tuple[str, ...] = (),
    ) -> list[ChunkSearchResult]:
        """Retrieve and fuse evidence for the original and focused subqueries."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        unknown = sorted(
            set(paper_ids) - {paper.paper_id for paper in self.catalog.profiles}
        )
        if unknown:
            raise ValueError(f"Unknown paper_id values: {', '.join(unknown)}")

        variants = evidence_query_variants(query)
        per_variant_k = min(max(top_k * 2, 10), self.index_sync.chunks)
        fused_scores: dict[str, float] = {}
        best_results: dict[str, ChunkSearchResult] = {}
        maximum_rrf = sum(1 / (60 + 1) for _ in variants)
        variant_rankings: list[list[ChunkSearchResult]] = []
        for variant_index, variant in enumerate(variants):
            variant_weight = 1.25 if variant_index == 0 else 1.0
            variant_results = self._search_single(
                variant, top_k=per_variant_k, paper_ids=paper_ids
            )
            variant_rankings.append(variant_results)
            for rank, result in enumerate(variant_results, start=1):
                fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                    variant_weight / (60 + rank)
                )
                current = best_results.get(result.chunk_id)
                if current is None or result.score > current.score:
                    best_results[result.chunk_id] = result

        focused_evidence = select_focused_evidence(variant_rankings)
        focused_ranks = {
            result.chunk_id: rank
            for rank, result in enumerate(focused_evidence, start=1)
        }
        ranked = []
        normalization = maximum_rrf + 0.25 / 61
        for chunk_id, result in best_results.items():
            normalized_rrf = fused_scores[chunk_id] / normalization
            ranked.append(
                replace(
                    result,
                    score=0.65 * result.score
                    + 0.35 * normalized_rrf
                    + (0.08 if chunk_id in focused_ranks else 0.0),
                    focus_rank=focused_ranks.get(chunk_id, 0),
                )
            )
        ranked.sort(key=lambda result: (-result.score, result.chunk_id))
        selected = ranked[:top_k]
        if focused_evidence:
            selected_ids = {result.chunk_id for result in selected}
            ranked_by_id = {result.chunk_id: result for result in ranked}
            reserved_ids: set[str] = set()
            for focused in focused_evidence[:top_k]:
                detail = ranked_by_id[focused.chunk_id]
                reserved_ids.add(detail.chunk_id)
                if detail.chunk_id in selected_ids:
                    continue
                replace_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if selected[index].chunk_id not in reserved_ids
                    ),
                    None,
                )
                if replace_index is None:
                    break
                selected_ids.remove(selected[replace_index].chunk_id)
                selected[replace_index] = detail
                selected_ids.add(detail.chunk_id)
            selected.sort(key=lambda result: (-result.score, result.chunk_id))
        return selected

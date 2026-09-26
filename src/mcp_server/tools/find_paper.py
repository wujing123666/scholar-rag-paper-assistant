"""MCP tool for finding a known paper from a fuzzy description."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp import types

from src.core.settings import resolve_path
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.fastembed_embedding import FastEmbedEmbedding
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.hybrid_retriever import PaperHybridRetriever
from src.paper_assistant.rejection import decide_retrieval
from src.paper_assistant.retriever import PaperBM25Retriever, PaperSearchResult

if TYPE_CHECKING:
    from src.libs.vector_store.base_vector_store import BaseVectorStore
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.paper_assistant.evaluation import PaperRetriever

logger = logging.getLogger(__name__)

TOOL_NAME = "find_paper"
TOOL_DESCRIPTION = """Find papers in the personal library from a fuzzy memory.

Use this tool when the user remembers a paper's topic, method, dataset, author,
year, venue, or another clue but not its exact title. It returns paper-level
candidates rather than individual text chunks and collapses duplicate PDF versions.
"""
TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A fuzzy description or remembered clue about the paper.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of paper candidates to return.",
            "default": 3,
            "minimum": 1,
            "maximum": 10,
        },
        "retriever": {
            "type": "string",
            "description": "Retrieval strategy. BM25 is fast and requires no model download.",
            "enum": ["bm25", "dense", "hybrid"],
            "default": "bm25",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


class FindPaperInputError(ValueError):
    """Raised when MCP arguments fail tool-level validation."""


class FindPaperTool:
    """Search the paper catalog with a lazily initialized retriever."""

    def __init__(
        self,
        catalog_path: str | Path = "data/papers/paper_catalog.csv",
        *,
        model: str = FastEmbedEmbedding.DEFAULT_MODEL,
        model_cache: str | Path = "data/models/fastembed",
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        chroma_mode: str = "local",
        chroma_path: str | Path = "data/db/chroma",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_ssl: bool = False,
        min_scores: dict[str, float] | None = None,
    ) -> None:
        self.catalog_path = resolve_path(catalog_path)
        self.model = model
        self.model_cache = resolve_path(model_cache)
        self._embedding = embedding
        self._vector_store = vector_store
        self.chroma_mode = chroma_mode
        self.chroma_path = resolve_path(chroma_path)
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.chroma_ssl = chroma_ssl
        self.min_scores = dict(min_scores or {})
        self._catalog: PaperCatalog | None = None
        self._catalog_signature: tuple[int, int] | None = None
        self._retrievers: dict[str, PaperRetriever] = {}
        self._init_lock = threading.RLock()

    def _current_catalog_signature(self) -> tuple[int, int]:
        stat = self.catalog_path.stat()
        return stat.st_mtime_ns, stat.st_size

    @property
    def catalog(self) -> PaperCatalog:
        with self._init_lock:
            signature = self._current_catalog_signature()
            if self._catalog is None or signature != self._catalog_signature:
                self._catalog = PaperCatalog.from_csv(self.catalog_path)
                self._catalog_signature = signature
                self._retrievers.clear()
            return self._catalog

    def _get_embedding(self) -> BaseEmbedding:
        if self._embedding is None:
            self._embedding = FastEmbedEmbedding(
                model=self.model,
                cache_dir=self.model_cache,
            )
        return self._embedding

    def _get_vector_store(self) -> BaseVectorStore:
        if self._vector_store is None:
            from src.libs.vector_store.chroma_store import ChromaStore

            self._vector_store = ChromaStore(
                persist_directory=self.chroma_path,
                collection_name="paper_profiles_v1",
                mode=self.chroma_mode,
                host=self.chroma_host,
                port=self.chroma_port,
                ssl=self.chroma_ssl,
            )
        return self._vector_store

    def _get_retriever(self, name: str) -> PaperRetriever:
        with self._init_lock:
            # Accessing the property detects catalog edits and clears stale retrievers.
            catalog = self.catalog
            if name in self._retrievers:
                return self._retrievers[name]
            if name == "bm25":
                retriever: PaperRetriever = PaperBM25Retriever(catalog)
            elif name == "dense":
                retriever = PaperDenseRetriever(
                    catalog, self._get_embedding(), self._get_vector_store()
                )
            elif name == "hybrid":
                sparse = self._get_retriever("bm25")
                dense = self._get_retriever("dense")
                if not isinstance(sparse, PaperBM25Retriever) or not isinstance(
                    dense, PaperDenseRetriever
                ):
                    raise RuntimeError("Hybrid paper retriever dependencies are invalid")
                retriever = PaperHybridRetriever(catalog, sparse, dense)
            else:
                raise FindPaperInputError(
                    "retriever must be one of: bm25, dense, hybrid"
                )
            self._retrievers[name] = retriever
            return retriever

    @staticmethod
    def _serialize_result(result: PaperSearchResult, rank: int) -> dict[str, Any]:
        paper = result.paper
        return {
            "rank": rank,
            "paper_id": paper.paper_id,
            "title": paper.display_title,
            "canonical_title": paper.canonical_title,
            "authors": list(paper.authors),
            "year": paper.year,
            "venue": paper.venue,
            "score": round(result.score, 6),
            "matched_terms": list(result.matched_terms),
            "method_summaries": list(paper.method_summaries),
            "pdf_files": list(paper.pdf_files),
        }

    def find_papers(
        self,
        query: str,
        *,
        top_k: int = 3,
        retriever: str = "bm25",
    ) -> dict[str, Any]:
        """Return structured paper candidates for one fuzzy query."""
        if not isinstance(query, str) or not query.strip():
            raise FindPaperInputError("query cannot be empty")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
            raise FindPaperInputError("top_k must be an integer between 1 and 10")
        if retriever not in {"bm25", "dense", "hybrid"}:
            raise FindPaperInputError(
                "retriever must be one of: bm25, dense, hybrid"
            )

        initialization_fallback = None
        try:
            paper_retriever = self._get_retriever(retriever)
        except Exception as error:
            if retriever != "hybrid":
                raise
            paper_retriever = self._get_retriever("bm25")
            initialization_fallback = (
                f"dense_initialization_unavailable:{type(error).__name__}"
            )
        threshold_key = (
            "bm25" if paper_retriever.name == "paper_bm25" else retriever
        )
        threshold = self.min_scores.get(threshold_key)
        decision = decide_retrieval(
            paper_retriever,
            query.strip(),
            top_k=top_k,
            min_score=threshold,
        )
        matches = decision.results
        fallback_reason = initialization_fallback or decision.fallback_reason
        return {
            "query": query.strip(),
            "retriever": retriever,
            "effective_retriever": decision.effective_retriever,
            "retriever_fallback": fallback_reason,
            "rejected": decision.rejected,
            "rejection_reason": decision.reason,
            "top_score": (
                round(decision.top_score, 6)
                if decision.top_score is not None
                else None
            ),
            "min_score": round(decision.min_score, 6),
            "result_count": len(matches),
            "results": [
                self._serialize_result(result, rank)
                for rank, result in enumerate(matches, start=1)
            ],
        }

    @staticmethod
    def format_response(payload: dict[str, Any]) -> str:
        results = payload["results"]
        if payload.get("rejected"):
            return (
                "当前论文库中没有找到足够可靠的匹配。"
                "请补充方法、数据集、作者、年份或期刊等线索后重试。"
            )
        if not results:
            return "当前论文库中没有找到包含这些关键词的候选论文。"
        lines = [
            f"找到 {len(results)} 篇候选论文（{payload['retriever']}）：",
        ]
        for item in results:
            lines.append(f"\n{item['rank']}. **{item['title']}** (`{item['paper_id']}`)")
            details = [value for value in (item["year"], item["venue"]) if value]
            if details:
                lines.append(f"   - 信息：{' · '.join(details)}")
            if item["matched_terms"]:
                lines.append(f"   - 命中词：{', '.join(item['matched_terms'])}")
            elif item["method_summaries"]:
                lines.append(f"   - 方法：{item['method_summaries'][0]}")
            lines.append(f"   - PDF：{', '.join(item['pdf_files'])}")
        return "\n".join(lines)

    async def execute(
        self,
        query: str,
        top_k: int = 3,
        retriever: str = "bm25",
    ) -> types.CallToolResult:
        """Execute without blocking the MCP event loop."""
        try:
            payload = await asyncio.to_thread(
                self.find_papers,
                query,
                top_k=top_k,
                retriever=retriever,
            )
            return types.CallToolResult(
                content=[
                    types.TextContent(type="text", text=self.format_response(payload))
                ],
                structuredContent=payload,
                isError=False,
            )
        except FindPaperInputError as error:
            logger.warning("find_paper rejected invalid input: %s", error)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"查找论文失败：{error}")],
                isError=True,
            )
        except FileNotFoundError:
            logger.exception("Paper catalog is not available: %s", self.catalog_path)
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text="查找论文失败：论文目录不可用，请检查服务端配置。",
                    )
                ],
                isError=True,
            )
        except (ValueError, RuntimeError):
            logger.exception("Paper retriever initialization failed")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text="查找论文失败：检索器暂时不可用，请检查服务端日志。",
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register ``find_paper`` with the MCP protocol handler."""
    tool = FindPaperTool()

    async def handler(
        query: str,
        top_k: int = 3,
        retriever: str = "bm25",
    ) -> types.CallToolResult:
        return await tool.execute(query=query, top_k=top_k, retriever=retriever)

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )

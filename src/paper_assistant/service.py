"""Reusable ScholarRAG application service for CLI and web clients."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.settings import load_settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm import LLMFactory
from src.libs.llm.resilient_llm import ResilientChatModel, RetryPolicy
from src.libs.vector_store.chroma_store import ChromaStore
from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.chunk_retriever import (
    ChunkSearchResult,
    PaperChunkRetriever,
    apply_paper_routing_prior,
)
from src.paper_assistant.claim_judgement_cache import (
    ClaimJudgementCache,
    open_claim_judgement_cache,
)
from src.paper_assistant.claim_safety import verify_and_filter_claims
from src.paper_assistant.dense_retriever import PaperDenseRetriever
from src.paper_assistant.facet_query import build_facet_queries
from src.paper_assistant.grounded_answer import (
    REFUSAL_TEXT,
    GroundedAnswer,
    answer_from_evidence,
)
from src.paper_assistant.hybrid_retriever import PaperHybridRetriever
from src.paper_assistant.query_split import merge_topic_groups, split_query
from src.paper_assistant.rejection import decide_retrieval
from src.paper_assistant.retriever import PaperBM25Retriever, PaperSearchResult

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_PAPER_COLLECTION = "paper_profiles_qwen_v4"
DEFAULT_CHUNK_COLLECTION = "paper_chunks_qwen_v4"


@dataclass(frozen=True)
class PaperAssistantConfig:
    """Configuration needed to build one reusable paper RAG service."""

    catalog_path: Path = Path("data/papers/paper_catalog.csv")
    inbox_path: Path = Path("data/papers/inbox")
    settings_path: Path = Path("config/settings.yaml")
    embedding_settings_path: Path | None = None
    fallback_settings_path: Path | None = None
    retriever: str = "hybrid"
    chroma_mode: str = "local"
    chroma_path: Path = Path("data/db/chroma")
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chunk_reranker: str = "none"
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_cache: Path = Path("data/models/fastembed")
    rerank_candidates: int = 14
    rerank_weight: float = 0.35
    candidate_papers: int = 3
    top_k: int = 7
    chunk_size: int = 1200
    chunk_overlap: int = 180
    min_score: float | None = None
    disable_rejection: bool = False
    max_context_chars: int = 16000
    min_evidence_chunks: int = 1
    max_answer_claims: int = 10
    verify_claims: str = "off"
    claim_judge_models: tuple[str, ...] = ()
    claim_retry_k: int = 3
    verification_failure_policy: str = "strict"
    claim_second_judge_policy: str = "risk_based"
    claim_cache_path: Path = Path("data/cache/claim_judgements.sqlite3")
    disable_claim_cache: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def validate(self) -> None:
        if self.retriever not in {"bm25", "dense", "hybrid"}:
            raise ValueError("retriever must be bm25, dense, or hybrid")
        if self.chunk_reranker not in {"none", "fastembed"}:
            raise ValueError("chunk_reranker must be none or fastembed")
        if self.chroma_mode not in {"local", "server"}:
            raise ValueError("chroma_mode must be local or server")
        if self.verify_claims not in {"off", "single", "consensus"}:
            raise ValueError("verify_claims must be off, single, or consensus")
        if self.verification_failure_policy not in {"strict", "evidence_only"}:
            raise ValueError("invalid verification failure policy")
        if self.claim_second_judge_policy not in {"all", "risk_based"}:
            raise ValueError("invalid second judge policy")
        if self.candidate_papers < 1 or self.top_k < 1:
            raise ValueError("candidate_papers and top_k must be at least one")
        if self.rerank_candidates < 1 or self.claim_retry_k < 1:
            raise ValueError("rerank_candidates and claim_retry_k must be at least one")
        if not 0 <= self.rerank_weight <= 1:
            raise ValueError("rerank_weight must be between zero and one")
        if self.max_context_chars < 1000 or self.min_evidence_chunks < 1:
            raise ValueError("invalid answer evidence limits")
        if not 1 <= self.max_answer_claims <= 50:
            raise ValueError("max_answer_claims must be between one and 50")


def _build_qwen_embedding(config: PaperAssistantConfig) -> BaseEmbedding:
    """Create the single supported ScholarRAG embedding provider."""
    settings_path = config.embedding_settings_path or config.settings_path
    settings = load_settings(settings_path)
    if settings.embedding.provider.casefold() != "qwen":
        raise ValueError(
            "ScholarRAG requires embedding.provider=qwen so indexing and queries "
            "use the same production embedding model"
        )
    return EmbeddingFactory.create(settings)


@dataclass(frozen=True)
class PaperSearchResponse:
    query: str
    requested_retriever: str
    effective_retriever: str
    fallback_reason: str | None
    rejected: bool
    reason: str
    top_score: float | None
    min_score: float | None
    results: tuple[PaperSearchResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retriever": self.requested_retriever,
            "effective_retriever": self.effective_retriever,
            "retriever_fallback": self.fallback_reason,
            "rejected": self.rejected,
            "reason": self.reason,
            "top_score": (
                round(self.top_score, 6) if self.top_score is not None else None
            ),
            "min_score": self.min_score,
            "results": [
                {
                    "rank": rank,
                    "paper_id": result.paper.paper_id,
                    "title": result.paper.display_title,
                    "canonical_title": result.paper.canonical_title,
                    "score": round(result.score, 6),
                    "matched_terms": list(result.matched_terms),
                    "authors": list(result.paper.authors),
                    "year": result.paper.year,
                    "venue": result.paper.venue,
                    "tags": list(result.paper.tags),
                    "pdf_files": list(result.paper.pdf_files),
                }
                for rank, result in enumerate(self.results, start=1)
            ],
        }


@dataclass(frozen=True)
class PaperAnswerResponse:
    query: str
    candidate_paper_ids: tuple[str, ...]
    answer: GroundedAnswer
    evidence_results: tuple[ChunkSearchResult, ...]
    paper_retriever: str
    paper_retriever_fallback: str | None
    chunk_reranker: str
    reranker_model: str | None
    reranker_fallback: str | None
    index_sync: dict[str, Any]
    claim_verification: dict[str, Any]

    def to_dict(self, *, include_evidence: bool = False) -> dict[str, Any]:
        payload = {
            "query": self.query,
            "paper_retriever": self.paper_retriever,
            "paper_retriever_fallback": self.paper_retriever_fallback,
            "chunk_reranker": self.chunk_reranker,
            "reranker_model": self.reranker_model,
            "reranker_fallback": self.reranker_fallback,
            "candidate_papers": list(self.candidate_paper_ids),
            "index_sync": self.index_sync,
            **self.answer.to_dict(),
            "claim_verification": self.claim_verification,
        }
        if include_evidence:
            payload["evidence"] = [
                {
                    "chunk_id": result.chunk_id,
                    "paper_id": result.paper.paper_id,
                    "paper_title": result.paper.display_title,
                    "pdf_file": result.pdf_file,
                    "page_number": result.page_number,
                    "section": result.section,
                    "score": round(result.score, 6),
                    "text": result.text,
                }
                for result in self.evidence_results
            ]
        return payload


@dataclass(frozen=True)
class PaperAnswerDependencies:
    """Dependencies that are expensive and only needed after paper routing."""

    chunk_retriever: Any
    chunk_reranker: Any | None
    generation_llm: Any
    judge_llm: Any
    llm_provider: str
    llm_model: str
    llm_base_url: str
    claim_cache: ClaimJudgementCache | None
    claim_cache_fallback: str | None


class PaperAssistantService:
    """Coordinate paper routing, evidence retrieval, generation, and verification."""

    def __init__(
        self,
        *,
        catalog: PaperCatalog,
        paper_retriever: Any,
        requested_retriever: str,
        paper_retriever_fallback: str | None = None,
        chunk_retriever: Any | None = None,
        chunk_reranker: Any | None = None,
        generation_llm: Any | None = None,
        judge_llm: Any | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        claim_cache: ClaimJudgementCache | None = None,
        claim_cache_fallback: str | None = None,
        answer_dependency_loader: Callable[[], PaperAnswerDependencies] | None = None,
        config: PaperAssistantConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.paper_retriever = paper_retriever
        self.requested_retriever = requested_retriever
        self.paper_retriever_fallback = paper_retriever_fallback
        self.chunk_retriever = chunk_retriever
        self.chunk_reranker = chunk_reranker
        self.generation_llm = generation_llm
        self.judge_llm = judge_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        self.claim_cache = claim_cache
        self.claim_cache_fallback = claim_cache_fallback
        self.answer_dependency_loader = answer_dependency_loader
        self._answer_dependency_lock = threading.Lock()
        self.config = config or PaperAssistantConfig(retriever=requested_retriever)

    def find_papers(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        disable_rejection: bool | None = None,
    ) -> PaperSearchResponse:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        limit = self.config.candidate_papers if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be at least one")
        raw = (
            self.config.disable_rejection
            if disable_rejection is None
            else disable_rejection
        )
        threshold = self.config.min_score if min_score is None else min_score
        if raw:
            search_with_diagnostics = getattr(
                self.paper_retriever, "search_with_diagnostics", None
            )
            if callable(search_with_diagnostics):
                diagnostics = search_with_diagnostics(query, top_k=limit)
                results = tuple(diagnostics.results)
                effective = str(diagnostics.effective_retriever)
                dynamic_fallback = diagnostics.fallback_reason
            else:
                results = tuple(self.paper_retriever.search(query, top_k=limit))
                effective = str(self.paper_retriever.name)
                dynamic_fallback = None
            return PaperSearchResponse(
                query,
                self.requested_retriever,
                effective,
                self.paper_retriever_fallback or dynamic_fallback,
                not results,
                "no_candidates" if not results else "rejection_disabled",
                results[0].score if results else None,
                None,
                results,
            )
        decision = decide_retrieval(
            self.paper_retriever,
            query,
            top_k=min(limit, len(self.catalog)),
            min_score=threshold,
        )
        return PaperSearchResponse(
            query,
            self.requested_retriever,
            decision.effective_retriever,
            self.paper_retriever_fallback or decision.fallback_reason,
            decision.rejected,
            decision.reason,
            decision.top_score,
            decision.min_score,
            decision.results,
        )

    def answer_question(
        self,
        query: str,
        *,
        paper_ids: tuple[str, ...] = (),
    ) -> PaperAnswerResponse:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        explicit_ids = tuple(dict.fromkeys(paper_ids))
        unknown = sorted(set(explicit_ids) - {item.paper_id for item in self.catalog.profiles})
        if unknown:
            raise ValueError(f"Unknown paper_id values: {', '.join(unknown)}")
        automatic_routing = not explicit_ids
        paper_fallback = self.paper_retriever_fallback
        if automatic_routing:
            routing = self.find_papers(query)
            candidate_ids = tuple(result.paper.paper_id for result in routing.results)
            paper_fallback = routing.fallback_reason
        else:
            candidate_ids = explicit_ids
        if not candidate_ids:
            return self._refusal_response(
                query,
                "no_candidate_papers",
                paper_fallback,
            )
        self._ensure_answer_dependencies()

        retrieval_k = (
            max(self.config.top_k, self.config.rerank_candidates)
            if self.chunk_reranker is not None
            else self.config.top_k
        )
        sub_questions = split_query(query, llm=self.generation_llm)
        reranker_fallback = None
        if len(sub_questions) > 1:
            evidence, reranker_fallback = self._multi_aspect_evidence(
                sub_questions,
                candidate_ids,
                retrieval_k,
                automatic_routing=automatic_routing,
            )
        else:
            evidence = self._search_for_question(
                query,
                candidate_ids,
                retrieval_k,
                automatic_routing=automatic_routing,
            )
            if self.chunk_reranker is not None:
                from src.paper_assistant.chunk_reranker import rerank_with_fallback

                evidence, reranker_fallback = rerank_with_fallback(
                    self.chunk_reranker,
                    query,
                    evidence[:retrieval_k],
                    top_k=self.config.top_k,
                )
            else:
                evidence = evidence[: self.config.top_k]

        answer = answer_from_evidence(
            self.generation_llm,
            query,
            evidence,
            max_context_chars=self.config.max_context_chars,
            min_evidence_chunks=self.config.min_evidence_chunks,
            max_claims=self.config.max_answer_claims,
        )
        verification: dict[str, Any] = {"mode": self.config.verify_claims}
        if self.config.verify_claims != "off" and answer.status == "answered":
            if self.judge_llm is None:
                raise RuntimeError("claim judge is not configured")
            judge_models = self._judge_models()
            original_chunk_ids = {result.chunk_id for result in evidence}

            def retrieve_more(
                claim_text: str, target_papers: tuple[str, ...], top_k: int
            ) -> list[ChunkSearchResult]:
                candidate_k = max(
                    self.config.rerank_candidates, top_k + len(evidence)
                )
                supplementary: list[ChunkSearchResult] = []
                for paper_id in target_papers:
                    supplementary.extend(
                        self.chunk_retriever.search(
                            claim_text,
                            top_k=candidate_k,
                            paper_ids=(paper_id,),
                        )
                    )
                supplementary = [
                    item
                    for item in supplementary
                    if item.chunk_id not in original_chunk_ids
                ]
                supplementary = apply_paper_routing_prior(
                    supplementary, target_papers
                )
                if self.chunk_reranker is not None and supplementary:
                    from src.paper_assistant.chunk_reranker import (
                        rerank_with_fallback,
                        select_rerank_candidates,
                    )

                    candidates = select_rerank_candidates(
                        supplementary, top_k=candidate_k
                    )
                    supplementary, _ = rerank_with_fallback(
                        self.chunk_reranker,
                        claim_text,
                        candidates,
                        top_k=top_k,
                    )
                return supplementary[:top_k]

            safety = verify_and_filter_claims(
                self.judge_llm,
                answer,
                evidence,
                judge_models=judge_models,
                retrieve_more=retrieve_more,
                retry_k=self.config.claim_retry_k,
                disable_thinking=self.llm_provider == "deepseek",
                failure_policy=self.config.verification_failure_policy,
                cache=self.claim_cache,
                cache_namespace=f"{self.llm_provider or ''}|{self.llm_base_url or ''}",
                second_judge_policy=self.config.claim_second_judge_policy,
            )
            answer = safety.answer
            evidence = list(safety.evidence_results)
            verification = {"mode": self.config.verify_claims, **safety.report}
            verification["cache_initialization_fallback"] = (
                self.claim_cache_fallback
            )
        elif self.config.verify_claims != "off":
            verification.update(
                {"status": "skipped", "reason": "answer_not_generated"}
            )
        return PaperAnswerResponse(
            query=query,
            candidate_paper_ids=candidate_ids,
            answer=answer,
            evidence_results=tuple(evidence),
            paper_retriever=self.requested_retriever,
            paper_retriever_fallback=paper_fallback,
            chunk_reranker=self.config.chunk_reranker,
            reranker_model=(
                self.config.reranker_model
                if self.config.chunk_reranker != "none"
                else None
            ),
            reranker_fallback=reranker_fallback,
            index_sync=asdict(self.chunk_retriever.index_sync),
            claim_verification=verification,
        )

    def _search_for_question(
        self,
        question: str,
        candidate_ids: tuple[str, ...],
        retrieval_k: int,
        *,
        automatic_routing: bool,
    ) -> list[ChunkSearchResult]:
        """Search one question across the already selected candidate papers."""
        if not automatic_routing:
            return list(
                self.chunk_retriever.search(
                    question, top_k=retrieval_k, paper_ids=candidate_ids
                )
            )
        evidence: list[ChunkSearchResult] = []
        for paper_id in candidate_ids:
            evidence.extend(
                self.chunk_retriever.search(
                    question, top_k=retrieval_k, paper_ids=(paper_id,)
                )
            )
        return apply_paper_routing_prior(evidence, candidate_ids)

    def _multi_aspect_evidence(
        self,
        sub_questions: tuple[str, ...],
        candidate_ids: tuple[str, ...],
        retrieval_k: int,
        *,
        automatic_routing: bool,
    ) -> tuple[list[ChunkSearchResult], str | None]:
        """Retrieve each aspect separately and interleave the ranked groups.

        Every aspect is first rewritten into a retrieval query that names its own
        facet.  A wide question's own words are mostly the paper's topic words,
        which match nearly every chunk and therefore bury the few chunks that
        answer one particular aspect.
        """
        groups: list[list[ChunkSearchResult]] = []
        fallback: str | None = None
        for query in self._facet_queries(sub_questions, candidate_ids):
            group = self._search_for_question(
                query,
                candidate_ids,
                retrieval_k,
                automatic_routing=automatic_routing,
            )
            if self.chunk_reranker is not None:
                from src.paper_assistant.chunk_reranker import rerank_with_fallback

                group, group_fallback = rerank_with_fallback(
                    self.chunk_reranker,
                    query,
                    group[:retrieval_k],
                    top_k=self.config.top_k,
                )
                fallback = fallback or group_fallback
            groups.append(group)
        return merge_topic_groups(groups, top_k=self.config.top_k), fallback

    def _facet_queries(
        self, sub_questions: tuple[str, ...], candidate_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Rewrite every aspect into a query that keeps its own facet words.

        The paper's topic terms and table captions are measured from the ready
        index, and ``build_facet_queries`` always returns one query per aspect, so
        an aspect whose facet words are not recognised is searched unchanged.
        """
        context = self.chunk_retriever.corpus_context(candidate_ids)
        return build_facet_queries(
            sub_questions,
            topic=context.topic_terms,
            caption_sources=context.caption_sources,
            max_queries=len(sub_questions),
        )

    def _ensure_answer_dependencies(self) -> None:
        if self.chunk_retriever is not None and self.generation_llm is not None:
            return
        with self._answer_dependency_lock:
            if self.chunk_retriever is not None and self.generation_llm is not None:
                return
            if self.answer_dependency_loader is None:
                raise RuntimeError("answer dependencies are not configured")
            dependencies = self.answer_dependency_loader()
            self.chunk_retriever = dependencies.chunk_retriever
            self.chunk_reranker = dependencies.chunk_reranker
            self.generation_llm = dependencies.generation_llm
            self.judge_llm = dependencies.judge_llm
            self.llm_provider = dependencies.llm_provider
            self.llm_model = dependencies.llm_model
            self.llm_base_url = dependencies.llm_base_url
            self.claim_cache = dependencies.claim_cache
            self.claim_cache_fallback = dependencies.claim_cache_fallback

    def _judge_models(self) -> list[str]:
        models = list(dict.fromkeys(self.config.claim_judge_models))
        if not models:
            if not self.llm_model:
                raise ValueError("claim judge model is not configured")
            models = [self.llm_model]
            if self.config.verify_claims == "consensus" and self.llm_provider == "deepseek":
                models.append("deepseek-v4-pro")
        if self.config.verify_claims == "single":
            return models[:1]
        if len(models) < 2:
            raise ValueError(
                "consensus verification requires at least two distinct judge models"
            )
        return models

    def _refusal_response(
        self, query: str, reason: str, paper_fallback: str | None
    ) -> PaperAnswerResponse:
        return PaperAnswerResponse(
            query=query,
            candidate_paper_ids=(),
            answer=GroundedAnswer(
                status="insufficient_evidence",
                answer=REFUSAL_TEXT,
                claims=(),
                citations=(),
                reason=reason,
                model=None,
                usage=None,
            ),
            evidence_results=(),
            paper_retriever=self.requested_retriever,
            paper_retriever_fallback=paper_fallback,
            chunk_reranker=self.config.chunk_reranker,
            reranker_model=(
                self.config.reranker_model
                if self.config.chunk_reranker != "none"
                else None
            ),
            reranker_fallback=None,
            index_sync=(
                asdict(self.chunk_retriever.index_sync)
                if self.chunk_retriever is not None
                else {}
            ),
            claim_verification={"mode": self.config.verify_claims, "status": "skipped"},
        )


def build_paper_assistant_service(
    config: PaperAssistantConfig,
) -> PaperAssistantService:
    """Build a service whose answer-only dependencies are loaded on first use."""
    config.validate()
    catalog = PaperCatalog.from_csv(config.catalog_path)
    embedding = None
    paper_fallback = None
    if config.retriever == "bm25":
        paper_retriever: Any = PaperBM25Retriever(catalog)
    else:
        try:
            embedding = _build_qwen_embedding(config)
            profile_store = ChromaStore(
                persist_directory=config.chroma_path,
                collection_name=DEFAULT_PAPER_COLLECTION,
                mode=config.chroma_mode,
                host=config.chroma_host,
                port=config.chroma_port,
                ssl=config.chroma_ssl,
            )
            dense = PaperDenseRetriever(catalog, embedding, profile_store)
            paper_retriever = (
                dense
                if config.retriever == "dense"
                else PaperHybridRetriever(
                    catalog, PaperBM25Retriever(catalog), dense
                )
            )
        except Exception as error:
            if config.retriever != "hybrid":
                raise
            paper_retriever = PaperBM25Retriever(catalog)
            paper_fallback = (
                f"dense_initialization_unavailable:{type(error).__name__}"
            )

    def load_answer_dependencies() -> PaperAnswerDependencies:
        answer_embedding = embedding or _build_qwen_embedding(config)
        chunk_store = ChromaStore(
            persist_directory=config.chroma_path,
            collection_name=DEFAULT_CHUNK_COLLECTION,
            mode=config.chroma_mode,
            host=config.chroma_host,
            port=config.chroma_port,
            ssl=config.chroma_ssl,
        )
        chunk_retriever = PaperChunkRetriever(
            catalog,
            config.inbox_path,
            answer_embedding,
            chunk_store,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunk_reranker = None
        if config.chunk_reranker == "fastembed":
            from src.paper_assistant.chunk_reranker import FastEmbedChunkReranker

            chunk_reranker = FastEmbedChunkReranker(
                config.reranker_model,
                cache_dir=config.reranker_cache,
                weight=config.rerank_weight,
            )

        settings = load_settings(config.settings_path)
        primary = LLMFactory.create(settings)
        judge_llm = ResilientChatModel(
            primary,
            primary_name=settings.llm.provider,
            policy=config.retry_policy,
        )
        fallback = None
        fallback_name = None
        fallback_model = None
        if config.fallback_settings_path is not None:
            fallback_settings = load_settings(config.fallback_settings_path)
            fallback = LLMFactory.create(fallback_settings)
            fallback_name = fallback_settings.llm.provider
            fallback_model = fallback_settings.llm.model
        generation_llm = ResilientChatModel(
            primary,
            primary_name=settings.llm.provider,
            fallback=fallback,
            fallback_name=fallback_name,
            fallback_model=fallback_model,
            policy=config.retry_policy,
        )
        claim_cache = None
        claim_cache_fallback = None
        if not config.disable_claim_cache and config.verify_claims != "off":
            claim_cache, claim_cache_fallback = open_claim_judgement_cache(
                config.claim_cache_path
            )
        return PaperAnswerDependencies(
            chunk_retriever=chunk_retriever,
            chunk_reranker=chunk_reranker,
            generation_llm=generation_llm,
            judge_llm=judge_llm,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            llm_base_url=settings.llm.base_url,
            claim_cache=claim_cache,
            claim_cache_fallback=claim_cache_fallback,
        )

    return PaperAssistantService(
        catalog=catalog,
        paper_retriever=paper_retriever,
        requested_retriever=config.retriever,
        paper_retriever_fallback=paper_fallback,
        answer_dependency_loader=load_answer_dependencies,
        config=config,
    )


def build_paper_search_service(
    config: PaperAssistantConfig,
) -> PaperAssistantService:
    """Build only paper-level retrieval for the lightweight search UI."""
    config.validate()
    catalog = PaperCatalog.from_csv(config.catalog_path)
    paper_fallback = None
    if config.retriever == "bm25":
        paper_retriever: Any = PaperBM25Retriever(catalog)
    else:
        try:
            embedding = _build_qwen_embedding(config)
            profile_store = ChromaStore(
                persist_directory=config.chroma_path,
                collection_name=DEFAULT_PAPER_COLLECTION,
                mode=config.chroma_mode,
                host=config.chroma_host,
                port=config.chroma_port,
                ssl=config.chroma_ssl,
            )
            dense = PaperDenseRetriever(catalog, embedding, profile_store)
            paper_retriever = (
                dense
                if config.retriever == "dense"
                else PaperHybridRetriever(
                    catalog, PaperBM25Retriever(catalog), dense
                )
            )
        except Exception as error:
            if config.retriever != "hybrid":
                raise
            paper_retriever = PaperBM25Retriever(catalog)
            paper_fallback = (
                f"dense_initialization_unavailable:{type(error).__name__}"
            )
    return PaperAssistantService(
        catalog=catalog,
        paper_retriever=paper_retriever,
        requested_retriever=config.retriever,
        paper_retriever_fallback=paper_fallback,
        config=config,
    )

"""One explicitly initialized RAG service per API process."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from src.paper_assistant.catalog import PaperCatalog
from src.paper_assistant.service import PaperAssistantService, build_paper_assistant_service
from src.web_api.library import Library
from src.web_api.read_only_store import ReadOnlyChromaStore
from src.web_api.schemas import AnswerResult, Citation, Claim, SearchResult, Verification
from src.web_api.settings import WebSettings


class Registry:
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings
        config = replace(settings.rag, read_only_indexes=True)
        # Validate all catalog paths before opening either collection.
        self.library = Library(config.catalog_path, config.inbox_path, config.chroma_path)
        self.stores: list[ReadOnlyChromaStore] = []

        def store_factory(**kwargs: Any) -> ReadOnlyChromaStore:
            store = ReadOnlyChromaStore(**kwargs)
            self.stores.append(store)
            return store

        try:
            self.service = build_paper_assistant_service(config, store_factory=store_factory)
            # Explicit startup preparation, never triggered by a GET endpoint.
            self.service._ensure_answer_dependencies()
            self.library.bind(settings.state_path / self.library.dataset_id)
        except BaseException:
            self.close()
            raise

    def answer(self, query: str, paper_ids: list[str]) -> AnswerResult:
        self.library.assert_unchanged()
        response = self.service.answer_question(query, paper_ids=tuple(paper_ids))
        answer = response.answer
        if answer.status == "insufficient_evidence" and answer.reason in {
            "llm_generation_failed", "invalid_model_output", "claim_judge_unavailable"
        }:
            # Provider/output failures are not proof that the paper lacks evidence.
            raise RuntimeError("Generation or verification service did not complete")
        status = {"answered": "answered", "insufficient_evidence": "refused",
                  "verification_unavailable": "evidence_only"}.get(answer.status)
        if status is None:
            raise ValueError("Unexpected answer state")
        verification = response.claim_verification.get("status", "skipped")
        verification = verification if verification in {"verified", "skipped"} else "unavailable"
        citations = []
        for citation in answer.citations:
            file_id = self.library.names[citation.paper_id, citation.pdf_file]
            citations.append(Citation(
                citation_id=citation.citation_id, paper_id=citation.paper_id,
                file_id=file_id, paper_title=citation.paper_title,
                page_number=citation.page_number, section=citation.section,
                chunk_id=citation.chunk_id, excerpt=citation.excerpt,
            ))
        # The adapter publishes the final filtered Claim objects without LLM rewriting.
        return AnswerResult(
            status=status, claims=[Claim(text=c.text, citations=list(c.citations)) for c in answer.claims],
            citations=citations, reason=answer.reason,
            verification=Verification(status=verification),
        )

    def search(self, query: str, paper_ids: list[str]) -> SearchResult:
        self.library.assert_unchanged()
        service = self.service
        if paper_ids:
            scoped_catalog = PaperCatalog(service.catalog.get(paper_id) for paper_id in paper_ids)
            service = PaperAssistantService(
                catalog=scoped_catalog,
                paper_retriever=ScopedPaperRetriever(service.paper_retriever, service.catalog, set(paper_ids)),
                requested_retriever=service.requested_retriever,
                paper_retriever_fallback=service.paper_retriever_fallback,
                paper_reranker=service.paper_reranker, config=service.config,
            )
        response = service.find_papers(query, top_k=3)
        results = [self.library.papers[r.paper.paper_id] for r in response.results
                   if not paper_ids or r.paper.paper_id in paper_ids]
        return SearchResult(papers=results, rejected=response.rejected or not results, reason=response.reason)

    def close(self) -> None:
        for store in self.stores:
            store.close()
        cache = getattr(getattr(self, "service", None), "claim_cache", None)
        if cache is not None and hasattr(cache, "close"):
            cache.close()


class ScopedPaperRetriever:
    """Filter shared-index candidates before the existing gate/reranker is applied."""

    def __init__(self, retriever: Any, full_catalog: PaperCatalog, allowed: set[str]) -> None:
        self.retriever, self.full_catalog, self.allowed = retriever, full_catalog, allowed
        self.name = retriever.name
        self.catalog = PaperCatalog(full_catalog.get(paper_id) for paper_id in allowed)

    def search_with_diagnostics(self, query: str, top_k: int = 3):
        method = getattr(self.retriever, "search_with_diagnostics", None)
        if callable(method):
            response = method(query, top_k=len(self.full_catalog))
            results, effective, fallback = response.results, response.effective_retriever, response.fallback_reason
        else:
            results, effective, fallback = self.retriever.search(query, top_k=len(self.full_catalog)), self.name, None
        return SimpleNamespace(
            results=[result for result in results if result.paper.paper_id in self.allowed][:top_k],
            effective_retriever=effective, fallback_reason=fallback,
        )

    def search(self, query: str, top_k: int = 3):
        return self.search_with_diagnostics(query, top_k).results

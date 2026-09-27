"""Tests for the reusable ScholarRAG application service."""

from __future__ import annotations

import json
import sys

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.catalog import PaperCatalog, PaperProfile
from src.paper_assistant.chunk_retriever import (
    ChunkCorpusContext,
    ChunkIndexSync,
    ChunkSearchResult,
)
from src.paper_assistant.retriever import PaperSearchResult
from src.paper_assistant.service import (
    PaperAssistantConfig,
    PaperAssistantService,
)


def _paper(paper_id: str, title: str) -> PaperProfile:
    return PaperProfile(
        paper_id=paper_id,
        pdf_files=(f"{paper_id}.pdf",),
        canonical_title=title,
        title_zh=title,
        authors=("作者",),
        year="2026",
        venue="期刊",
        languages=("zh",),
        tags=("插补",),
        method_summaries=("方法摘要",),
        datasets=("数据集",),
        memory_cues=("记忆线索",),
        duplicate_groups=(),
    )


class _PaperRetriever:
    name = "paper_bm25"

    def __init__(self, catalog: PaperCatalog) -> None:
        self.catalog = catalog
        self.calls = []

    def search(self, query: str, top_k: int = 3):
        self.calls.append((query, top_k))
        return [
            PaperSearchResult(self.catalog.profiles[0], 12.0, ("插补",)),
            PaperSearchResult(self.catalog.profiles[1], 9.0, ("数据",)),
        ][:top_k]


class _ChunkRetriever:
    def __init__(self, catalog: PaperCatalog) -> None:
        self.catalog = catalog
        self.calls = []
        self.context_calls = []
        self.corpus = ChunkCorpusContext(frozenset(), ())
        self.index_sync = ChunkIndexSync(2, 2, 0, 2, 0, False)

    def corpus_context(self, paper_ids: tuple[str, ...] = ()) -> ChunkCorpusContext:
        self.context_calls.append(paper_ids)
        return self.corpus

    def search(self, query: str, *, top_k: int, paper_ids=()):
        self.calls.append((query, top_k, paper_ids))
        paper = self.catalog.get(paper_ids[0])
        return [
            ChunkSearchResult(
                chunk_id=f"{paper.paper_id}-chunk",
                paper=paper,
                pdf_file=paper.pdf_files[0],
                page_number=3,
                chunk_index=0,
                section="Method",
                text=f"{paper.display_title}使用扩散模型完成插补。",
                score=0.9,
                dense_score=0.8,
                sparse_score=1.0,
            )
        ]


class _GenerationLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ChatResponse(
            content=json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {"text": "论文使用扩散模型完成插补", "citations": ["C1"]}
                    ],
                },
                ensure_ascii=False,
            ),
            model="generator",
            usage={"total_tokens": 11},
        )


def _service() -> PaperAssistantService:
    catalog = PaperCatalog((_paper("paper-a", "论文A"), _paper("paper-b", "论文B")))
    config = PaperAssistantConfig(
        retriever="bm25",
        disable_rejection=False,
        min_score=0.0,
        candidate_papers=2,
        top_k=2,
    )
    return PaperAssistantService(
        catalog=catalog,
        paper_retriever=_PaperRetriever(catalog),
        requested_retriever="bm25",
        chunk_retriever=_ChunkRetriever(catalog),
        generation_llm=_GenerationLLM(),
        config=config,
    )


def test_find_papers_returns_ui_ready_metadata():
    result = _service().find_papers("帮我找插补论文")
    payload = result.to_dict()

    assert result.rejected is False
    assert payload["effective_retriever"] == "paper_bm25"
    assert payload["results"][0]["paper_id"] == "paper-a"
    assert payload["results"][0]["authors"] == ["作者"]
    assert payload["results"][0]["pdf_files"] == ["paper-a.pdf"]


def test_answer_question_runs_routing_retrieval_and_grounded_generation():
    service = _service()

    result = service.answer_question("这篇论文如何插补？")
    payload = result.to_dict(include_evidence=True)

    assert result.answer.status == "answered"
    assert result.candidate_paper_ids == ("paper-a", "paper-b")
    assert len(service.chunk_retriever.calls) == 2
    assert payload["claims"][0]["citations"] == ("C1",)
    assert payload["citations"][0]["page_number"] == 3
    assert payload["evidence"][0]["text"] == "论文A使用扩散模型完成插补。"


def test_answer_question_rule_splits_explicit_aspects_without_model_cost():
    service = _service()
    query = "这篇综述总结了哪些核心环节、常用数据集和开放问题？"

    result = service.answer_question(query)

    searched_questions = [call[0] for call in service.chunk_retriever.calls]
    assert result.answer.status == "answered"
    assert searched_questions == [
        "life cycle stages",
        "life cycle stages",
        "datasets dataset",
        "datasets dataset",
        "open research problems",
        "open research problems",
    ]
    assert service.chunk_retriever.context_calls == [("paper-a", "paper-b")]


def test_answer_question_anchors_each_aspect_to_the_paper_table_captions():
    service = _service()
    service.chunk_retriever.corpus = ChunkCorpusContext(
        topic_terms=frozenset({"sparse", "mobile", "crowdsensing", "data"}),
        caption_sources=(
            (
                "Overview of life cycle of MCS and SMCS",
                "Table 3 Overview of life cycle of MCS and SMCS\nFigure 2 Middleware",
            ),
        ),
    )

    service.answer_question("这篇综述总结了哪些核心环节、常用数据集和开放问题？")

    searched = {call[0] for call in service.chunk_retriever.calls}
    assert "life cycle stages overview mcs smcs" in searched
    assert "datasets dataset" in searched
    assert len(service.generation_llm.calls) == 1


def test_answer_question_uses_model_split_for_ambiguous_wide_question():
    class _SplitterAndGenerator:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if messages[0].content.startswith("You rewrite one research question"):
                return ChatResponse(
                    content='["子问题甲？", "子问题乙？"]', model="splitter"
                )
            return ChatResponse(
                content=json.dumps(
                    {
                        "status": "answered",
                        "claims": [{"text": "有证据的回答", "citations": ["C1"]}],
                    },
                    ensure_ascii=False,
                ),
                model="generator",
            )

    service = _service()
    service.generation_llm = _SplitterAndGenerator()

    result = service.answer_question(
        "这篇综述的核心环节、数据来源以及研究挑战分别是什么？"
    )

    searched_questions = [call[0] for call in service.chunk_retriever.calls]
    assert result.answer.status == "answered"
    assert searched_questions == ["子问题甲？", "子问题甲？", "子问题乙？", "子问题乙？"]
    assert len(service.generation_llm.calls) == 2


def test_answer_question_rejects_when_no_paper_passes_gate():
    service = _service()

    # This test double always returns candidates; raise the threshold to exercise refusal.
    service.config = PaperAssistantConfig(
        retriever="bm25", candidate_papers=2, min_score=99.0
    )
    result = service.answer_question("完全无关的问题", paper_ids=())

    assert result.answer.status == "insufficient_evidence"
    assert result.answer.reason == "no_candidate_papers"
    assert result.candidate_paper_ids == ()


def test_answer_question_does_not_load_expensive_dependencies_when_routing_rejects():
    catalog = PaperCatalog((_paper("paper-a", "论文A"), _paper("paper-b", "论文B")))
    calls = []

    def load_dependencies():
        calls.append("loaded")
        raise AssertionError("answer dependencies should stay lazy")

    service = PaperAssistantService(
        catalog=catalog,
        paper_retriever=_PaperRetriever(catalog),
        requested_retriever="bm25",
        answer_dependency_loader=load_dependencies,
        config=PaperAssistantConfig(retriever="bm25", min_score=99.0),
    )

    result = service.answer_question("完全无关的问题")

    assert result.answer.status == "insufficient_evidence"
    assert calls == []


def test_answer_question_validates_explicit_paper_ids():
    service = _service()

    try:
        service.answer_question("问题", paper_ids=("missing",))
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected an unknown paper id to fail")


def test_answer_cli_uses_shared_service(monkeypatch, capsys):
    captured = {}

    class _Response:
        def to_dict(self):
            return {"status": "answered", "answer": "共享服务回答"}

    class _Service:
        def answer_question(self, query, *, paper_ids=()):
            captured["query"] = query
            captured["paper_ids"] = paper_ids
            return _Response()

    def build(config):
        captured["config"] = config
        return _Service()

    monkeypatch.setattr(
        "src.paper_assistant.service.build_paper_assistant_service", build
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper-assistant",
            "--retriever",
            "bm25",
            "answer",
            "论文如何插补？",
            "--paper-id",
            "paper-a",
            "--settings",
            "config/settings.yaml",
        ],
    )
    from src.paper_assistant.__main__ import main

    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"] == "共享服务回答"
    assert captured["query"] == "论文如何插补？"
    assert captured["paper_ids"] == ("paper-a",)
    assert captured["config"].retriever == "bm25"
    assert captured["config"].max_answer_claims == 10
    assert captured["config"].claim_second_judge_policy == "risk_based"

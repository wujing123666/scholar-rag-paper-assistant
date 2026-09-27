"""ScholarRAG end-user page for paper discovery and grounded Q&A."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.paper_assistant.request_gate import (
    RequestExecutionError,
    RequestGate,
    RequestQueueFullError,
    RequestQueueWaitTimeoutError,
)
from src.paper_assistant.service import (
    PaperAnswerResponse,
    PaperAssistantConfig,
    PaperSearchResponse,
    build_paper_assistant_service,
    build_paper_search_service,
)

REQUEST_LOG_PATH = Path("logs/paper_requests.jsonl")


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


@st.cache_resource(show_spinner=False)
def _cached_search_service(config: PaperAssistantConfig):
    return build_paper_search_service(config)


@st.cache_resource(show_spinner=False)
def _cached_answer_service(config: PaperAssistantConfig):
    return build_paper_assistant_service(config)


@st.cache_resource(show_spinner=False)
def _cached_search_gate() -> RequestGate:
    return RequestGate(
        kind="paper_search",
        max_workers=_positive_env_int("SCHOLARRAG_SEARCH_WORKERS", 8),
        max_queue_size=_positive_env_int("SCHOLARRAG_MAX_QUEUE", 20),
        max_queue_wait_seconds=_positive_env_float(
            "SCHOLARRAG_QUEUE_WAIT_SECONDS", 60.0
        ),
        log_path=REQUEST_LOG_PATH,
    )


@st.cache_resource(show_spinner=False)
def _cached_answer_gate() -> RequestGate:
    return RequestGate(
        kind="paper_answer",
        max_workers=_positive_env_int("SCHOLARRAG_ANSWER_WORKERS", 3),
        max_queue_size=_positive_env_int("SCHOLARRAG_MAX_QUEUE", 20),
        max_queue_wait_seconds=_positive_env_float(
            "SCHOLARRAG_QUEUE_WAIT_SECONDS", 60.0
        ),
        log_path=REQUEST_LOG_PATH,
    )


def _build_config() -> PaperAssistantConfig:
    default_settings = os.getenv("SCHOLARRAG_SETTINGS", "config/settings.yaml")
    with st.expander("运行配置", expanded=False):
        settings_path = st.text_input(
            "LLM配置文件",
            value=default_settings,
            help="仅填写本地配置路径；页面不会显示或保存API Key。",
        )
        fallback_path = st.text_input(
            "备用生成模型配置（可选）",
            value=os.getenv("SCHOLARRAG_FALLBACK_SETTINGS", ""),
        )
        col_a, col_b = st.columns(2)
        with col_a:
            retriever = st.selectbox(
                "论文检索器", ("hybrid", "bm25", "dense"), index=0
            )
            reranker = st.selectbox(
                "正文重排", ("fastembed", "none"), index=0
            )
        with col_b:
            verification = st.selectbox(
                "Claim验证", ("consensus", "single", "off"), index=0
            )
            second_policy = st.selectbox(
                "第二评审策略", ("risk_based", "all"), index=0
            )
        max_answer_claims = st.number_input(
            "每次回答最多Claim数",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="限制细碎结论和后续评审成本；模型超出上限时只保留优先级最高的Claim。",
        )
        if st.button("重新加载模型与索引", key="paper_assistant_reload"):
            _cached_search_service.clear()
            _cached_answer_service.clear()
            st.success("缓存的服务实例已清除，下次查询会重新加载。")
    return PaperAssistantConfig(
        settings_path=Path(settings_path.strip() or default_settings),
        fallback_settings_path=(Path(fallback_path.strip()) if fallback_path.strip() else None),
        retriever=retriever,
        chunk_reranker=reranker,
        verify_claims=verification,
        claim_second_judge_policy=second_policy,
        max_answer_claims=int(max_answer_claims),
    )


def _render_search(response: PaperSearchResponse) -> None:
    payload = response.to_dict()
    if response.rejected:
        st.warning("没有论文通过当前置信度门槛。请补充方法名、任务或数据集等线索。")
        if response.top_score is not None:
            st.caption(
                f"最高分 {response.top_score:.4f}，门槛 {response.min_score:.4f}"
            )
        return
    st.success(f"找到 {len(response.results)} 篇候选论文")
    for item in payload["results"]:
        with st.container(border=True):
            st.subheader(f"{item['rank']}. {item['title']}")
            if item["canonical_title"] != item["title"]:
                st.caption(item["canonical_title"])
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("匹配分数", f"{item['score']:.4f}")
            col_b.metric("年份", item["year"] or "—")
            col_c.metric("发表渠道", item["venue"] or "—")
            st.write("作者：" + ("、".join(item["authors"]) or "未知"))
            if item["tags"]:
                st.write("标签：" + " · ".join(item["tags"]))
            st.caption("本地文件：" + "、".join(item["pdf_files"]))
    if response.fallback_reason:
        st.info(f"本次检索发生降级：{response.fallback_reason}")


def _render_request_record(record: dict | None) -> None:
    if not record:
        return
    st.caption(
        "请求调度："
        f"排队 {record.get('queue_wait_seconds', 0):.3f}s · "
        f"执行 {record.get('execution_seconds', 0):.3f}s · "
        f"总耗时 {record.get('total_seconds', 0):.3f}s"
    )


def _classify_search_result(
    response: PaperSearchResponse,
) -> tuple[str, str | None]:
    return ("rejected" if response.rejected else "matched", response.reason)


def _classify_answer_result(
    response: PaperAnswerResponse,
) -> tuple[str, str | None]:
    return response.answer.status, response.answer.reason


def _render_verification(report: dict) -> None:
    if report.get("mode") == "off":
        st.caption("Claim验证未启用。")
        return
    st.markdown("#### Claim验证")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("状态", report.get("status", "—"))
    col_b.metric("保留Claim", report.get("accepted_claims", 0))
    col_c.metric("高风险Claim", report.get("high_risk_claims", 0))
    col_d.metric("缓存命中", report.get("cache_hits", 0))
    if report.get("cache_errors"):
        st.warning(f"缓存错误 {report['cache_errors']} 次，已退回实时评审。")


def _render_answer(
    response: PaperAnswerResponse, request_record: dict | None = None
) -> None:
    if response.answer.status != "answered":
        st.warning(response.answer.answer)
        st.caption(f"原因：{response.answer.reason or 'unknown'}")
        if response.paper_retriever_fallback:
            st.info(f"论文检索降级：{response.paper_retriever_fallback}")
        _render_verification(response.claim_verification)
        _render_request_record(request_record)
        return

    st.success("已生成有原文依据的回答")
    st.markdown("### 回答")
    citation_by_id = {
        citation.citation_id: citation for citation in response.answer.citations
    }
    evidence_by_chunk = {
        result.chunk_id: result for result in response.evidence_results
    }
    for index, claim in enumerate(response.answer.claims, start=1):
        st.markdown(f"**{index}. {claim.text}**")
        for citation_id in claim.citations:
            citation = citation_by_id[citation_id]
            label = (
                f"{citation_id} · {citation.paper_title} · 第{citation.page_number}页"
                f" · {citation.section or '未标注章节'}"
            )
            with st.expander(label, expanded=False):
                evidence = evidence_by_chunk.get(citation.chunk_id)
                st.write(evidence.text if evidence is not None else citation.excerpt)
                st.caption(
                    f"文件：{citation.pdf_file} · Chunk：{citation.chunk_id}"
                )

    _render_verification(response.claim_verification)
    with st.expander("运行诊断", expanded=False):
        st.json(
            {
                "candidate_papers": list(response.candidate_paper_ids),
                "paper_retriever": response.paper_retriever,
                "paper_retriever_fallback": response.paper_retriever_fallback,
                "chunk_reranker": response.chunk_reranker,
                "reranker_fallback": response.reranker_fallback,
                "index_sync": response.index_sync,
                "generation_model": response.answer.model,
                "generation_usage": response.answer.usage,
                "service_diagnostics": response.answer.service_diagnostics,
                "request_queue": request_record,
            }
        )
    _render_request_record(request_record)


def render() -> None:
    """Render the ScholarRAG paper search and grounded-answer interface."""
    st.header("📚 ScholarRAG 论文助手")
    st.caption(
        "根据模糊记忆找回论文，或从论文正文中生成带页码、章节和原文依据的回答。"
    )
    search_gate = _cached_search_gate()
    answer_gate = _cached_answer_gate()
    search_state = search_gate.snapshot()
    answer_state = answer_gate.snapshot()
    st.caption(
        "单进程共享队列："
        f"找论文最多 {search_state.max_workers} 个并发"
        f"（执行 {search_state.active} / 等待 {search_state.queued}）；"
        f"论文问答最多 {answer_state.max_workers} 个并发"
        f"（执行 {answer_state.active} / 等待 {answer_state.queued}）；"
        f"每类最多等待 {answer_state.max_queue_size} 个请求。"
    )
    config = _build_config()
    search_tab, answer_tab = st.tabs(("🔎 找论文", "💬 论文问答"))

    with search_tab:
        with st.form("paper_search_form"):
            search_query = st.text_area(
                "描述你记得的论文",
                placeholder="例如：帮我找使用扩散模型和切比雪夫图卷积做插补的论文",
                height=100,
            )
            search_clicked = st.form_submit_button("开始检索", type="primary")
        if search_clicked:
            st.session_state.pop("paper_search_response", None)
            st.session_state.pop("paper_search_request_record", None)
            if not search_query.strip():
                st.warning("请先输入论文描述。")
            else:
                try:
                    with st.spinner("正在检索论文目录……"):
                        execution = search_gate.run(
                            _cached_search_service(config).find_papers,
                            search_query,
                            result_classifier=_classify_search_result,
                        )
                    st.session_state["paper_search_response"] = execution.value
                    st.session_state["paper_search_request_record"] = (
                        execution.record.to_dict()
                    )
                except RequestQueueFullError as error:
                    st.session_state["paper_search_request_record"] = (
                        error.record.to_dict()
                    )
                    st.warning("找论文请求队列已满，请稍后再试。")
                except RequestQueueWaitTimeoutError as error:
                    st.session_state["paper_search_request_record"] = (
                        error.record.to_dict()
                    )
                    st.warning("找论文请求排队超过允许时间，请稍后重试。")
                except RequestExecutionError as error:
                    st.session_state["paper_search_request_record"] = (
                        error.record.to_dict()
                    )
                    st.error(
                        f"论文检索初始化失败（{error.record.error_type}）。"
                        "请检查论文目录、向量索引和本地模型配置。"
                    )
                except Exception as error:
                    st.error(
                        f"论文检索初始化失败（{type(error).__name__}）。"
                        "请检查论文目录、向量索引和本地模型配置。"
                    )
        cached_search = st.session_state.get("paper_search_response")
        if isinstance(cached_search, PaperSearchResponse):
            _render_search(cached_search)
        _render_request_record(
            st.session_state.get("paper_search_request_record")
        )

    with answer_tab:
        with st.form("paper_answer_form"):
            answer_query = st.text_area(
                "输入论文知识问题",
                placeholder="例如：TCDI如何利用拓扑信息完成缺失数据插补？",
                height=120,
            )
            answer_clicked = st.form_submit_button("生成证据回答", type="primary")
        if answer_clicked:
            st.session_state.pop("paper_answer_response", None)
            st.session_state.pop("paper_answer_request_record", None)
            if not answer_query.strip():
                st.warning("请先输入问题。")
            else:
                try:
                    with st.spinner("正在检索正文、生成答案并核验Claim……"):
                        execution = answer_gate.run(
                            _cached_answer_service(config).answer_question,
                            answer_query,
                            result_classifier=_classify_answer_result,
                        )
                    st.session_state["paper_answer_response"] = execution.value
                    st.session_state["paper_answer_request_record"] = (
                        execution.record.to_dict()
                    )
                except RequestQueueFullError as error:
                    st.session_state["paper_answer_request_record"] = (
                        error.record.to_dict()
                    )
                    st.warning("论文问答请求队列已满，请稍后再试。")
                except RequestQueueWaitTimeoutError as error:
                    st.session_state["paper_answer_request_record"] = (
                        error.record.to_dict()
                    )
                    st.warning("论文问答请求排队超过允许时间，请稍后重试。")
                except RequestExecutionError as error:
                    st.session_state["paper_answer_request_record"] = (
                        error.record.to_dict()
                    )
                    st.error(
                        f"论文问答初始化或运行失败（{error.record.error_type}）。"
                        "请检查LLM私有配置、论文PDF和索引状态。"
                    )
                except Exception as error:
                    st.error(
                        f"论文问答初始化或运行失败（{type(error).__name__}）。"
                        "请检查LLM私有配置、论文PDF和索引状态。"
                    )
        cached_answer = st.session_state.get("paper_answer_response")
        if isinstance(cached_answer, PaperAnswerResponse):
            _render_answer(
                cached_answer,
                st.session_state.get("paper_answer_request_record"),
            )

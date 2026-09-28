"""Tests for rewriting a wide question's aspects into facet retrieval queries."""

from __future__ import annotations

import pytest

from src.paper_assistant.facet_query import (
    anchor_words,
    build_facet_queries,
    canonical_paper_query,
    facet_english_words,
    table_captions,
    topic_terms,
)

SURVEY_ASPECTS = (
    "这篇综述总结了哪些核心环节？",
    "这篇综述总结了哪些常用数据集？",
    "这篇综述总结了哪些开放问题？",
)

LIFE_CYCLE_CAPTION = (
    "Overview of life cycle of MCS and SMCS",
    "Table 3 Overview of life cycle of MCS and SMCS\nFigure 2 Middleware",
)


def test_facet_words_use_the_longest_matching_cue():
    assert facet_english_words("这篇综述总结了哪些核心环节？") == (
        "life cycle",
        "stages",
    )
    assert facet_english_words("常用数据集") == ("datasets", "dataset")
    assert facet_english_words("开放问题") == ("open research problems",)


def test_facet_words_ignore_unknown_blank_and_non_string_aspects():
    assert facet_english_words("这篇论文为什么有效？") == ()
    assert facet_english_words("   ") == ()
    assert facet_english_words("") == ()


def test_topic_terms_keep_only_terms_that_fill_half_the_chunks():
    counts = {"sparse": 5, "mobile": 5, "dataset": 2, "life": 1}

    assert topic_terms(counts, 5) == frozenset({"sparse", "mobile"})
    assert topic_terms(counts, 0) == frozenset()
    with pytest.raises(ValueError):
        topic_terms(counts, 5, share=0)


def test_table_captions_extract_unique_complete_captions():
    first = "Table 1 Some notable applications of MCS\nbody"
    duplicate = "Table 1 some notable applications of MCS\nother body"
    too_short = "Table 2 Ab\nbody"
    prose = "We show the Table 1 results in the appendix."

    assert table_captions([first, duplicate, too_short, prose]) == (
        ("Some notable applications of MCS", first),
    )


def test_anchor_words_pick_only_the_caption_that_holds_the_facet():
    words = ("life cycle", "stages")

    assert anchor_words(words, (LIFE_CYCLE_CAPTION,)) == (
        "overview",
        "mcs",
        "smcs",
    )
    assert anchor_words(("datasets", "dataset"), (LIFE_CYCLE_CAPTION,)) == ()
    assert anchor_words(words, ()) == ()
    assert anchor_words(words, (LIFE_CYCLE_CAPTION,), max_words=0) == ()


def test_anchor_words_drop_topic_words_stopwords_and_budget_the_length():
    assert (
        anchor_words(
            ("life cycle", "stages"),
            (LIFE_CYCLE_CAPTION,),
            topic=frozenset({"mcs"}),
            max_words=2,
        )
        == ("overview", "smcs")
    )


def test_build_facet_queries_returns_one_english_query_per_aspect():
    assert build_facet_queries(SURVEY_ASPECTS) == (
        "life cycle stages",
        "datasets dataset",
        "open research problems",
    )


def test_build_facet_queries_keep_aspects_without_a_known_facet():
    assert build_facet_queries(("这篇论文为什么有效？",)) == (
        "这篇论文为什么有效？",
    )
    assert build_facet_queries(("datasets used for sensing area",)) == (
        "datasets used for sensing area",
    )


def test_build_facet_queries_drop_topic_words_and_add_caption_anchors():
    topic = frozenset({"sparse", "mobile", "crowdsensing", "data"})

    assert build_facet_queries(
        ("这篇综述总结了哪些核心环节？",),
        topic=topic,
        caption_sources=(LIFE_CYCLE_CAPTION,),
    ) == ("life cycle stages overview mcs smcs",)


def test_build_facet_queries_honour_the_query_budgets():
    aspect = ("这篇综述总结了哪些核心环节？",)

    assert build_facet_queries(SURVEY_ASPECTS, max_queries=1) == (
        "life cycle stages",
    )
    assert build_facet_queries(aspect, max_query_tokens=2) == ("life cycle",)
    assert build_facet_queries(
        aspect, caption_sources=(LIFE_CYCLE_CAPTION,), use_anchors=False
    ) == ("life cycle stages",)
    with pytest.raises(ValueError):
        build_facet_queries(SURVEY_ASPECTS, max_queries=0)


def test_canonical_paper_query_normalizes_precise_iot_synonym_clusters():
    assert canonical_paper_query(
        "IoT device operation and maintenance management lifecycle"
    ) == "IoT devices device management"
    assert canonical_paper_query(
        "边缘计算场景下，物联网任务应如何卸载并进行计算资源调度？"
    ) == "IoT edge computing computation offloading resource allocation"
    assert canonical_paper_query(
        "医疗物联网有哪些患者连续监护方案？"
    ) == "medical IoT patient monitoring"
    assert canonical_paper_query(
        "医疗物联网有哪些可穿戴健康感知方案？"
    ) == "medical IoT wearable health sensing"
    assert canonical_paper_query(
        "无线传感器网络有哪些节能路由方法？"
    ) == "wireless sensor networks energy consumption optimization"


def test_canonical_paper_query_preserves_unrelated_iot_constraints():
    edge_unknown = "边缘计算利用CRISPR-Cas9进行任务卸载的论文"
    healthcare_unknown = "医疗物联网利用古埃及象形文字加密患者数据的论文"

    assert canonical_paper_query(edge_unknown) == edge_unknown
    assert canonical_paper_query(healthcare_unknown) == healthcare_unknown


def test_build_facet_queries_keeps_domain_topic_in_compound_facets():
    assert build_facet_queries(
        (
            "无线传感器网络有哪些覆盖增强？",
            "无线传感器网络有哪些节点定位？",
            "无线传感器网络有哪些节能路由方法？",
        ),
        use_anchors=False,
    ) == (
        "wireless sensor networks coverage optimization",
        "wireless sensor networks node localization",
        "wireless sensor networks energy consumption optimization",
    )

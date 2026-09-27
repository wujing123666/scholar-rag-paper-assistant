"""Tests for wide-question splitting and topic-coverage merging."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.paper_assistant.query_split import (
    looks_wide,
    merge_topic_groups,
    parse_model_subquestions,
    rule_split_query,
    split_query,
)

SURVEY_QUERY = "稀疏移动群智感知综述总结了哪些核心环节、常用数据集和开放问题？"


class FakeLLM:
    def __init__(self, content=None, *, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return ChatResponse(content=self.content, model="fake-model")


@dataclass
class _Item:
    chunk_id: str


def test_rule_split_repeats_the_shared_lead_in_for_every_aspect():
    parts = rule_split_query(SURVEY_QUERY)

    assert len(parts) == 3
    assert all(part.startswith("稀疏移动群智感知综述总结了哪些") for part in parts)
    assert parts == (
        "稀疏移动群智感知综述总结了哪些核心环节？",
        "稀疏移动群智感知综述总结了哪些常用数据集？",
        "稀疏移动群智感知综述总结了哪些开放问题？",
    )


def test_rule_split_keeps_one_question_about_one_method_whole():
    assert rule_split_query("HDDI 模型如何融合历史数据匹配和卡尔曼滤波？") == ()


def test_rule_split_declines_when_every_item_would_need_its_own_cue():
    assert rule_split_query("文中使用了什么数据集和什么方法？") == ()


def test_rule_split_caps_the_number_of_subquestions():
    parts = rule_split_query("综述总结了哪些A、B、C、D和E？")

    assert len(parts) == 3
    assert parts[-1].endswith("C；D；E？")


def test_looks_wide_only_flags_broad_questions():
    assert looks_wide(SURVEY_QUERY)
    assert not looks_wide("HDDI 模型如何融合预测？")


def test_split_query_falls_back_to_the_original_question_without_a_model():
    assert split_query("这篇论文提出了什么方法？") == ("这篇论文提出了什么方法？",)


def test_rule_split_does_not_spend_a_model_call():
    llm = FakeLLM('["不应调用", "不应调用"]')

    parts = split_query(SURVEY_QUERY, llm=llm)

    assert len(parts) == 3
    assert llm.calls == []


def test_split_query_asks_the_model_only_for_an_unsplittable_wide_question():
    llm = FakeLLM('["冷启动补全用了哪些数据集？", "冷启动补全有哪些开放问题？"]')

    parts = split_query("这篇综述的核心环节、常用数据集和开放问题分别是什么？", llm=llm)

    assert parts == ("冷启动补全用了哪些数据集？", "冷启动补全有哪些开放问题？")
    assert len(llm.calls) == 1


def test_split_prompt_asks_the_model_for_english_search_queries():
    llm = FakeLLM('["a", "b"]')

    split_query("这篇综述的核心环节、数据来源以及研究挑战分别是什么？", llm=llm)

    assert "English" in llm.calls[0][0][0].content


def test_split_query_does_not_spend_a_model_call_on_a_focused_question():
    llm = FakeLLM('["a", "b"]')

    parts = split_query("这篇论文提出了什么方法？", llm=llm)

    assert parts == ("这篇论文提出了什么方法？",)
    assert llm.calls == []


def test_split_query_keeps_the_original_question_when_the_model_fails():
    llm = FakeLLM(error=RuntimeError("provider down"))
    query = "这篇综述的核心环节、常用数据集和开放问题分别是什么？"

    assert split_query(query, llm=llm) == (query,)


def test_parse_model_subquestions_reads_fenced_json_and_rejects_junk():
    assert parse_model_subquestions('```json\n["问题一", "问题二"]\n```') == (
        "问题一",
        "问题二",
    )
    assert parse_model_subquestions("not json at all") == ()
    assert parse_model_subquestions('["only one"]') == ()
    assert parse_model_subquestions('["a", "b", "c", "d"]') == (
        "a",
        "b",
        "c；d",
    )


def test_merge_topic_groups_interleaves_topics_and_drops_duplicates():
    groups = [
        [_Item("a1"), _Item("a2"), _Item("a3")],
        [_Item("b1"), _Item("a2")],
        [_Item("c1")],
    ]

    merged = merge_topic_groups(groups, top_k=5)

    assert [item.chunk_id for item in merged] == ["a1", "b1", "c1", "a2", "a3"]


def test_merge_topic_groups_validates_and_honors_the_budget():
    groups = [[_Item("a1")], [_Item("b1")]]

    assert [item.chunk_id for item in merge_topic_groups(groups, top_k=1)] == ["a1"]
    assert merge_topic_groups([], top_k=3) == []
    with pytest.raises(ValueError):
        merge_topic_groups(groups, top_k=0)

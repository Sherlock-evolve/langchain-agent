from __future__ import annotations

import json
from collections import deque

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

import main as cli
import session_store
from agent import WorkspaceAgent
from audit_log import AuditLogError, JsonlAuditLogger
from contracts import (
    CitationPolicyEvent,
    CitationValidationEvent,
    EventEnvelope,
    MemoryUpdatedEvent,
    ModelCallMetricsEvent,
    SessionSavedEvent,
)
from knowledge_citation_validator import validate_knowledge_citations
from knowledge_retriever import RetrievedChunk
from knowledge_tools import create_search_knowledge_tool
from persistent_session import PersistentSession


class ScriptedModel:
    def __init__(self, responses, *, tools_enabled=False, shared=None):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.shared = shared if shared is not None else {"messages": []}

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
            shared=self.shared,
        )

    def stream(self, messages):
        self.shared["messages"].append(list(messages))
        if not self.responses:
            raise AssertionError("测试模型响应队列已耗尽")
        yield from self.responses.popleft()


class QueryRetriever:
    corpus_id = "d" * 64

    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, query, k=4, score_threshold=None):
        self.calls.append((query, k, score_threshold))
        return self.results_by_query.get(query, [])[:k]


def retrieved(
    chunk_id,
    source,
    start_line,
    end_line,
    content="knowledge body",
):
    return RetrievedChunk(
        rank=1,
        score=0.9,
        content=content,
        source=source,
        start_line=start_line,
        end_line=end_line,
        chunk_id=chunk_id,
    )


def payload_for(results):
    retriever = QueryRetriever({"payload": results})
    tool = create_search_knowledge_tool(retriever)
    return tool.invoke({"query": "payload", "k": max(1, len(results))})


def turn_messages(tool_payload, answer):
    return (
        HumanMessage(content="question"),
        ToolMessage(
            content=tool_payload,
            tool_call_id="citation-call",
            name="search_knowledge",
        ),
        AIMessage(content=answer),
    )


def test_online_validator_classifies_all_answer_statuses():
    chunk = retrieved(
        "known",
        "docs/known.md",
        3,
        5,
    )
    payload = payload_for([chunk])

    valid = validate_knowledge_citations(
        turn_messages(payload, "答案 docs/known.md:L3-L5")
    )
    missing = validate_knowledge_citations(
        turn_messages(payload, "没有引用")
    )
    unknown = validate_knowledge_citations(
        turn_messages(payload, "伪造 docs/fake.md:L1")
    )
    not_applicable = validate_knowledge_citations(
        (
            HumanMessage(content="question"),
            AIMessage(content="direct answer"),
        )
    )

    assert valid.status == "valid"
    assert valid.citation_count == 1
    assert valid.valid_citation_count == 1
    assert valid.retrieved_chunk_count == 1
    assert missing.status == "missing"
    assert unknown.status == "unknown"
    assert unknown.unknown_citation_count == 1
    assert not_applicable.status == "not_applicable"
    assert not_applicable.retrieved_chunk_count == 0


def test_agent_merges_multiple_searches_and_isolates_the_next_turn():
    retriever = QueryRetriever(
        {
            "first": [
                retrieved("first-id", "docs/first.md", 1, 2)
            ],
            "second": [
                retrieved("second-id", "docs/second.txt", 8, 8)
            ],
        }
    )
    search_tool = create_search_knowledge_tool(retriever)
    model = ScriptedModel(
        [
            [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "search_knowledge",
                            "args": '{"query":"first","k":1}',
                            "id": "first-call",
                            "index": 0,
                        },
                        {
                            "name": "search_knowledge",
                            "args": '{"query":"second","k":1}',
                            "id": "second-call",
                            "index": 1,
                        },
                    ],
                )
            ],
            [
                AIMessageChunk(
                    content=(
                        "合并 docs/first.md:L1-L2 "
                        "和 docs/second.txt:L8"
                    )
                )
            ],
            [
                AIMessageChunk(
                    content="下一轮提到 docs/first.md:L1-L2"
                )
            ],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[search_tool],
        citation_validator=validate_knowledge_citations,
        monotonic_clock=lambda: 0.0,
    )

    first_events = list(agent.stream_turn("执行两次检索"))
    second_events = list(agent.stream_turn("不要使用旧检索"))
    first_validation = next(
        event
        for event in first_events
        if isinstance(event, CitationValidationEvent)
    )
    second_validation = next(
        event
        for event in second_events
        if isinstance(event, CitationValidationEvent)
    )
    search_messages = [
        message
        for message in agent.messages
        if (
            isinstance(message, ToolMessage)
            and message.name == "search_knowledge"
        )
    ]

    assert first_validation.status == "valid"
    assert first_validation.citation_count == 2
    assert first_validation.retrieved_chunk_count == 2
    assert second_validation.status == "not_applicable"
    assert second_validation.citation_count == 0
    assert len(search_messages) == 2
    assert retriever.calls == [
        ("first", 1, None),
        ("second", 1, None),
    ]


def test_malformed_tampered_and_agent_truncated_json_only_warns():
    valid_payload = json.loads(
        payload_for(
            [
                retrieved(
                    "tampered-id",
                    "docs/tampered.md",
                    4,
                    6,
                )
            ]
        )
    )
    valid_payload["results"][0]["citation"] = "docs/other.md:L99"
    cases = [
        ("{invalid-json", 12_000),
        (
            json.dumps(valid_payload, ensure_ascii=False),
            12_000,
        ),
        (payload_for([retrieved("long-id", "docs/long.md", 1, 3)]), 80),
    ]

    for index, (tool_output, result_budget) in enumerate(cases):
        def search_knowledge(query: str, k: int = 4):
            return tool_output

        search_tool = StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description="返回测试知识 JSON，并要求回答使用 citation。",
        )
        model = ScriptedModel(
            [
                [
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": "search_knowledge",
                                "args": '{"query":"safe","k":1}',
                                "id": f"error-call-{index}",
                                "index": 0,
                            }
                        ],
                    )
                ],
                [AIMessageChunk(content="回答仍然提交")],
            ]
        )
        agent = WorkspaceAgent(
            model=model,
            tools=[search_tool],
            citation_validator=validate_knowledge_citations,
            max_tool_result_characters=result_budget,
            monotonic_clock=lambda: 0.0,
        )

        events = list(agent.stream_turn("触发校验错误"))
        validation = next(
            event
            for event in events
            if isinstance(event, CitationValidationEvent)
        )
        tool_message = next(
            message
            for message in agent.messages
            if isinstance(message, ToolMessage)
        )

        assert validation == CitationValidationEvent(
            status="error",
            citation_count=0,
            valid_citation_count=0,
            unknown_citation_count=0,
            retrieved_chunk_count=0,
        )
        assert agent.messages[-1].content == "回答仍然提交"
        assert tool_message.name == "search_knowledge"


def test_event_order_is_metrics_validation_memory_then_session_save(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    model = ScriptedModel(
        [[AIMessageChunk(content="当前回答")]]
    )
    summary_model = ScriptedModel(
        [[AIMessageChunk(content="新的长期摘要")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        summary_model=summary_model,
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
        citation_validator=validate_knowledge_citations,
        monotonic_clock=lambda: 0.0,
    )
    agent.messages.extend(
        [
            HumanMessage(content="被裁剪问题"),
            AIMessage(content="被裁剪回答"),
            HumanMessage(content="保留问题"),
            AIMessage(content="保留回答"),
        ]
    )
    session = PersistentSession(
        "citation-order",
        agent,
        turn_id_factory=lambda: "citation-order-turn",
    )

    stream = session.stream_turn("当前问题")
    envelopes = []
    while True:
        envelope = next(stream)
        envelopes.append(envelope)
        if isinstance(envelope.event, CitationValidationEvent):
            break

    assert agent.messages[-1].content == "当前回答"
    assert agent.memory_summary == "新的长期摘要"
    envelopes.extend(stream)
    event_types = [type(envelope.event) for envelope in envelopes]
    metrics_index = event_types.index(ModelCallMetricsEvent)
    citation_index = event_types.index(CitationValidationEvent)
    policy_index = event_types.index(CitationPolicyEvent)
    memory_index = event_types.index(MemoryUpdatedEvent)
    saved_index = event_types.index(SessionSavedEvent)

    assert (
        metrics_index
        < citation_index
        < policy_index
        < memory_index
        < saved_index
    )
    assert saved_index == len(envelopes) - 1
    assert envelopes[citation_index].event.status == "not_applicable"
    assert session_store.load("citation-order") == agent.export_snapshot()


def test_cli_and_audit_render_only_citation_status_and_counts(
    tmp_path,
    capsys,
):
    event = CitationValidationEvent(
        status="unknown",
        citation_count=2,
        valid_citation_count=1,
        unknown_citation_count=1,
        retrieved_chunk_count=3,
    )
    cli.start_turn()
    cli.render_event(event)
    cli.finish_turn()
    output = capsys.readouterr().out

    logger = JsonlAuditLogger(
        root=tmp_path / ".agent_audit",
        timestamp_factory=lambda: "2026-07-24T00:00:00Z",
    )
    logger.record(
        EventEnvelope(
            session_id="citation-audit",
            turn_id="citation-turn",
            sequence=1,
            event=event,
        )
    )
    raw_log = (
        tmp_path / ".agent_audit" / "citation-audit.jsonl"
    ).read_text(encoding="utf-8")
    record = json.loads(raw_log)

    assert output == "\n[引用] 检测到 1 个未知引用\n"
    assert record["data"] == {
        "status": "unknown",
        "citation_count": 2,
        "valid_citation_count": 1,
        "unknown_citation_count": 1,
        "retrieved_chunk_count": 3,
    }
    for sentinel in (
        "ANSWER-BODY-SECRET",
        "QUERY-SECRET",
        "docs/private.md",
        "docs/private.md:L1",
        '{"results":"TOOL-JSON-SECRET"}',
    ):
        assert sentinel not in output
        assert sentinel not in raw_log

    invalid_events = [
        CitationValidationEvent("forged", 0, 0, 0, 0),
        CitationValidationEvent("valid", -1, 0, 0, 0),
        CitationValidationEvent("valid", True, 0, 0, 0),
    ]
    for invalid_event in invalid_events:
        with pytest.raises(AuditLogError):
            logger.record(
                EventEnvelope(
                    session_id="citation-audit",
                    turn_id="citation-turn",
                    sequence=2,
                    event=invalid_event,
                )
            )

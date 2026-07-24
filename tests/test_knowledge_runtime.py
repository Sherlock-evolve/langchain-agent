from __future__ import annotations

import json
from collections import deque

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

import audit_log
import main as cli
import session_store
from agent import WorkspaceAgent
from knowledge_base import KnowledgeBuildResult, SkippedFileReport
from knowledge_retriever import RetrievedChunk
from knowledge_runtime import (
    KnowledgeConfigurationError,
    KnowledgeCorpusTruncatedError,
    KnowledgeEmbeddingError,
    KnowledgeRuntime,
    create_knowledge_runtime,
)
from knowledge_tools import create_search_knowledge_tool


class ScriptedModel:
    def __init__(
        self,
        responses,
        *,
        tools_enabled=False,
        shared=None,
    ):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.shared = shared if shared is not None else {
            "tools": [],
            "messages": [],
        }

    def bind_tools(self, tools):
        self.shared["tools"].append(list(tools))
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


class RecordingEmbeddings(Embeddings):
    def __init__(self, order=None):
        self.order = order
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        if self.order is not None:
            self.order.append("index")
        self.document_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [1.0, 0.0]


class StaticRetriever:
    def __init__(self, results, corpus_id="a" * 64):
        self.results = list(results)
        self.corpus_id = corpus_id

    def search(self, query, k=4, score_threshold=None):
        return self.results[:k]


def make_build_result(*, truncated=False):
    chunk = Document(
        page_content="RUNTIME-DOCUMENT-SECRET",
        metadata={
            "source": "docs/guide.md",
            "chunk_index": 0,
            "start_index": 0,
            "start_line": 2,
            "end_line": 4,
            "document_sha256": "document-sha",
            "chunk_id": "runtime-chunk-id",
        },
    )
    return KnowledgeBuildResult(
        chunks=[chunk],
        indexed_files=["docs/guide.md"],
        skipped_files=[
            SkippedFileReport(
                source="docs/ignored.bin",
                reason="unsupported_extension",
            )
        ],
        truncated=truncated,
    )


def make_search_tool(content="runtime knowledge"):
    return create_search_knowledge_tool(
        StaticRetriever(
            [
                RetrievedChunk(
                    rank=1,
                    score=0.9,
                    content=content,
                    source="docs/guide.md",
                    start_line=2,
                    end_line=4,
                    chunk_id="runtime-chunk-id",
                )
            ]
        )
    )


def set_inputs(monkeypatch, values):
    responses = iter(values)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(responses),
    )


def isolate_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    monkeypatch.setattr(
        audit_log,
        "AUDIT_LOG_ROOT",
        tmp_path / ".agent_audit",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)


def test_cli_default_mode_never_builds_knowledge(
    monkeypatch,
):
    runtime_calls = []

    class ExitOnlySession:
        session_id = "default"
        dirty = False

    def fake_open(cls, session_id, agent_factory):
        return ExitOnlySession()

    def forbidden_runtime(**kwargs):
        runtime_calls.append(kwargs)
        raise AssertionError("默认模式不得构建知识库")

    monkeypatch.setattr(
        cli.PersistentSession,
        "open",
        classmethod(fake_open),
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    set_inputs(monkeypatch, ["exit"])

    status = cli.main(
        ["--knowledge-directory", "../must-not-be-scanned"],
        knowledge_runtime_factory=forbidden_runtime,
    )

    assert status == 0
    assert runtime_calls == []


def test_enabled_runtime_builds_once_registers_tool_and_uses_fallbacks(
    tmp_path,
    monkeypatch,
    capsys,
):
    isolate_runtime_paths(tmp_path, monkeypatch)
    order = []
    embedding_arguments = []
    runtime_calls = []
    created_agents = []

    def knowledge_builder(**kwargs):
        order.append("build")
        assert kwargs["docs_directory"] == "manuals"
        return make_build_result()

    def embeddings_factory(**kwargs):
        order.append("embedding")
        embedding_arguments.append(kwargs)
        return RecordingEmbeddings(order)

    def runtime_factory(*, knowledge_directory):
        runtime_calls.append(knowledge_directory)
        return create_knowledge_runtime(
            workspace_root=tmp_path,
            knowledge_directory=knowledge_directory,
            environment={
                "EMBEDDING_MODEL": "embedding-model",
                "ZHIPU_API_KEY": "FALLBACK-KEY-SECRET",
                "ZHIPU_BASE_URL": "https://fallback.invalid/v1",
            },
            embeddings_factory=embeddings_factory,
            knowledge_builder=knowledge_builder,
        )

    def agent_factory(
        extra_tools=None,
        citation_validator=None,
        citation_policy="observe",
        citation_guard_tool_names=None,
    ):
        agent = WorkspaceAgent(
            model=ScriptedModel([]),
            tools=list(extra_tools or ()),
            citation_validator=citation_validator,
            citation_policy=citation_policy,
            citation_guard_tool_names=set(
                citation_guard_tool_names or ()
            ),
        )
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(cli, "create_workspace_agent", agent_factory)
    set_inputs(monkeypatch, ["exit"])

    status = cli.main(
        [
            "--session",
            "knowledge",
            "--enable-knowledge",
            "--knowledge-directory",
            "manuals",
        ],
        knowledge_runtime_factory=runtime_factory,
    )
    output = capsys.readouterr().out

    assert status == 0
    assert runtime_calls == ["manuals"]
    assert order == ["build", "embedding", "index"]
    assert embedding_arguments == [
        {
            "model": "embedding-model",
            "api_key": "FALLBACK-KEY-SECRET",
            "base_url": "https://fallback.invalid/v1",
        }
    ]
    assert len(created_agents) == 1
    assert set(created_agents[0].tools_by_name) == {"search_knowledge"}
    assert "[知识库] 已索引文件 1，分块 1，跳过文件 1" in output
    assert "RUNTIME-DOCUMENT-SECRET" not in output
    assert "docs/ignored.bin" not in output
    assert "FALLBACK-KEY-SECRET" not in output


def test_session_switches_share_tool_but_not_agent_history(
    tmp_path,
    monkeypatch,
):
    isolate_runtime_paths(tmp_path, monkeypatch)
    search_tool = make_search_tool()
    runtime = KnowledgeRuntime(
        search_tool=search_tool,
        corpus_id="b" * 64,
        indexed_file_count=1,
        chunk_count=1,
        skipped_file_count=0,
    )
    runtime_calls = []
    created_agents = []

    def runtime_factory(**kwargs):
        runtime_calls.append(kwargs)
        return runtime

    def agent_factory(
        extra_tools=None,
        citation_validator=None,
        citation_policy="observe",
        citation_guard_tool_names=None,
    ):
        responses = (
            [[AIMessageChunk(content="第一个会话回答")]]
            if not created_agents
            else []
        )
        agent = WorkspaceAgent(
            model=ScriptedModel(responses),
            tools=list(extra_tools or ()),
            citation_validator=citation_validator,
            citation_policy=citation_policy,
            citation_guard_tool_names=set(
                citation_guard_tool_names or ()
            ),
        )
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(cli, "create_workspace_agent", agent_factory)
    set_inputs(
        monkeypatch,
        ["第一个会话问题", ":switch second", "exit"],
    )

    status = cli.main(
        ["--session", "first", "--enable-knowledge"],
        knowledge_runtime_factory=runtime_factory,
    )

    assert status == 0
    assert len(runtime_calls) == 1
    assert len(created_agents) == 2
    assert (
        created_agents[0].tools_by_name["search_knowledge"]
        is search_tool
    )
    assert (
        created_agents[1].tools_by_name["search_knowledge"]
        is search_tool
    )
    assert [
        message.content
        for message in created_agents[0].messages
        if isinstance(message, HumanMessage)
    ] == ["第一个会话问题"]
    assert [type(message) for message in created_agents[1].messages] == [
        SystemMessage
    ]


def test_configuration_truncation_and_index_failures_stop_safely(
    tmp_path,
    monkeypatch,
    capsys,
):
    embedding_calls = []

    def embeddings_factory(**kwargs):
        embedding_calls.append(kwargs)
        return RecordingEmbeddings()

    with pytest.raises(KnowledgeConfigurationError):
        create_knowledge_runtime(
            workspace_root=tmp_path,
            environment={},
            knowledge_builder=lambda **kwargs: make_build_result(),
            embeddings_factory=embeddings_factory,
        )
    assert embedding_calls == []

    with pytest.raises(KnowledgeCorpusTruncatedError):
        create_knowledge_runtime(
            workspace_root=tmp_path,
            environment={
                "EMBEDDING_MODEL": "model",
                "EMBEDDING_API_KEY": "key",
            },
            knowledge_builder=lambda **kwargs: make_build_result(
                truncated=True
            ),
            embeddings_factory=embeddings_factory,
        )
    assert embedding_calls == []

    def failing_retriever(**kwargs):
        raise RuntimeError("INDEX-CREDENTIAL-SECRET")

    with pytest.raises(KnowledgeEmbeddingError) as index_error:
        create_knowledge_runtime(
            workspace_root=tmp_path,
            environment={
                "EMBEDDING_MODEL": "model",
                "EMBEDDING_API_KEY": "key",
            },
            knowledge_builder=lambda **kwargs: make_build_result(),
            embeddings_factory=embeddings_factory,
            retriever_factory=failing_retriever,
        )
    assert "INDEX-CREDENTIAL-SECRET" not in str(index_error.value)

    isolate_runtime_paths(tmp_path, monkeypatch)
    original_snapshot = {
        "version": 1,
        "messages": [
            {
                "type": "system",
                "data": {
                    "content": (
                        "你是一个工作区助手。根据用户问题，"
                        "决定是否需要调用工具读取项目文件。"
                        "如果需要，先调用合适的工具；"
                        "拿到工具结果后再回答。"
                    ),
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "system",
                    "name": None,
                    "id": None,
                },
            }
        ],
        "memory_summary": "",
    }
    session_store.save("existing", original_snapshot)
    original_bytes = (
        session_store.SESSION_STORE_ROOT / "existing.json"
    ).read_bytes()

    def input_must_not_run(prompt=""):
        raise AssertionError("知识库失败后不得进入输入循环")

    monkeypatch.setattr("builtins.input", input_must_not_run)

    def unsafe_runtime(**kwargs):
        raise RuntimeError("STARTUP-API-KEY-SECRET")

    status = cli.main(
        ["--session", "existing", "--enable-knowledge"],
        knowledge_runtime_factory=unsafe_runtime,
    )
    output = capsys.readouterr().out

    assert status != 0
    assert "[知识库启动失败]" in output
    assert "STARTUP-API-KEY-SECRET" not in output
    assert "Traceback" not in output
    assert (
        session_store.SESSION_STORE_ROOT / "existing.json"
    ).read_bytes() == original_bytes


def test_cli_rag_round_trip_persists_citation_and_audits_no_body(
    tmp_path,
    monkeypatch,
    capsys,
):
    isolate_runtime_paths(tmp_path, monkeypatch)
    secret_content = "RAG-TOOL-BODY-SECRET-SENTINEL"
    search_tool = make_search_tool(secret_content)
    runtime = KnowledgeRuntime(
        search_tool=search_tool,
        corpus_id="c" * 64,
        indexed_file_count=1,
        chunk_count=1,
        skipped_file_count=0,
    )
    model = ScriptedModel(
        [
            [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "search_knowledge",
                            "args": '{"query":"runtime lookup","k":1}',
                            "id": "runtime-search-call",
                            "index": 0,
                        }
                    ],
                )
            ],
            [
                AIMessageChunk(
                    content="检索答案 docs/guide.md:L2-L4"
                )
            ],
        ]
    )
    created_agents = []

    def agent_factory(
        extra_tools=None,
        citation_validator=None,
        citation_policy="observe",
        citation_guard_tool_names=None,
    ):
        agent = WorkspaceAgent(
            model=model,
            tools=list(extra_tools or ()),
            monotonic_clock=lambda: 0.0,
            citation_validator=citation_validator,
            citation_policy=citation_policy,
            citation_guard_tool_names=set(
                citation_guard_tool_names or ()
            ),
        )
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(cli, "create_workspace_agent", agent_factory)
    set_inputs(monkeypatch, ["查阅运行时知识", "exit"])

    status = cli.main(
        ["--session", "rag", "--enable-knowledge"],
        knowledge_runtime_factory=lambda **kwargs: runtime,
    )
    output = capsys.readouterr().out
    snapshot = session_store.load("rag")
    audit_text = (
        audit_log.AUDIT_LOG_ROOT / "rag.jsonl"
    ).read_text(encoding="utf-8")

    assert status == 0
    assert len(created_agents) == 1
    assert "检索答案 docs/guide.md:L2-L4" in output
    assert "[引用] 有效，引用 1，未知 0，可用资料 1" in output
    assert "[会话] 已保存：rag" in output
    assert secret_content not in output
    assert secret_content in str(snapshot)
    assert "docs/guide.md:L2-L4" in str(snapshot)
    assert secret_content not in audit_text
    assert "runtime-chunk-id" not in audit_text
    assert "docs/guide.md:L2-L4" not in audit_text
    audit_records = [
        json.loads(line)
        for line in audit_text.splitlines()
    ]
    citation_record = next(
        record
        for record in audit_records
        if record["event_type"] == "CitationValidationEvent"
    )
    assert citation_record["data"] == {
        "status": "valid",
        "citation_count": 1,
        "valid_citation_count": 1,
        "unknown_citation_count": 0,
        "retrieved_chunk_count": 1,
    }

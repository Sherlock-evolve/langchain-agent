import json
import os
import stat
import time
from collections import deque
from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool

import main as cli
import session_store
import tools as workspace_tools
from agent import WorkspaceAgent
from audit_log import (
    AuditLogError,
    AuditLogLimitError,
    JsonlAuditLogger,
    UnsupportedAuditEventError,
)
from contracts import (
    ApprovalDecision,
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    CitationPolicyEvent,
    CitationValidationEvent,
    ContextTrimmedEvent,
    EventEnvelope,
    MemoryUpdatedEvent,
    ModelCallMetricsEvent,
    SessionSavedEvent,
    SystemEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from persistent_session import PersistentSession


FIXED_TIMESTAMP = "2026-07-23T08:00:00Z"
AUDIT_TOOL_EXECUTIONS = []


def make_envelope(event, sequence=1, session_id="audit-session"):
    return EventEnvelope(
        session_id=session_id,
        turn_id="audit-turn",
        sequence=sequence,
        event=event,
    )


def make_logger(root, **kwargs):
    return JsonlAuditLogger(
        root=root,
        timestamp_factory=lambda: FIXED_TIMESTAMP,
        **kwargs,
    )


def read_records(root, session_id="audit-session"):
    audit_file = root / f"{session_id}.jsonl"
    return [
        json.loads(line)
        for line in audit_file.read_text(encoding="utf-8").splitlines()
    ]


def test_audit_whitelists_every_known_event_type_and_rejects_unknown(
    tmp_path,
):
    root = tmp_path / ".agent_audit"
    logger = make_logger(root)
    events = [
        TokenEvent(text="answer"),
        ToolCallEvent(
            tool_call_id="call-1",
            step=1,
            name="read_file",
            args={"path": "secret.txt"},
        ),
        ToolResultEvent(
            tool_call_id="call-1",
            name="read_file",
            status="success",
            character_count=42,
            truncated=True,
            duration_ms=12,
        ),
        ApprovalRequiredEvent(
            tool_call_id="call-2",
            tool_name="write_file",
            args={"content": "secret"},
            preview="private diff",
        ),
        ApprovalResolvedEvent(
            tool_call_id="call-2",
            tool_name="write_file",
            outcome="approved",
        ),
        SystemEvent(message="private system message"),
        ContextTrimmedEvent(
            removed_message_count=8,
            remaining_message_count=4,
        ),
        MemoryUpdatedEvent(character_count=90),
        CitationValidationEvent(
            status="valid",
            citation_count=2,
            valid_citation_count=2,
            unknown_citation_count=0,
            retrieved_chunk_count=3,
        ),
        CitationPolicyEvent(
            policy="observe",
            action="observed",
            validation_status="valid",
        ),
        ModelCallMetricsEvent(
            call_index=2,
            status="success",
            duration_ms=35,
            first_chunk_ms=7,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            token_source="provider",
        ),
        SessionSavedEvent(session_id="audit-session"),
    ]

    for sequence, event in enumerate(events, start=1):
        logger.record(make_envelope(event, sequence))

    records = read_records(root)
    assert [record["event_type"] for record in records] == [
        type(event).__name__ for event in events
    ]
    assert all(
        set(record) == {
            "schema_version",
            "recorded_at",
            "session_id",
            "turn_id",
            "sequence",
            "event_type",
            "data",
        }
        for record in records
    )
    assert set(records[0]["data"]) == {"character_count"}
    assert set(records[1]["data"]) == {
        "tool_call_id",
        "step",
        "name",
        "argument_count",
    }
    assert set(records[2]["data"]) == {
        "tool_call_id",
        "name",
        "status",
        "character_count",
        "truncated",
        "duration_ms",
        "error_type",
    }
    assert set(records[3]["data"]) == {
        "tool_call_id",
        "tool_name",
        "argument_count",
        "has_preview",
        "preview_character_count",
    }
    assert set(records[4]["data"]) == {
        "tool_call_id",
        "tool_name",
        "outcome",
    }
    assert set(records[5]["data"]) == {"character_count"}
    citation_record = next(
        record
        for record in records
        if record["event_type"] == "CitationValidationEvent"
    )
    assert set(citation_record["data"]) == {
        "status",
        "citation_count",
        "valid_citation_count",
        "unknown_citation_count",
        "retrieved_chunk_count",
    }

    @dataclass(frozen=True)
    class UnknownEvent:
        secret: str

    with pytest.raises(UnsupportedAuditEventError):
        logger.record(
            make_envelope(
                UnknownEvent(secret="must-not-be-written"),
                len(events) + 1,
            )
        )

    assert len(read_records(root)) == len(events)


def test_sensitive_sentinels_never_enter_audit_file(tmp_path):
    sentinel = "SENSITIVE-SENTINEL-DO-NOT-LOG"
    root = tmp_path / ".agent_audit"
    logger = make_logger(root)
    events = [
        TokenEvent(text=sentinel),
        ToolCallEvent(
            tool_call_id="safe-call",
            step=1,
            name="write_file",
            args={
                "content": sentinel,
                "nested": {"token": sentinel},
            },
        ),
        ToolResultEvent(
            tool_call_id="safe-call",
            name="write_file",
            status="error",
            character_count=len(sentinel),
            detail=sentinel,
            error_type="ValueError",
        ),
        ApprovalRequiredEvent(
            tool_call_id="safe-call",
            tool_name="write_file",
            args={"content": sentinel},
            preview=sentinel,
        ),
        ApprovalResolvedEvent(
            tool_call_id="safe-call",
            tool_name="write_file",
            outcome="mismatched",
        ),
        SystemEvent(message=sentinel),
    ]

    for sequence, event in enumerate(events, start=1):
        logger.record(make_envelope(event, sequence))

    raw_log = (root / "audit-session.jsonl").read_text(
        encoding="utf-8"
    )
    records = [
        json.loads(line) for line in raw_log.splitlines()
    ]

    assert sentinel not in raw_log
    assert all(
        "text" not in record["data"]
        and "args" not in record["data"]
        and "preview" not in record["data"]
        and "message" not in record["data"]
        and "detail" not in record["data"]
        for record in records
    )


def test_audit_permissions_jsonl_correlation_and_workspace_hiding(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / ".agent_audit"
    logger = make_logger(root)
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )

    for sequence in range(1, 4):
        logger.record(
            make_envelope(
                MemoryUpdatedEvent(character_count=sequence),
                sequence,
            )
        )

    audit_file = root / "audit-session.jsonl"
    raw_log = audit_file.read_bytes()
    records = read_records(root)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(audit_file.stat().st_mode) == 0o600
    assert raw_log.endswith(b"\n")
    assert len(records) == 3
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert {record["session_id"] for record in records} == {
        "audit-session"
    }
    assert {record["turn_id"] for record in records} == {"audit-turn"}
    assert {record["recorded_at"] for record in records} == {
        FIXED_TIMESTAMP
    }
    assert {record["schema_version"] for record in records} == {1}

    assert workspace_tools.list_files.invoke({}) == "（目录为空）"
    with pytest.raises(ValueError):
        workspace_tools.read_file.invoke(
            {"path": ".agent_audit/audit-session.jsonl"}
        )
    with pytest.raises(ValueError):
        workspace_tools.write_file.invoke(
            {
                "path": ".agent_audit/injected.jsonl",
                "content": "forbidden",
            }
        )
    assert workspace_tools.search_text.invoke(
        {"query": "schema_version"}
    ) == "未找到匹配结果"


def test_audit_rejects_symlinks_nonregular_paths_and_size_limits(
    tmp_path,
):
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    linked_root = tmp_path / "linked-audit"
    linked_root.symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(AuditLogError, match="符号链接"):
        make_logger(linked_root).record(
            make_envelope(TokenEvent(text="blocked"))
        )

    file_root = tmp_path / "file-audit"
    file_root.mkdir()
    outside_file = tmp_path / "outside.jsonl"
    outside_file.write_text("unchanged", encoding="utf-8")
    (file_root / "audit-session.jsonl").symlink_to(outside_file)
    with pytest.raises(AuditLogError, match="符号链接"):
        make_logger(file_root).record(
            make_envelope(TokenEvent(text="blocked"))
        )
    assert outside_file.read_text(encoding="utf-8") == "unchanged"

    directory_root = tmp_path / "directory-audit"
    directory_root.mkdir()
    (directory_root / "audit-session.jsonl").mkdir()
    with pytest.raises(AuditLogError, match="常规文件"):
        make_logger(directory_root).record(
            make_envelope(TokenEvent(text="blocked"))
        )

    invalid_root = tmp_path / "invalid-id-audit"
    with pytest.raises(AuditLogError, match="会话 ID"):
        make_logger(invalid_root).record(
            make_envelope(
                TokenEvent(text="blocked"),
                session_id="../escape",
            )
        )
    assert not invalid_root.exists()

    record_limit_root = tmp_path / "record-limit-audit"
    limited_record_logger = make_logger(
        record_limit_root,
        max_record_bytes=128,
    )
    with pytest.raises(AuditLogLimitError, match="单条"):
        limited_record_logger.record(
            make_envelope(
                ToolCallEvent(
                    tool_call_id="call",
                    step=1,
                    name="x" * 1000,
                    args={},
                )
            )
        )
    assert not record_limit_root.exists()

    total_limit_root = tmp_path / "total-limit-audit"
    initial_logger = make_logger(total_limit_root)
    initial_logger.record(make_envelope(TokenEvent(text="first")))
    audit_file = total_limit_root / "audit-session.jsonl"
    original_bytes = audit_file.read_bytes()
    full_logger = make_logger(
        total_limit_root,
        max_log_bytes=len(original_bytes),
    )
    with pytest.raises(AuditLogLimitError, match="总大小"):
        full_logger.record(
            make_envelope(TokenEvent(text="second"), sequence=2)
        )
    assert audit_file.read_bytes() == original_bytes


@tool
def audit_cli_tool(value: str) -> str:
    """记录审计 CLI 测试工具的执行。"""
    AUDIT_TOOL_EXECUTIONS.append(value)
    return value


class ScriptedModel:
    def __init__(self, responses, *, tools_enabled=False, shared=None):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.shared = shared if shared is not None else {
            "calls": [],
        }

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
            shared=self.shared,
        )

    def stream(self, messages):
        self.shared["calls"].append(self.tools_enabled)
        yield from self.responses.popleft()


class FailingAuditLogger:
    def __init__(self):
        self.calls = []

    def record(self, envelope):
        self.calls.append(envelope)
        raise RuntimeError("SENSITIVE-AUDIT-FAILURE")


def test_cli_audit_failure_does_not_block_approval_or_persistence(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    AUDIT_TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": audit_cli_tool.name,
                            "args": '{"value":"approved"}',
                            "id": "audit-cli-call",
                            "index": 0,
                        }
                    ],
                )
            ],
            [AIMessageChunk(content="审计失败仍完成回答")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[audit_cli_tool],
        approval_required_tools={audit_cli_tool.name},
        monotonic_clock=lambda: 0.0,
    )
    session = PersistentSession(
        "audit-cli",
        agent,
        turn_id_factory=lambda: "audit-cli-turn",
    )
    failing_auditor = FailingAuditLogger()
    inputs = iter(["执行审批工具", "yes", "exit"])
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs),
    )

    status = cli.run_cli(
        session,
        lambda: WorkspaceAgent(model=ScriptedModel([]), tools=[]),
        audit_logger=failing_auditor,
    )
    output = capsys.readouterr().out

    assert status == 0
    assert AUDIT_TOOL_EXECUTIONS == ["approved"]
    assert model.shared["calls"] == [True, True]
    assert session_store.load("audit-cli") == agent.export_snapshot()
    assert "审计失败仍完成回答" in output
    assert "[会话] 已保存：audit-cli" in output
    assert output.count("[审计警告]") == 1
    assert "Traceback" not in output
    assert "SENSITIVE-AUDIT-FAILURE" not in output
    assert len(failing_auditor.calls) > 1


def test_approval_resolution_envelope_and_audit_hide_wrong_decision_id(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    AUDIT_TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            [
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": audit_cli_tool.name,
                            "args": '{"value":"PROTECTED-ARGUMENT"}',
                            "id": "original-tool-call",
                            "index": 0,
                        }
                    ],
                )
            ],
            [AIMessageChunk(content="审批不匹配后回答")],
        ]
    )
    session = PersistentSession(
        "approval-audit",
        WorkspaceAgent(
            model=model,
            tools=[audit_cli_tool],
            approval_required_tools={audit_cli_tool.name},
            monotonic_clock=lambda: 0.0,
        ),
        turn_id_factory=lambda: "approval-audit-turn",
    )
    audit_root = tmp_path / ".agent_audit"
    logger = make_logger(audit_root)
    envelopes = []
    stream = session.stream_turn("记录审批结果")

    while True:
        envelope = next(stream)
        envelopes.append(envelope)
        logger.record(envelope)
        if isinstance(envelope.event, ApprovalRequiredEvent):
            break

    wrong_id = "SENSITIVE-EXTERNAL-WRONG-ID"
    resolved_envelope = stream.send(
        ApprovalDecision(
            tool_call_id=wrong_id,
            approved=True,
        )
    )
    envelopes.append(resolved_envelope)
    logger.record(resolved_envelope)

    result_envelope = next(stream)
    envelopes.append(result_envelope)
    logger.record(result_envelope)
    for envelope in stream:
        envelopes.append(envelope)
        logger.record(envelope)

    event_types = [type(envelope.event) for envelope in envelopes]
    required_index = event_types.index(ApprovalRequiredEvent)
    resolved_index = event_types.index(ApprovalResolvedEvent)
    result_index = event_types.index(ToolResultEvent)
    raw_log = (
        audit_root / "approval-audit.jsonl"
    ).read_text(encoding="utf-8")
    records = read_records(audit_root, "approval-audit")
    resolved_record = next(
        record
        for record in records
        if record["event_type"] == "ApprovalResolvedEvent"
    )

    assert [envelope.sequence for envelope in envelopes] == list(
        range(1, len(envelopes) + 1)
    )
    assert required_index < resolved_index < result_index
    assert resolved_envelope.event == ApprovalResolvedEvent(
        tool_call_id="original-tool-call",
        tool_name=audit_cli_tool.name,
        outcome="mismatched",
    )
    assert resolved_record["data"] == {
        "tool_call_id": "original-tool-call",
        "tool_name": audit_cli_tool.name,
        "outcome": "mismatched",
    }
    assert wrong_id not in raw_log
    assert "PROTECTED-ARGUMENT" not in raw_log
    assert AUDIT_TOOL_EXECUTIONS == []


def test_audit_rotation_retention_and_explicit_deletion(tmp_path):
    root = tmp_path / ".agent_audit"
    logger = make_logger(
        root,
        max_log_bytes=350,
        rotation_count=2,
        retention_days=1,
    )
    for sequence in range(1, 7):
        logger.record(
            make_envelope(
                TokenEvent(text="x" * 100),
                sequence=sequence,
            )
        )

    current = root / "audit-session.jsonl"
    rotated = root / "audit-session.jsonl.1"
    assert current.is_file()
    assert rotated.is_file()
    assert len(list(root.glob("audit-session.jsonl*"))) <= 3

    old_timestamp = time.time() - 2 * 24 * 60 * 60
    os.utime(rotated, (old_timestamp, old_timestamp))
    assert logger.prune_expired() == 1
    assert not rotated.exists()

    removed = logger.delete_session_logs("audit-session")
    assert removed >= 1
    assert list(root.glob("audit-session.jsonl*")) == []

import json
import stat
from collections import deque
from copy import deepcopy
from uuid import UUID

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

import session_store
import tools as workspace_tools
from agent import (
    ApprovalDecision,
    ApprovalRequiredEvent,
    ContextTrimmedEvent,
    MemoryUpdatedEvent,
    SystemEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkspaceAgent,
)
from contracts import (
    EventEnvelope,
    ModelCallMetricsEvent,
    SessionSavedEvent,
)
from persistent_session import (
    PersistentSession,
    PersistentSessionOpenError,
    PersistentSessionSaveError,
)


TOOL_EXECUTIONS = []
ECHO_EXECUTIONS = []


@tool
def read_test_note() -> str:
    """读取测试笔记内容。"""
    TOOL_EXECUTIONS.append("read_test_note")
    return "测试笔记内容"


@tool
def echo_test(value: str) -> str:
    """返回测试输入。"""
    ECHO_EXECUTIONS.append(value)
    return value


class ScriptedModel:
    def __init__(
        self,
        responses,
        *,
        tools_enabled=False,
        call_log=None,
        message_log=None,
    ):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.call_log = call_log if call_log is not None else []
        self.message_log = message_log if message_log is not None else []

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
            call_log=self.call_log,
            message_log=self.message_log,
        )

    def stream(self, messages):
        self.call_log.append(self.tools_enabled)
        self.message_log.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel 响应队列已耗尽")
        yield from self.responses.popleft()


def without_metrics(events):
    return [
        event
        for event in events
        if not isinstance(event, ModelCallMetricsEvent)
    ]


def next_without_metrics(stream):
    while True:
        event = next(stream)
        if not isinstance(event, ModelCallMetricsEvent):
            return event


def envelope_events_without_metrics(envelopes):
    return [
        envelope.event
        for envelope in envelopes
        if not isinstance(envelope.event, ModelCallMetricsEvent)
    ]


def next_envelope_without_metrics(stream):
    while True:
        envelope = next(stream)
        if not isinstance(
            envelope.event,
            ModelCallMetricsEvent,
        ):
            return envelope


class ScriptedClock:
    def __init__(self, values):
        self.values = deque(values)

    def __call__(self):
        if not self.values:
            raise AssertionError("测试时钟读数已耗尽")
        return self.values.popleft()


def tool_call_response(tool_call_id):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "read_test_note",
                    "args": "{",
                    "id": tool_call_id,
                    "index": 0,
                }
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": 0,
                }
            ],
        ),
    ]


def parallel_tool_call_response(*tool_calls):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": name,
                    "args": args,
                    "id": tool_call_id,
                    "index": index,
                }
                for index, (tool_call_id, name, args) in enumerate(tool_calls)
            ],
        )
    ]


def agent_with_committed_tool_history():
    model = ScriptedModel(
        [
            tool_call_response("snapshot-tool-call"),
            [AIMessageChunk(content="快照中的最终回答。")],
        ]
    )
    agent = WorkspaceAgent(model=model, tools=[read_test_note])
    list(agent.stream_turn("生成包含工具调用的历史"))
    agent.memory_summary = "快照中的长期摘要"
    return agent


def test_direct_answer():
    model = ScriptedModel(
        [
            [
                AIMessageChunk(content="直接"),
                AIMessageChunk(content="回答"),
            ]
        ]
    )
    agent = WorkspaceAgent(model=model, tools=[])

    events = list(agent.stream_turn("你好"))

    assert without_metrics(events) == [
        TokenEvent(text="直接"),
        TokenEvent(text="回答"),
    ]
    assert model.call_log == [True]
    assert agent.model_with_tools is not model
    assert agent.model_with_tools.responses is model.responses
    assert list(model.responses) == []
    assert [type(message) for message in agent.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
    ]
    assert agent.messages[-1].content == "直接回答"


def test_tool_call_then_answer():
    TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            tool_call_response("call-1"),
            [
                AIMessageChunk(content="根据笔记，"),
                AIMessageChunk(content="最终回答。"),
            ],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[read_test_note],
        max_tool_calls=1,
    )

    events = list(agent.stream_turn("读取笔记"))
    events = without_metrics(events)

    assert isinstance(events[0], ToolCallEvent)
    assert events[0].tool_call_id == "call-1"
    assert events[0].name == "read_test_note"
    assert events[0].args == {}
    assert isinstance(events[1], ToolResultEvent)
    assert events[1].tool_call_id == "call-1"
    assert events[1].status == "success"
    assert events[2:] == [
        TokenEvent(text="根据笔记，"),
        TokenEvent(text="最终回答。"),
    ]
    assert TOOL_EXECUTIONS == ["read_test_note"]
    assert model.call_log == [True, False]
    assert [type(message) for message in agent.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert agent.messages[3].tool_call_id == "call-1"
    assert agent.messages[3].content == "测试笔记内容"
    assert list(model.responses) == []


def test_cancelled_stream_does_not_commit_history():
    TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            tool_call_response("cancelled-call"),
            [AIMessageChunk(content="下一轮正常回答。")],
        ]
    )
    agent = WorkspaceAgent(model=model, tools=[read_test_note])
    original_history = agent.messages

    stream = agent.stream_turn("这轮会取消")
    event = next_without_metrics(stream)

    assert isinstance(event, ToolCallEvent)
    assert event.tool_call_id == "cancelled-call"
    assert agent.messages is original_history
    assert [type(message) for message in agent.messages] == [SystemMessage]
    assert TOOL_EXECUTIONS == []

    stream.close()
    next_events = list(agent.stream_turn("下一轮"))

    assert without_metrics(next_events) == [
        TokenEvent(text="下一轮正常回答。")
    ]
    assert model.call_log == [True, True]
    assert TOOL_EXECUTIONS == []
    assert [
        message.content
        for message in agent.messages
        if isinstance(message, HumanMessage)
    ] == ["下一轮"]
    assert [type(message) for message in agent.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
    ]
    assert list(model.responses) == []


def test_approved_tool_call_executes_once():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("approved-call", "echo_test", '{"value":"approved"}'),
            ),
            [AIMessageChunk(content="已根据批准的结果回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        approval_required_tools={"echo_test"},
    )

    stream = agent.stream_turn("执行受控工具")
    call_event = next_without_metrics(stream)
    approval_event = next_without_metrics(stream)

    assert isinstance(call_event, ToolCallEvent)
    assert approval_event == ApprovalRequiredEvent(
        tool_call_id="approved-call",
        tool_name="echo_test",
        args={"value": "approved"},
    )

    result_event = stream.send(
        ApprovalDecision(
            tool_call_id="approved-call",
            approved=True,
        )
    )
    remaining_events = list(stream)

    assert isinstance(result_event, ToolResultEvent)
    assert result_event.status == "success"
    assert ECHO_EXECUTIONS == ["approved"]
    assert without_metrics(remaining_events) == [
        TokenEvent(text="已根据批准的结果回答。")
    ]
    assert model.call_log == [True, True]


def test_rejected_tool_call_is_skipped_and_model_continues():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("rejected-call", "echo_test", '{"value":"rejected"}'),
            ),
            [AIMessageChunk(content="工具未批准，继续回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        approval_required_tools={"echo_test"},
    )

    stream = agent.stream_turn("拒绝受控工具")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=False,
        )
    )
    remaining_events = list(stream)

    tool_messages = [
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    ]
    assert result_event.status == "skipped"
    assert result_event.detail == "用户未批准"
    assert ECHO_EXECUTIONS == []
    assert len(tool_messages) == 1
    assert "用户未批准" in tool_messages[0].content
    assert tool_messages[0].content != ""
    assert without_metrics(remaining_events) == [
        TokenEvent(text="工具未批准，继续回答。")
    ]
    assert model.call_log == [True, True]


def test_mismatched_approval_id_is_rejected():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("expected-call", "echo_test", '{"value":"protected"}'),
            ),
            [AIMessageChunk(content="审批无效，继续回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        approval_required_tools={"echo_test"},
    )

    stream = agent.stream_turn("发送错误审批 ID")
    next_without_metrics(stream)
    next_without_metrics(stream)
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id="different-call",
            approved=True,
        )
    )
    remaining_events = list(stream)

    assert result_event.status == "skipped"
    assert result_event.detail == "审批调用 ID 不匹配"
    assert ECHO_EXECUTIONS == []
    assert without_metrics(remaining_events) == [
        TokenEvent(text="审批无效，继续回答。")
    ]
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    assert "用户未批准" in tool_message.content


def test_closing_while_waiting_for_approval_rolls_back_and_releases_lock():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("cancelled-approval", "echo_test", '{"value":"unsafe"}'),
            ),
            [AIMessageChunk(content="下一轮正常回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        approval_required_tools={"echo_test"},
    )
    original_history = agent.messages

    stream = agent.stream_turn("等待审批后取消")
    assert isinstance(next_without_metrics(stream), ToolCallEvent)
    assert isinstance(next_without_metrics(stream), ApprovalRequiredEvent)

    stream.close()

    assert ECHO_EXECUTIONS == []
    assert agent.messages is original_history
    assert [type(message) for message in agent.messages] == [SystemMessage]

    next_events = list(agent.stream_turn("下一轮"))

    assert without_metrics(next_events) == [
        TokenEvent(text="下一轮正常回答。")
    ]
    assert model.call_log == [True, True]
    assert ECHO_EXECUTIONS == []
    assert [
        message.content
        for message in agent.messages
        if isinstance(message, HumanMessage)
    ] == ["下一轮"]


def test_same_tool_cannot_have_previewer_and_preparer():
    model = ScriptedModel([])

    with pytest.raises(ValueError, match="不能同时配置"):
        WorkspaceAgent(
            model=model,
            tools=[echo_test],
            approval_required_tools={"echo_test"},
            approval_previewers={
                "echo_test": lambda **kwargs: "preview"
            },
            approval_preparers={
                "echo_test": lambda **kwargs: "prepared"
            },
        )

    assert model.call_log == []


def test_invalid_preparer_result_fails_closed_before_approval():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("invalid-preparer", "echo_test", '{"value":"unsafe"}'),
            ),
            [AIMessageChunk(content="准备失败后继续回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        approval_required_tools={"echo_test"},
        approval_preparers={
            "echo_test": lambda **kwargs: "不是 PreparedToolAction"
        },
    )

    events = list(agent.stream_turn("错误 preparer 返回类型"))

    result_event = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent)
    )
    assert not any(
        isinstance(event, ApprovalRequiredEvent) for event in events
    )
    assert result_event.status == "error"
    assert result_event.detail == "审批准备失败"
    assert ECHO_EXECUTIONS == []
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    assert "PreparedToolAction" in tool_message.content
    assert model.call_log == [True, True]


def test_approved_write_atomically_creates_file(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    replace_calls = []
    real_replace = workspace_tools.os.replace

    def tracking_replace(source, destination):
        replace_calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(workspace_tools.os, "replace", tracking_replace)
    content = "原子创建内容\n"
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-create",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {"path": "created.txt", "content": content},
                        ensure_ascii=False,
                    ),
                ),
            ),
            [AIMessageChunk(content="文件已创建。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_previewers={
            workspace_tools.write_file.name:
            workspace_tools.preview_write_file
        },
    )

    stream = agent.stream_turn("创建文件")
    assert isinstance(next_without_metrics(stream), ToolCallEvent)
    approval_event = next_without_metrics(stream)
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    created_file = tmp_path / "created.txt"
    assert result_event.status == "success"
    assert created_file.read_text(encoding="utf-8") == content
    assert len(replace_calls) == 1
    temporary_path, destination_path = map(
        workspace_tools.Path,
        replace_calls[0],
    )
    assert temporary_path.parent == tmp_path
    assert destination_path == created_file
    assert not temporary_path.exists()
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    assert "已创建 created.txt" in tool_message.content
    assert f"{len(content)} 个字符" in tool_message.content


def test_approved_write_overwrites_file_with_correct_preview(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    target_file = tmp_path / "note.txt"
    target_file.write_text("旧内容\n保留行\n", encoding="utf-8")
    new_content = "新内容\n保留行\n"
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-update",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {"path": "note.txt", "content": new_content},
                        ensure_ascii=False,
                    ),
                ),
            ),
            [AIMessageChunk(content="文件已更新。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_previewers={
            workspace_tools.write_file.name:
            workspace_tools.preview_write_file
        },
    )

    stream = agent.stream_turn("更新文件")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)

    assert isinstance(approval_event, ApprovalRequiredEvent)
    assert "--- a/note.txt" in approval_event.preview
    assert "+++ b/note.txt" in approval_event.preview
    assert "-旧内容" in approval_event.preview
    assert "+新内容" in approval_event.preview

    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    assert result_event.status == "success"
    assert target_file.read_text(encoding="utf-8") == new_content
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    assert "已更新 note.txt" in tool_message.content


def test_approved_write_preserves_existing_file_mode(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    script_file = tmp_path / "run.sh"
    script_file.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    script_file.chmod(0o755)
    new_content = "#!/bin/sh\necho new\n"
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-script",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {"path": "run.sh", "content": new_content}
                    ),
                ),
            ),
            [AIMessageChunk(content="脚本已更新。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_previewers={
            workspace_tools.write_file.name:
            workspace_tools.preview_write_file
        },
    )

    stream = agent.stream_turn("更新脚本")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    assert result_event.status == "success"
    assert script_file.read_text(encoding="utf-8") == new_content
    assert stat.S_IMODE(script_file.stat().st_mode) == 0o755


def test_existing_file_change_during_approval_causes_conflict(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    target_file = tmp_path / "conflict.txt"
    target_file.write_text("审批时内容\n", encoding="utf-8")
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-existing-conflict",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {
                            "path": "conflict.txt",
                            "content": "Agent 新内容\n",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
            [AIMessageChunk(content="检测到写入冲突。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_preparers={
            workspace_tools.write_file.name:
            workspace_tools.prepare_write_file
        },
    )

    stream = agent.stream_turn("更新可能冲突的文件")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)
    target_file.write_text("外部程序的新内容\n", encoding="utf-8")
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    assert result_event.status == "error"
    assert result_event.detail == "写入冲突"
    assert target_file.read_text(encoding="utf-8") == "外部程序的新内容\n"
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    assert "工具执行冲突" in tool_message.content
    assert "外部修改已保留" in tool_message.content


def test_new_file_created_during_approval_causes_conflict(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    target_file = tmp_path / "new-conflict.txt"
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-new-conflict",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {
                            "path": "new-conflict.txt",
                            "content": "Agent 创建内容\n",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
            [AIMessageChunk(content="新文件写入发生冲突。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_preparers={
            workspace_tools.write_file.name:
            workspace_tools.prepare_write_file
        },
    )

    stream = agent.stream_turn("创建可能冲突的文件")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)
    target_file.write_text("外部程序抢先创建\n", encoding="utf-8")
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    assert result_event.status == "error"
    assert result_event.detail == "写入冲突"
    assert target_file.read_text(encoding="utf-8") == "外部程序抢先创建\n"
    tool_message = next(
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    )
    assert "审批时文件不存在，但执行前已被创建" in tool_message.content


def test_prepared_write_succeeds_when_snapshot_is_unchanged(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    script_file = tmp_path / "stable.sh"
    script_file.write_text("#!/bin/sh\necho stable\n", encoding="utf-8")
    script_file.chmod(0o755)
    new_content = "#!/bin/sh\necho updated\n"
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-stable",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {"path": "stable.sh", "content": new_content}
                    ),
                ),
            ),
            [AIMessageChunk(content="无冲突写入完成。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_preparers={
            workspace_tools.write_file.name:
            workspace_tools.prepare_write_file
        },
    )

    stream = agent.stream_turn("更新未变化的脚本")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    assert result_event.status == "success"
    assert script_file.read_text(encoding="utf-8") == new_content
    assert stat.S_IMODE(script_file.stat().st_mode) == 0o755


def test_rejected_write_leaves_file_unchanged(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    target_file = tmp_path / "protected.txt"
    original_content = "保持不变\n"
    target_file.write_text(original_content, encoding="utf-8")
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-rejected",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {
                            "path": "protected.txt",
                            "content": "不应写入\n",
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
            [AIMessageChunk(content="写入已取消。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_previewers={
            workspace_tools.write_file.name:
            workspace_tools.preview_write_file
        },
    )

    stream = agent.stream_turn("尝试覆盖文件")
    next_without_metrics(stream)
    approval_event = next_without_metrics(stream)
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=False,
        )
    )
    list(stream)

    assert result_event.status == "skipped"
    assert target_file.read_text(encoding="utf-8") == original_content


def test_approved_write_still_rejects_traversal_and_sensitive_paths(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    outside_file = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-traversal",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {
                            "path": f"../{outside_file.name}",
                            "content": "outside",
                        }
                    ),
                ),
                (
                    "write-sensitive",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {"path": ".env", "content": "API_KEY=secret"}
                    ),
                ),
            ),
            [AIMessageChunk(content="非法写入均已拒绝。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
    )

    stream = agent.stream_turn("尝试非法写入")
    assert isinstance(next_without_metrics(stream), ToolCallEvent)
    first_approval = next_without_metrics(stream)
    first_result = stream.send(
        ApprovalDecision(
            tool_call_id=first_approval.tool_call_id,
            approved=True,
        )
    )
    assert isinstance(next_without_metrics(stream), ToolCallEvent)
    second_approval = next_without_metrics(stream)
    second_result = stream.send(
        ApprovalDecision(
            tool_call_id=second_approval.tool_call_id,
            approved=True,
        )
    )
    list(stream)

    assert first_result.status == "error"
    assert second_result.status == "error"
    assert not outside_file.exists()
    assert not (tmp_path / ".env").exists()


def test_write_preview_is_limited_and_tool_event_redacts_content(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        workspace_tools,
        "WORKSPACE_ROOT",
        tmp_path.resolve(),
    )
    content = "SECRET-CONTENT-LINE\n" * 600
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "write-preview",
                    workspace_tools.write_file.name,
                    json.dumps(
                        {"path": "preview.txt", "content": content}
                    ),
                ),
            ),
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[workspace_tools.write_file],
        approval_required_tools={workspace_tools.write_file.name},
        approval_previewers={
            workspace_tools.write_file.name:
            workspace_tools.preview_write_file
        },
    )

    stream = agent.stream_turn("预览长文件")
    call_event = next_without_metrics(stream)
    approval_event = next_without_metrics(stream)

    assert call_event.args["path"] == "preview.txt"
    assert call_event.args["content"] == f"<{len(content)} characters>"
    assert content not in json.dumps(call_event.args, ensure_ascii=False)
    assert approval_event.args["content"] == (
        f"<{len(content)} characters>"
    )
    assert "SECRET-CONTENT-LINE" in approval_event.preview
    assert (
        len(approval_event.preview)
        <= workspace_tools.MAX_WRITE_PREVIEW_CHARACTERS
    )
    assert approval_event.preview.endswith(
        workspace_tools.WRITE_PREVIEW_TRUNCATION_MARKER
    )

    stream.close()

    assert not (tmp_path / "preview.txt").exists()


def test_duplicate_tool_call_is_skipped():
    TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("call-1", "read_test_note", "{}"),
                ("call-2", "read_test_note", "{}"),
            ),
            [AIMessageChunk(content="已使用首次结果回答。")],
        ]
    )
    agent = WorkspaceAgent(model=model, tools=[read_test_note])

    events = list(agent.stream_turn("重复调用测试"))

    result_events = [
        event for event in events if isinstance(event, ToolResultEvent)
    ]
    tool_messages = [
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    ]
    assert len(result_events) == 2
    assert [event.tool_call_id for event in result_events] == [
        "call-1",
        "call-2",
    ]
    assert result_events[0].status == "success"
    assert result_events[0].detail == ""
    assert result_events[1].status == "skipped"
    assert result_events[1].detail == "重复调用"
    assert len(tool_messages) == 2
    assert [message.tool_call_id for message in tool_messages] == [
        "call-1",
        "call-2",
    ]
    assert tool_messages[0].content == "测试笔记内容"
    assert "重复工具调用已跳过" in tool_messages[1].content
    assert TOOL_EXECUTIONS == ["read_test_note"]


def test_tool_budget_is_enforced():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("call-1", "echo_test", '{"value":"first"}'),
                ("call-2", "echo_test", '{"value":"second"}'),
            ),
            [AIMessageChunk(content="根据预算内结果回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        max_tool_calls=1,
    )

    events = list(agent.stream_turn("预算测试"))

    call_events = [
        event for event in events if isinstance(event, ToolCallEvent)
    ]
    result_events = [
        event for event in events if isinstance(event, ToolResultEvent)
    ]
    tool_messages = [
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    ]
    assert [event.args["value"] for event in call_events] == [
        "first",
        "second",
    ]
    assert [event.status for event in result_events] == [
        "success",
        "skipped",
    ]
    assert result_events[1].detail == "工具预算已耗尽"
    assert len(tool_messages) == 2
    assert "工具预算已耗尽" in tool_messages[1].content
    assert ECHO_EXECUTIONS == ["first"]
    assert model.call_log == [True, False]


def test_single_large_tool_result_is_truncated_and_disables_tools():
    ECHO_EXECUTIONS.clear()
    large_result = "x" * 50
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "call-large",
                    "echo_test",
                    json.dumps({"value": large_result}),
                ),
            ),
            [AIMessageChunk(content="根据截断结果回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        max_tool_result_characters=24,
    )

    events = list(agent.stream_turn("读取超长工具结果"))

    result_events = [
        event for event in events if isinstance(event, ToolResultEvent)
    ]
    tool_messages = [
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    ]
    assert len(result_events) == 1
    assert result_events[0].status == "success"
    assert result_events[0].character_count == 24
    assert result_events[0].truncated is True
    assert len(tool_messages[0].content) == 24
    assert tool_messages[0].content.endswith("[工具结果已截断]")
    assert ECHO_EXECUTIONS == [large_result]
    assert model.call_log == [True, False]


def test_parallel_tool_results_share_one_character_budget():
    ECHO_EXECUTIONS.clear()
    first_result = "first123"
    second_result = "y" * 30
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "call-first",
                    "echo_test",
                    json.dumps({"value": first_result}),
                ),
                (
                    "call-second",
                    "echo_test",
                    json.dumps({"value": second_result}),
                ),
                (
                    "call-skipped",
                    "echo_test",
                    json.dumps({"value": "not-executed"}),
                ),
            ),
            [AIMessageChunk(content="根据累计预算内结果回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        max_tool_result_characters=20,
    )

    events = list(agent.stream_turn("并行工具结果预算测试"))

    result_events = [
        event for event in events if isinstance(event, ToolResultEvent)
    ]
    tool_messages = [
        message
        for message in agent.messages
        if isinstance(message, ToolMessage)
    ]
    assert [event.status for event in result_events] == [
        "success",
        "success",
        "skipped",
    ]
    assert [event.character_count for event in result_events] == [8, 12, 0]
    assert [event.truncated for event in result_events] == [
        False,
        True,
        False,
    ]
    assert result_events[2].detail == "工具结果预算已耗尽"
    assert sum(len(message.content) for message in tool_messages[:2]) == 20
    assert [message.tool_call_id for message in tool_messages] == [
        "call-first",
        "call-second",
        "call-skipped",
    ]
    assert "工具结果预算已耗尽" in tool_messages[2].content
    assert tool_messages[2].content != ""
    assert ECHO_EXECUTIONS == [first_result, second_result]
    assert model.call_log == [True, False]


def test_small_tool_result_does_not_exhaust_character_budget():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                (
                    "call-small",
                    "echo_test",
                    json.dumps({"value": "small"}),
                ),
            ),
            [AIMessageChunk(content="根据完整结果回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        max_agent_loops=3,
        max_tool_result_characters=20,
    )

    events = list(agent.stream_turn("小工具结果测试"))

    result_events = [
        event for event in events if isinstance(event, ToolResultEvent)
    ]
    assert len(result_events) == 1
    assert result_events[0].status == "success"
    assert result_events[0].character_count == len("small")
    assert result_events[0].truncated is False
    assert ECHO_EXECUTIONS == ["small"]
    assert model.call_log == [True, True]


def test_last_step_disables_tools():
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("call-1", "echo_test", '{"value":"only"}'),
            ),
            [AIMessageChunk(content="最后一轮直接回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[echo_test],
        max_agent_loops=2,
        max_tool_calls=8,
    )

    events = list(agent.stream_turn("最后一轮测试"))

    assert model.call_log == [True, False]
    assert ECHO_EXECUTIONS == ["only"]
    assert [
        event.status
        for event in events
        if isinstance(event, ToolResultEvent)
    ] == ["success"]
    assert without_metrics(events)[-1] == TokenEvent(
        text="最后一轮直接回答。"
    )


def test_agents_do_not_share_history():
    first_model = ScriptedModel(
        [[AIMessageChunk(content="第一个 Agent 的回答。")]]
    )
    second_model = ScriptedModel(
        [[AIMessageChunk(content="未使用的回答。")]]
    )
    first_agent = WorkspaceAgent(model=first_model, tools=[])
    second_agent = WorkspaceAgent(model=second_model, tools=[])

    list(first_agent.stream_turn("只运行第一个 Agent"))

    assert [type(message) for message in first_agent.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
    ]
    assert [type(message) for message in second_agent.messages] == [
        SystemMessage,
    ]
    assert first_agent.messages is not second_agent.messages
    assert first_model.call_log == [True]
    assert second_model.call_log == []
    assert len(second_model.responses) == 1


def test_old_history_is_trimmed_and_current_question_is_kept():
    model = ScriptedModel(
        [[AIMessageChunk(content="裁剪后回答。")]]
    )
    summary_model = ScriptedModel(
        [[AIMessageChunk(content="旧历史摘要")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
        summary_model=summary_model,
    )
    agent.messages.extend(
        [
            HumanMessage(content="旧问题一"),
            AIMessage(content="旧回答一"),
            HumanMessage(content="旧问题二"),
            AIMessage(content="旧回答二"),
            HumanMessage(content="最近问题"),
            AIMessage(content="最近回答"),
        ]
    )

    events = list(agent.stream_turn("当前问题"))
    events = without_metrics(events)

    assert isinstance(events[0], ContextTrimmedEvent)
    assert events[0].removed_message_count == 4
    assert events[0].remaining_message_count == 4
    assert events[1] == TokenEvent(text="裁剪后回答。")
    assert events[2] == MemoryUpdatedEvent(
        character_count=len("旧历史摘要")
    )
    model_messages = model.message_log[0]
    assert [message.content for message in model_messages] == [
        agent.messages[0].content,
        "最近问题",
        "最近回答",
        "当前问题",
    ]
    assert isinstance(model_messages[-1], HumanMessage)
    assert model_messages[-1].content == "当前问题"
    assert agent.messages[-1].content == "裁剪后回答。"


def test_tool_history_is_kept_or_removed_as_a_complete_group():
    def historical_messages():
        return [
            HumanMessage(content="历史工具问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_test_note",
                        "args": {},
                        "id": "historic-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="历史工具结果",
                tool_call_id="historic-call",
            ),
            AIMessage(content="历史最终回答"),
        ]

    trimmed_model = ScriptedModel(
        [[AIMessageChunk(content="删除旧工具组后回答。")]]
    )
    trimmed_summary_model = ScriptedModel(
        [[AIMessageChunk(content="历史工具轮次摘要")]]
    )
    trimmed_agent = WorkspaceAgent(
        model=trimmed_model,
        tools=[],
        max_context_tokens=5,
        token_counter=lambda messages: len(messages),
        summary_model=trimmed_summary_model,
    )
    trimmed_agent.messages.extend(historical_messages())

    trimmed_events = list(trimmed_agent.stream_turn("当前问题"))

    assert isinstance(trimmed_events[0], ContextTrimmedEvent)
    assert trimmed_events[0].removed_message_count == 4
    assert [type(message) for message in trimmed_model.message_log[0]] == [
        SystemMessage,
        HumanMessage,
    ]
    assert trimmed_model.message_log[0][-1].content == "当前问题"

    kept_model = ScriptedModel(
        [[AIMessageChunk(content="保留完整工具组后回答。")]]
    )
    kept_agent = WorkspaceAgent(
        model=kept_model,
        tools=[],
        max_context_tokens=6,
        token_counter=lambda messages: len(messages),
    )
    kept_agent.messages.extend(historical_messages())

    kept_events = list(kept_agent.stream_turn("当前问题"))

    assert not any(
        isinstance(event, ContextTrimmedEvent) for event in kept_events
    )
    kept_messages = kept_model.message_log[0]
    assert [type(message) for message in kept_messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        HumanMessage,
    ]
    assert kept_messages[2].tool_calls[0]["id"] == "historic-call"
    assert kept_messages[3].tool_call_id == "historic-call"


def test_context_within_budget_does_not_emit_trim_event():
    model = ScriptedModel(
        [[AIMessageChunk(content="无需裁剪。")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=10,
        token_counter=lambda messages: len(messages),
    )

    events = list(agent.stream_turn("短问题"))

    assert without_metrics(events) == [TokenEvent(text="无需裁剪。")]
    assert len(model.message_log) == 1
    assert [type(message) for message in model.message_log[0]] == [
        SystemMessage,
        HumanMessage,
    ]


def test_oversized_current_question_skips_model_and_history_commit():
    model = ScriptedModel(
        [[AIMessageChunk(content="不应被调用")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=1,
        token_counter=lambda messages: len(messages),
    )
    original_history = agent.messages

    events = list(agent.stream_turn("超出预算的问题"))

    assert len(events) == 1
    assert isinstance(events[0], SystemEvent)
    assert "超过上下文预算" in events[0].message
    assert model.call_log == []
    assert model.message_log == []
    assert len(model.responses) == 1
    assert agent.messages is original_history
    assert [type(message) for message in agent.messages] == [SystemMessage]


def test_trimmed_dialogue_updates_memory_and_injects_context():
    model = ScriptedModel(
        [
            [
                AIMessageChunk(content="新的"),
                AIMessageChunk(content="长期摘要"),
            ],
            [AIMessageChunk(content="当前回答。")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
    )
    agent.messages.extend(
        [
            HumanMessage(content="被删除的旧问题"),
            AIMessage(content="被删除的旧回答"),
            HumanMessage(content="保留的最近问题"),
            AIMessage(content="保留的最近回答"),
        ]
    )

    events = list(agent.stream_turn("当前问题"))
    events = without_metrics(events)

    assert isinstance(events[0], ContextTrimmedEvent)
    assert events[1] == TokenEvent(text="当前回答。")
    assert events[2] == MemoryUpdatedEvent(
        character_count=len("新的长期摘要")
    )
    assert model.call_log == [False, True]
    summary_prompt = model.message_log[0][1].content
    assert "被删除的旧问题" in summary_prompt
    assert "被删除的旧回答" in summary_prompt
    assert "保留的最近问题" not in summary_prompt
    current_context = model.message_log[1]
    assert "新的长期摘要" in current_context[0].content
    assert current_context[-1].content == "当前问题"
    assert agent.memory_summary == "新的长期摘要"
    assert agent.messages[0].content == current_context[0].content


def test_existing_memory_is_used_for_next_summary_update():
    model = ScriptedModel(
        [[AIMessageChunk(content="使用更新记忆回答。")]]
    )
    summary_output = "更新后的长期摘要内容"
    summary_model = ScriptedModel(
        [[AIMessageChunk(content=summary_output)]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
        summary_model=summary_model,
        max_summary_characters=6,
    )
    agent.memory_summary = "已有摘要"
    agent.messages.extend(
        [
            HumanMessage(content="旧问题"),
            AIMessage(content="旧回答"),
            HumanMessage(content="最近问题"),
            AIMessage(content="最近回答"),
        ]
    )

    events = list(agent.stream_turn("当前问题"))

    expected_summary = summary_output[:6]
    summary_prompt = summary_model.message_log[0][1].content
    assert "已有摘要" in summary_prompt
    assert "旧问题" in summary_prompt
    assert "旧回答" in summary_prompt
    assert agent.memory_summary == expected_summary
    assert len(agent.memory_summary) == 6
    assert MemoryUpdatedEvent(character_count=6) in events
    assert expected_summary in model.message_log[0][0].content


def test_raw_tool_results_are_excluded_from_summary_prompt():
    model = ScriptedModel(
        [[AIMessageChunk(content="当前回答。")]]
    )
    summary_model = ScriptedModel(
        [[AIMessageChunk(content="安全摘要")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
        summary_model=summary_model,
    )
    agent.messages.extend(
        [
            HumanMessage(content="历史工具问题"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_test_note",
                        "args": {},
                        "id": "secret-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="SECRET_RAW_TOOL_RESULT",
                tool_call_id="secret-call",
            ),
            AIMessage(content="历史重要结论"),
            HumanMessage(content="最近问题"),
            AIMessage(content="最近回答"),
        ]
    )

    events = list(agent.stream_turn("当前问题"))

    assert MemoryUpdatedEvent(
        character_count=len("安全摘要")
    ) in events
    summary_messages = summary_model.message_log[0]
    assert [type(message) for message in summary_messages] == [
        SystemMessage,
        HumanMessage,
    ]
    summary_prompt = summary_messages[1].content
    assert "历史工具问题" in summary_prompt
    assert "历史重要结论" in summary_prompt
    assert "SECRET_RAW_TOOL_RESULT" not in summary_prompt


def test_cancelled_stream_does_not_commit_memory_update():
    model = ScriptedModel(
        [[AIMessageChunk(content="尚未提交的当前回答")]]
    )
    summary_model = ScriptedModel(
        [[AIMessageChunk(content="尚未提交的新摘要")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
        summary_model=summary_model,
    )
    agent.memory_summary = "原有摘要"
    agent.messages.extend(
        [
            HumanMessage(content="被删除的问题"),
            AIMessage(content="被删除的回答"),
            HumanMessage(content="最近问题"),
            AIMessage(content="最近回答"),
        ]
    )
    original_history = agent.messages

    stream = agent.stream_turn("当前问题")
    first_event = next(stream)
    second_event = next(stream)

    assert isinstance(first_event, ContextTrimmedEvent)
    assert second_event == TokenEvent(text="尚未提交的当前回答")
    assert not isinstance(second_event, MemoryUpdatedEvent)
    assert agent.memory_summary == "原有摘要"
    assert agent.messages is original_history

    stream.close()

    assert agent.memory_summary == "原有摘要"
    assert agent.messages is original_history
    assert model.call_log == [True]
    assert summary_model.call_log == [False]


def test_summary_model_failure_does_not_block_current_answer():
    model = ScriptedModel(
        [[AIMessageChunk(content="摘要失败仍正常回答。")]]
    )
    summary_model = ScriptedModel([])
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        max_context_tokens=4,
        token_counter=lambda messages: len(messages),
        summary_model=summary_model,
    )
    agent.memory_summary = "原有摘要"
    agent.messages.extend(
        [
            HumanMessage(content="被删除的问题"),
            AIMessage(content="被删除的回答"),
            HumanMessage(content="最近问题"),
            AIMessage(content="最近回答"),
        ]
    )

    events = list(agent.stream_turn("当前问题"))

    assert isinstance(events[0], ContextTrimmedEvent)
    assert without_metrics(events[1:]) == [
        TokenEvent(text="摘要失败仍正常回答。")
    ]
    assert not any(
        isinstance(event, MemoryUpdatedEvent) for event in events
    )
    assert summary_model.call_log == [False]
    assert model.call_log == [True]
    assert agent.memory_summary == "原有摘要"
    assert agent.messages[-1].content == "摘要失败仍正常回答。"


def test_model_metrics_report_deterministic_timing_before_commit():
    model = ScriptedModel(
        [[AIMessageChunk(content="确定性回答")]]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        monotonic_clock=ScriptedClock([10.0, 10.025, 10.150]),
    )
    original_history = agent.messages
    stream = agent.stream_turn("测量耗时")

    assert next(stream) == TokenEvent(text="确定性回答")
    metrics = next(stream)

    assert metrics == ModelCallMetricsEvent(
        call_index=1,
        status="success",
        duration_ms=150,
        first_chunk_ms=25,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        token_source="unavailable",
    )
    assert agent.messages is original_history

    with pytest.raises(StopIteration):
        next(stream)

    assert agent.messages is not original_history
    assert agent.messages[-1].content == "确定性回答"


def test_model_metrics_extract_provider_usage_metadata():
    model = ScriptedModel(
        [
            [
                AIMessageChunk(
                    content="带用量回答",
                    usage_metadata={
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "total_tokens": 18,
                    },
                )
            ]
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        monotonic_clock=ScriptedClock([1.0, 1.01, 1.02]),
    )

    events = list(agent.stream_turn("读取供应商用量"))
    metrics = next(
        event
        for event in events
        if isinstance(event, ModelCallMetricsEvent)
    )

    assert metrics.input_tokens == 11
    assert metrics.output_tokens == 7
    assert metrics.total_tokens == 18
    assert metrics.token_source == "provider"
    assert agent.messages[-1].usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }


def test_model_metrics_degrade_safely_for_missing_or_invalid_usage():
    invalid_usage_chunk = AIMessageChunk(content="畸形用量仍可回答")
    invalid_usage_chunk.usage_metadata = {"input_tokens": 3}
    cases = [
        AIMessageChunk(content="缺少用量"),
        invalid_usage_chunk,
    ]

    for chunk in cases:
        model = ScriptedModel([[chunk]])
        agent = WorkspaceAgent(
            model=model,
            tools=[],
            monotonic_clock=lambda: 0.0,
        )

        events = list(agent.stream_turn("安全降级"))
        metrics = next(
            event
            for event in events
            if isinstance(event, ModelCallMetricsEvent)
        )

        assert metrics.token_source == "unavailable"
        assert metrics.input_tokens is None
        assert metrics.output_tokens is None
        assert metrics.total_tokens is None
        assert agent.messages[-1].content == chunk.content
        assert agent.messages[-1].usage_metadata is None


def test_model_error_emits_metrics_then_reraises_and_releases_lock():
    class ModelStreamFailure(RuntimeError):
        pass

    original_error = ModelStreamFailure(
        "敏感异常正文不应进入指标事件"
    )

    def failing_chunks():
        raise original_error
        yield

    model = ScriptedModel(
        [
            failing_chunks(),
            [AIMessageChunk(content="锁释放后的回答")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[],
        monotonic_clock=ScriptedClock(
            [2.0, 2.075, 3.0, 3.01, 3.03]
        ),
    )
    original_history = agent.messages
    stream = agent.stream_turn("触发模型异常")

    metrics = next(stream)

    assert metrics == ModelCallMetricsEvent(
        call_index=1,
        status="error",
        duration_ms=75,
        first_chunk_ms=None,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        token_source="unavailable",
        error_type="ModelStreamFailure",
    )
    assert "敏感异常正文" not in repr(metrics)
    assert agent.messages is original_history

    with pytest.raises(ModelStreamFailure) as raised:
        next(stream)

    assert raised.value is original_error
    assert agent.messages is original_history
    next_events = list(agent.stream_turn("验证锁释放"))
    assert without_metrics(next_events) == [
        TokenEvent(text="锁释放后的回答")
    ]


def test_model_metrics_call_indexes_increase_across_tool_loop():
    TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            tool_call_response("metrics-tool-call"),
            [AIMessageChunk(content="工具后的最终回答")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[read_test_note],
        monotonic_clock=ScriptedClock(
            [0.0, 0.01, 0.02, 1.0, 1.03, 1.10]
        ),
    )

    events = list(agent.stream_turn("执行工具循环"))
    metrics = [
        event
        for event in events
        if isinstance(event, ModelCallMetricsEvent)
    ]
    first_tool_event_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, ToolCallEvent)
    )

    assert [event.call_index for event in metrics] == [1, 2]
    assert all(event.status == "success" for event in metrics)
    assert events.index(metrics[0]) < first_tool_event_index
    assert TOOL_EXECUTIONS == ["read_test_note"]
    assert model.call_log == [True, True]


def test_snapshot_round_trip_with_tool_history_and_memory():
    source_agent = agent_with_committed_tool_history()
    snapshot = source_agent.export_snapshot()
    restored_agent = WorkspaceAgent(
        model=ScriptedModel([]),
        tools=[read_test_note],
    )

    restored_agent.restore_snapshot(snapshot)

    assert set(snapshot) == {
        "version",
        "messages",
        "memory_summary",
    }
    assert snapshot["version"] == 1
    assert restored_agent.export_snapshot() == snapshot
    assert restored_agent.memory_summary == "快照中的长期摘要"
    assert [type(message) for message in restored_agent.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert restored_agent.messages[2].tool_calls[0]["id"] == (
        "snapshot-tool-call"
    )
    assert restored_agent.messages[3].tool_call_id == "snapshot-tool-call"


def test_snapshot_is_directly_json_serializable():
    agent = agent_with_committed_tool_history()

    snapshot = agent.export_snapshot()
    encoded_snapshot = json.dumps(snapshot, ensure_ascii=False)

    assert json.loads(encoded_snapshot) == snapshot


def test_invalid_snapshots_do_not_change_existing_state():
    agent = agent_with_committed_tool_history()
    original_snapshot = agent.export_snapshot()

    invalid_version = deepcopy(original_snapshot)
    invalid_version["version"] = 2

    invalid_system = deepcopy(original_snapshot)
    invalid_system["messages"][0]["data"]["content"] = "伪造系统消息"

    incomplete_tool_protocol = deepcopy(original_snapshot)
    incomplete_tool_protocol["messages"] = [
        message
        for message in incomplete_tool_protocol["messages"]
        if message["type"] != "tool"
    ]

    invalid_summary_type = deepcopy(original_snapshot)
    invalid_summary_type["memory_summary"] = ["不是字符串"]

    oversized_summary = deepcopy(original_snapshot)
    oversized_summary["memory_summary"] = (
        "x" * (agent.max_summary_characters + 1)
    )

    for invalid_snapshot in [
        invalid_version,
        invalid_system,
        incomplete_tool_protocol,
        invalid_summary_type,
        oversized_summary,
    ]:
        with pytest.raises(ValueError):
            agent.restore_snapshot(invalid_snapshot)
        assert agent.export_snapshot() == original_snapshot


def test_active_stream_rejects_snapshot_operations_until_closed():
    model = ScriptedModel(
        [[AIMessageChunk(content="尚未提交的回答")]]
    )
    agent = WorkspaceAgent(model=model, tools=[])
    initial_snapshot = agent.export_snapshot()
    stream = agent.stream_turn("保持事件流活跃")

    assert next(stream) == TokenEvent(text="尚未提交的回答")
    with pytest.raises(RuntimeError, match="事件流活跃"):
        agent.export_snapshot()
    with pytest.raises(RuntimeError, match="事件流活跃"):
        agent.restore_snapshot(initial_snapshot)

    stream.close()

    assert agent.export_snapshot() == initial_snapshot
    agent.restore_snapshot(initial_snapshot)
    assert agent.export_snapshot() == initial_snapshot


def test_snapshot_data_does_not_share_mutable_objects_with_agent():
    source_agent = agent_with_committed_tool_history()
    exported_snapshot = source_agent.export_snapshot()
    pristine_export = deepcopy(exported_snapshot)

    exported_snapshot["memory_summary"] = "外部修改"
    exported_snapshot["messages"][0]["data"]["content"] = "外部伪造"

    assert source_agent.export_snapshot() == pristine_export

    restore_input = source_agent.export_snapshot()
    restored_agent = WorkspaceAgent(
        model=ScriptedModel([]),
        tools=[read_test_note],
    )
    restored_agent.restore_snapshot(restore_input)
    restored_state = restored_agent.export_snapshot()

    restore_input["memory_summary"] = "恢复后修改输入"
    restore_input["messages"][1]["data"]["content"] = "篡改用户消息"

    assert restored_agent.export_snapshot() == restored_state

    restored_agent.messages[0].content = "内部后续修改"
    assert restore_input["messages"][0]["data"]["content"] == (
        pristine_export["messages"][0]["data"]["content"]
    )


def test_persistent_session_creates_and_saves_new_session(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    model = ScriptedModel(
        [[AIMessageChunk(content="首次持久化回答。")]]
    )
    session = PersistentSession.open(
        "new-session",
        lambda: WorkspaceAgent(model=model, tools=[]),
    )

    events = list(session.stream_turn("创建新会话"))

    assert envelope_events_without_metrics(events) == [
        TokenEvent(text="首次持久化回答。"),
        SessionSavedEvent(session_id="new-session"),
    ]
    assert session.dirty is False
    assert session_store.load("new-session") == (
        session.agent.export_snapshot()
    )


def test_persistent_session_open_restores_existing_snapshot(
    tmp_path,
    monkeypatch,
):
    store_root = tmp_path / ".agent_sessions"
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        store_root,
    )
    source_agent = agent_with_committed_tool_history()
    snapshot = source_agent.export_snapshot()
    session_store.save("existing", snapshot)
    original_file = (store_root / "existing.json").read_bytes()

    session = PersistentSession.open(
        "existing",
        lambda: WorkspaceAgent(
            model=ScriptedModel([]),
            tools=[read_test_note],
        ),
    )

    assert session.agent.export_snapshot() == snapshot
    assert session.dirty is False
    assert (store_root / "existing.json").read_bytes() == original_file


def test_persistent_session_forwards_approval_decisions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("persistent-approval", "echo_test", '{"value":"once"}'),
            ),
            [AIMessageChunk(content="审批后的持久化回答。")],
        ]
    )
    session = PersistentSession.open(
        "approval-session",
        lambda: WorkspaceAgent(
            model=model,
            tools=[echo_test],
            approval_required_tools={"echo_test"},
        ),
    )

    stream = session.stream_turn("执行需要审批的工具")
    call_event = next_envelope_without_metrics(stream).event
    approval_event = next_envelope_without_metrics(stream).event
    result_event = stream.send(
        ApprovalDecision(
            tool_call_id=approval_event.tool_call_id,
            approved=True,
        )
    ).event
    remaining_events = list(stream)

    assert isinstance(call_event, ToolCallEvent)
    assert isinstance(approval_event, ApprovalRequiredEvent)
    assert result_event.status == "success"
    assert ECHO_EXECUTIONS == ["once"]
    assert envelope_events_without_metrics(remaining_events) == [
        TokenEvent(text="审批后的持久化回答。"),
        SessionSavedEvent(session_id="approval-session"),
    ]
    assert session_store.load("approval-session") == (
        session.agent.export_snapshot()
    )


def test_cancelled_or_incomplete_turn_is_not_saved(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    cancelled_model = ScriptedModel(
        [[AIMessageChunk(content="尚未提交")]]
    )
    cancelled_session = PersistentSession.open(
        "cancelled",
        lambda: WorkspaceAgent(model=cancelled_model, tools=[]),
    )

    stream = cancelled_session.stream_turn("取消本轮")
    assert next(stream).event == TokenEvent(text="尚未提交")
    stream.close()

    assert cancelled_session.dirty is False
    assert session_store.list_sessions() == []

    incomplete_model = ScriptedModel([[]])
    incomplete_session = PersistentSession.open(
        "incomplete",
        lambda: WorkspaceAgent(model=incomplete_model, tools=[]),
    )
    events = list(incomplete_session.stream_turn("模型无响应"))

    assert envelope_events_without_metrics(events) == [
        SystemEvent(message="模型未返回任何消息，当前任务已停止。")
    ]
    assert not any(
        isinstance(envelope.event, SessionSavedEvent)
        for envelope in events
    )
    assert incomplete_session.dirty is False
    assert session_store.list_sessions() == []


def test_save_failure_marks_dirty_and_flush_does_not_replay_tools(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("dirty-tool-call", "echo_test", '{"value":"side-effect"}'),
            ),
            [AIMessageChunk(content="工具调用后的回答。")],
        ]
    )
    session = PersistentSession.open(
        "dirty-session",
        lambda: WorkspaceAgent(model=model, tools=[echo_test]),
    )
    real_save = session_store.save
    save_attempts = []

    def flaky_save(session_id, snapshot):
        save_attempts.append(session_id)
        if len(save_attempts) == 1:
            raise session_store.SessionStoreError("模拟保存失败")
        real_save(session_id, snapshot)

    monkeypatch.setattr(session_store, "save", flaky_save)
    events = []
    stream = session.stream_turn("产生一次工具副作用")

    with pytest.raises(PersistentSessionSaveError):
        while True:
            events.append(next(stream))

    assert session.dirty is True
    assert ECHO_EXECUTIONS == ["side-effect"]
    assert model.call_log == [True, True]
    assert not any(
        isinstance(envelope.event, SessionSavedEvent)
        for envelope in events
    )

    with pytest.raises(PersistentSessionSaveError, match="flush"):
        list(session.stream_turn("不得开始下一轮"))

    session.flush()

    assert session.dirty is False
    assert ECHO_EXECUTIONS == ["side-effect"]
    assert model.call_log == [True, True]
    assert save_attempts == ["dirty-session", "dirty-session"]
    assert session_store.load("dirty-session") == (
        session.agent.export_snapshot()
    )

    session.flush()
    assert save_attempts == ["dirty-session", "dirty-session"]


def test_open_rejects_corrupt_or_semantically_invalid_snapshot(
    tmp_path,
    monkeypatch,
):
    store_root = tmp_path / ".agent_sessions"
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        store_root,
    )
    store_root.mkdir(mode=0o700)
    corrupt_file = store_root / "corrupt.json"
    corrupt_file.write_text("{invalid-json", encoding="utf-8")
    corrupt_bytes = corrupt_file.read_bytes()
    factory_calls = []

    def agent_factory():
        factory_calls.append("called")
        return WorkspaceAgent(model=ScriptedModel([]), tools=[])

    with pytest.raises(PersistentSessionOpenError, match="无法加载"):
        PersistentSession.open("corrupt", agent_factory)

    assert factory_calls == []
    assert corrupt_file.read_bytes() == corrupt_bytes

    invalid_snapshot = {
        "version": 2,
        "messages": [],
        "memory_summary": "",
    }
    session_store.save("semantic", invalid_snapshot)
    semantic_file = store_root / "semantic.json"
    semantic_bytes = semantic_file.read_bytes()

    with pytest.raises(PersistentSessionOpenError, match="语义无效"):
        PersistentSession.open("semantic", agent_factory)

    assert factory_calls == ["called"]
    assert semantic_file.read_bytes() == semantic_bytes


def test_event_envelope_correlates_turn_and_orders_saved_event(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    session = PersistentSession(
        "correlated",
        WorkspaceAgent(
            model=ScriptedModel(
                [
                    [
                        AIMessageChunk(content="第一段"),
                        AIMessageChunk(content="第二段"),
                    ]
                ]
            ),
            tools=[],
        ),
        turn_id_factory=lambda: "deterministic-turn",
    )

    envelopes = list(session.stream_turn("关联这一轮"))

    assert all(
        isinstance(envelope, EventEnvelope)
        for envelope in envelopes
    )
    assert {
        envelope.session_id for envelope in envelopes
    } == {"correlated"}
    assert {
        envelope.turn_id for envelope in envelopes
    } == {"deterministic-turn"}
    assert [envelope.sequence for envelope in envelopes] == [1, 2, 3, 4]
    assert [envelope.event for envelope in envelopes[:2]] == [
        TokenEvent(text="第一段"),
        TokenEvent(text="第二段"),
    ]
    assert isinstance(envelopes[2].event, ModelCallMetricsEvent)
    assert envelopes[-1].event == SessionSavedEvent(
        session_id="correlated"
    )


def test_event_envelope_default_factory_uses_new_uuid_per_turn(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    session = PersistentSession(
        "uuid-turns",
        WorkspaceAgent(
            model=ScriptedModel(
                [
                    [AIMessageChunk(content="第一轮")],
                    [AIMessageChunk(content="第二轮")],
                ]
            ),
            tools=[],
        ),
    )

    first_turn = list(session.stream_turn("问题一"))
    second_turn = list(session.stream_turn("问题二"))
    first_turn_id = first_turn[0].turn_id
    second_turn_id = second_turn[0].turn_id

    assert UUID(first_turn_id).version == 4
    assert UUID(second_turn_id).version == 4
    assert first_turn_id != second_turn_id
    assert [item.sequence for item in first_turn] == [1, 2, 3]
    assert [item.sequence for item in second_turn] == [1, 2, 3]


def test_event_envelope_isolates_session_ids(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )

    def make_session(session_id, answer):
        return PersistentSession(
            session_id,
            WorkspaceAgent(
                model=ScriptedModel(
                    [[AIMessageChunk(content=answer)]]
                ),
                tools=[],
            ),
            turn_id_factory=lambda: "shared-test-turn-id",
        )

    first_events = list(
        make_session("first-session", "第一会话").stream_turn("问题")
    )
    second_events = list(
        make_session("second-session", "第二会话").stream_turn("问题")
    )

    assert {
        envelope.session_id for envelope in first_events
    } == {"first-session"}
    assert {
        envelope.session_id for envelope in second_events
    } == {"second-session"}


def test_event_envelope_forwards_approval_decision(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    ECHO_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            parallel_tool_call_response(
                ("enveloped-approval", "echo_test", '{"value":"approved"}'),
            ),
            [AIMessageChunk(content="审批完成")],
        ]
    )
    session = PersistentSession(
        "approval-envelope",
        WorkspaceAgent(
            model=model,
            tools=[echo_test],
            approval_required_tools={"echo_test"},
        ),
        turn_id_factory=lambda: "approval-turn",
    )

    stream = session.stream_turn("执行审批工具")
    metrics_envelope = next(stream)
    call_envelope = next(stream)
    approval_envelope = next(stream)
    result_envelope = stream.send(
        ApprovalDecision(
            tool_call_id=(
                approval_envelope.event.tool_call_id
            ),
            approved=True,
        )
    )
    remaining = list(stream)
    envelopes = [
        metrics_envelope,
        call_envelope,
        approval_envelope,
        result_envelope,
        *remaining,
    ]

    assert isinstance(
        metrics_envelope.event,
        ModelCallMetricsEvent,
    )
    assert isinstance(call_envelope.event, ToolCallEvent)
    assert isinstance(
        approval_envelope.event,
        ApprovalRequiredEvent,
    )
    assert result_envelope.event.status == "success"
    assert ECHO_EXECUTIONS == ["approved"]
    assert {
        envelope.turn_id for envelope in envelopes
    } == {"approval-turn"}
    assert [envelope.sequence for envelope in envelopes] == list(
        range(1, len(envelopes) + 1)
    )
    assert isinstance(envelopes[-1].event, SessionSavedEvent)


def test_invalid_turn_ids_fail_before_model_call(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )

    for invalid_turn_id in ["", None, "x" * 129]:
        model = ScriptedModel(
            [[AIMessageChunk(content="不应调用模型")]]
        )
        session = PersistentSession(
            "invalid-turn-id",
            WorkspaceAgent(model=model, tools=[]),
            turn_id_factory=(
                lambda value=invalid_turn_id: value
            ),
        )

        with pytest.raises(ValueError, match="turn_id"):
            list(session.stream_turn("不应开始"))

        assert model.call_log == []
        assert model.message_log == []
        assert len(model.responses) == 1

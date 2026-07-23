from collections import deque

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from agent import TokenEvent, ToolCallEvent, ToolResultEvent, WorkspaceAgent


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
    ):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.call_log = call_log if call_log is not None else []

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
            call_log=self.call_log,
        )

    def stream(self, messages):
        self.call_log.append(self.tools_enabled)
        if not self.responses:
            raise AssertionError("ScriptedModel 响应队列已耗尽")
        yield from self.responses.popleft()


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

    assert events == [TokenEvent(text="直接"), TokenEvent(text="回答")]
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
    event = next(stream)

    assert isinstance(event, ToolCallEvent)
    assert event.tool_call_id == "cancelled-call"
    assert agent.messages is original_history
    assert [type(message) for message in agent.messages] == [SystemMessage]
    assert TOOL_EXECUTIONS == []

    stream.close()
    next_events = list(agent.stream_turn("下一轮"))

    assert next_events == [TokenEvent(text="下一轮正常回答。")]
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
    assert events[-1] == TokenEvent(text="最后一轮直接回答。")


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

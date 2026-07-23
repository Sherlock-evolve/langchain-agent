import json
from collections import deque

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from agent import (
    ContextTrimmedEvent,
    MemoryUpdatedEvent,
    SystemEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkspaceAgent,
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

    assert events == [TokenEvent(text="无需裁剪。")]
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
    assert events[1:] == [TokenEvent(text="摘要失败仍正常回答。")]
    assert not any(
        isinstance(event, MemoryUpdatedEvent) for event in events
    )
    assert summary_model.call_log == [False]
    assert model.call_log == [True]
    assert agent.memory_summary == "原有摘要"
    assert agent.messages[-1].content == "摘要失败仍正常回答。"

import asyncio
from collections import deque
from threading import Event

from langchain_core.messages import AIMessageChunk
from langchain_core.tools import StructuredTool, tool

from agent import WorkspaceAgent
from contracts import (
    ApprovalDecision,
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    TokenEvent,
    ToolResultEvent,
)
from tool_execution import ToolExecutionPolicy


class AsyncScriptedModel:
    def __init__(self, responses, *, tools_enabled=False):
        self.responses = (
            responses
            if isinstance(responses, deque)
            else deque(responses)
        )
        self.tools_enabled = tools_enabled

    def bind_tools(self, tools):
        return AsyncScriptedModel(
            self.responses,
            tools_enabled=True,
        )

    async def astream(self, messages):
        await asyncio.sleep(0)
        for chunk in self.responses.popleft():
            yield chunk


def tool_call_response(name, call_id, args="{}"):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": name,
                    "args": args,
                    "id": call_id,
                    "index": 0,
                }
            ],
        )
    ]


def test_astream_turn_prefers_async_model_and_async_tool():
    async_calls = []

    async def async_echo(value: str) -> str:
        """Return a value asynchronously."""
        await asyncio.sleep(0)
        async_calls.append(value)
        return value

    async_tool = StructuredTool.from_function(
        coroutine=async_echo,
        name="async_echo",
        description="Async echo.",
    )
    model = AsyncScriptedModel(
        [
            tool_call_response(
                "async_echo",
                "async-call",
                '{"value":"hello"}',
            ),
            [AIMessageChunk(content="异步完成")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[async_tool],
        tool_execution_policies={
            "async_echo": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=1,
            )
        },
    )

    async def collect():
        return [event async for event in agent.astream_turn("异步执行")]

    events = asyncio.run(collect())

    assert async_calls == ["hello"]
    assert any(
        isinstance(event, ToolResultEvent)
        and event.status == "success"
        for event in events
    )
    assert any(
        isinstance(event, TokenEvent)
        and event.text == "异步完成"
        for event in events
    )
    assert agent.messages[-1].content == "异步完成"


def test_astream_turn_preserves_independent_approval_asend():
    executions = []

    @tool
    def async_approval_write(value: str) -> str:
        """Record an approved value."""
        executions.append(value)
        return "written"

    model = AsyncScriptedModel(
        [
            tool_call_response(
                "async_approval_write",
                "approval-call",
                '{"value":"approved"}',
            ),
            [AIMessageChunk(content="审批完成")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[async_approval_write],
        approval_required_tools={"async_approval_write"},
    )

    async def drive():
        stream = agent.astream_turn("需要审批")
        seen = []
        try:
            while True:
                event = await stream.__anext__()
                seen.append(event)
                if isinstance(event, ApprovalRequiredEvent):
                    seen.append(
                        await stream.asend(
                            ApprovalDecision(
                                tool_call_id=event.tool_call_id,
                                approved=True,
                            )
                        )
                    )
        except StopAsyncIteration:
            pass
        return seen

    events = asyncio.run(drive())

    assert executions == ["approved"]
    assert any(
        isinstance(event, ApprovalResolvedEvent)
        and event.outcome == "approved"
        for event in events
    )
    assert agent.messages[-1].content == "审批完成"


def test_astream_close_requests_client_disconnect_cancellation():
    started = Event()
    release = Event()

    def blocking_read() -> str:
        started.set()
        release.wait(timeout=2)
        return "late"

    blocking_tool = StructuredTool.from_function(
        func=blocking_read,
        name="blocking_async_read",
        description="Blocking read used to test disconnects.",
    )
    model = AsyncScriptedModel(
        [
            tool_call_response(
                "blocking_async_read",
                "blocking-call",
            )
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[blocking_tool],
        default_tool_timeout_seconds=None,
    )
    original_messages = agent.messages
    cancellation_reasons = []
    original_cancel = agent.cancel_active_turn

    def record_cancel(reason="user"):
        cancellation_reasons.append(reason)
        return original_cancel(reason)

    agent.cancel_active_turn = record_cancel

    async def disconnect():
        stream = agent.astream_turn("disconnect")
        while True:
            event = await stream.__anext__()
            if isinstance(event, ToolResultEvent):
                raise AssertionError("tool unexpectedly completed")
            if started.is_set():
                break
            if type(event).__name__ == "ToolCallEvent":
                pending = asyncio.create_task(stream.__anext__())
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.002)
                assert started.is_set()
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
                break
        await stream.aclose()

    asyncio.run(disconnect())
    release.set()

    assert "client_disconnect" in cancellation_reasons
    assert agent.messages is original_messages

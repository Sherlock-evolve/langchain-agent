import time
from collections import deque
from threading import Event, Thread

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, tool

from agent import WorkspaceAgent
from contracts import (
    ModelCallMetricsEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCancelledEvent,
)
from tool_execution import (
    CancellationToken,
    ToolCircuitOpen,
    ToolExecutionBudgetExceeded,
    ToolIdempotencyKeyRequired,
    ToolExecutionMiddleware,
    ToolExecutionPolicy,
    ToolPolicyNotRegistered,
)


class ScriptedModel:
    def __init__(self, responses, *, tools_enabled=False):
        self.responses = (
            responses
            if isinstance(responses, deque)
            else deque(responses)
        )
        self.tools_enabled = tools_enabled

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
        )

    def stream(self, messages):
        yield from self.responses.popleft()


def tool_call_response(tool_name, tool_call_id):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": tool_name,
                    "args": "{}",
                    "id": tool_call_id,
                    "index": 0,
                }
            ],
        )
    ]


def test_read_only_timeout_returns_safe_error_and_model_continues():
    def slow_read():
        time.sleep(0.08)
        return "late-private-result"

    slow_tool = StructuredTool.from_function(
        func=slow_read,
        name="slow_read",
        description="Slow deterministic read.",
    )
    model = ScriptedModel(
        [
            tool_call_response("slow_read", "slow-call"),
            [AIMessageChunk(content="超时后安全收尾")],
        ]
    )
    middleware = ToolExecutionMiddleware(
        {
            "slow_read": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=0.01,
                abandon_on_cancel=True,
            )
        },
        poll_interval_seconds=0.002,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[slow_tool],
        tool_execution_middleware=middleware,
    )

    events = list(agent.stream_turn("执行慢读取"))
    result = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent)
    )

    assert result.status == "error"
    assert result.detail == "工具执行超时"
    assert result.error_type == "ToolExecutionTimeout"
    assert "late-private-result" not in repr(events)
    assert agent.messages[-1].content == "超时后安全收尾"


def test_agent_cancellation_stops_waiting_and_rolls_back_turn():
    started = Event()
    release = Event()

    def blocking_read():
        started.set()
        release.wait(timeout=2)
        return "uncommitted-result"

    blocking_tool = StructuredTool.from_function(
        func=blocking_read,
        name="blocking_read",
        description="Cancellable deterministic read.",
    )
    model = ScriptedModel(
        [
            tool_call_response("blocking_read", "blocking-call"),
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[blocking_tool],
        default_tool_timeout_seconds=None,
    )
    original_messages = agent.messages
    stream = agent.stream_turn("开始可取消读取")

    first = next(stream)
    second = next(stream)
    assert isinstance(first, ModelCallMetricsEvent)
    assert isinstance(second, ToolCallEvent)

    collected = []
    failure = []

    def consume():
        try:
            collected.extend(list(stream))
        except BaseException as error:
            failure.append(error)

    consumer = Thread(target=consume)
    consumer.start()
    assert started.wait(timeout=1)
    assert agent.cancel_active_turn("client_disconnect") is True
    consumer.join(timeout=1)
    release.set()

    assert not consumer.is_alive()
    assert failure == []
    assert any(
        isinstance(event, ToolResultEvent)
        and event.status == "skipped"
        and event.error_type == "ToolExecutionCancelled"
        for event in collected
    )
    assert collected[-1] == TurnCancelledEvent(
        reason="client_disconnect"
    )
    assert agent.messages is original_messages
    assert agent.cancel_active_turn() is False


def test_side_effect_policy_finishes_started_atomic_boundary():
    started = Event()
    release = Event()
    token = CancellationToken()
    middleware = ToolExecutionMiddleware(
        {
            "write_like": ToolExecutionPolicy(
                risk="workspace_write",
                timeout_seconds=None,
                abandon_on_cancel=False,
            )
        }
    )
    outcome = []

    def action():
        started.set()
        release.wait(timeout=1)
        return "committed"

    worker = Thread(
        target=lambda: outcome.append(
            middleware.execute("write_like", action, token)
        )
    )
    worker.start()
    assert started.wait(timeout=1)
    assert token.cancel("user") is True
    release.set()
    worker.join(timeout=1)

    assert outcome == ["committed"]
    assert not worker.is_alive()


def test_strict_registration_rejects_tools_without_risk_policy():
    middleware = ToolExecutionMiddleware(
        {},
        require_registered_policies=True,
    )

    with pytest.raises(ToolPolicyNotRegistered, match="unregistered"):
        middleware.execute(
            "unregistered",
            lambda: "unsafe-default",
            CancellationToken(),
        )


def test_read_only_retry_uses_bounded_backoff_then_succeeds():
    attempts = []
    sleeps = []
    middleware = ToolExecutionMiddleware(
        {
            "flaky_read": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=1,
                max_attempts=3,
                initial_backoff_seconds=0.1,
                max_backoff_seconds=0.1,
            )
        },
        poll_interval_seconds=0.05,
        sleeper=sleeps.append,
    )

    def action():
        attempts.append(len(attempts) + 1)
        if len(attempts) < 3:
            raise OSError("transient")
        return "stable"

    assert middleware.execute(
        "flaky_read",
        action,
        CancellationToken(),
    ) == "stable"
    assert attempts == [1, 2, 3]
    assert sum(sleeps) == pytest.approx(0.2)


def test_external_side_effect_retries_only_with_idempotency_key():
    attempts = []
    middleware = ToolExecutionMiddleware(
        {
            "external": ToolExecutionPolicy(
                risk="external_side_effect",
                timeout_seconds=None,
                abandon_on_cancel=False,
                max_attempts=2,
                initial_backoff_seconds=0,
            )
        }
    )

    def action():
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            raise OSError("retryable")
        return "created-once"

    with pytest.raises(ToolIdempotencyKeyRequired):
        middleware.execute(
            "external",
            action,
            CancellationToken(),
        )
    assert attempts == []

    assert middleware.execute(
        "external",
        action,
        CancellationToken(),
        idempotency_key="request-123",
    ) == "created-once"
    assert attempts == [1, 2]


def test_circuit_breaker_opens_after_consecutive_failures():
    calls = []
    middleware = ToolExecutionMiddleware(
        {
            "unstable": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=1,
                circuit_failure_threshold=2,
                circuit_recovery_seconds=60,
            )
        }
    )

    def fail():
        calls.append("called")
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        middleware.execute("unstable", fail, CancellationToken())
    with pytest.raises(RuntimeError, match="down"):
        middleware.execute("unstable", fail, CancellationToken())
    with pytest.raises(ToolCircuitOpen):
        middleware.execute("unstable", fail, CancellationToken())
    assert calls == ["called", "called"]


def test_cooperative_action_receives_deadline_aware_child_token():
    observed = []
    middleware = ToolExecutionMiddleware(
        {
            "cooperative": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=1,
                cooperative_cancellation=True,
                total_budget_seconds=1,
            )
        }
    )

    def action(token):
        observed.append(token)
        token.raise_if_cancelled()
        return token.remaining_seconds

    remaining = middleware.execute(
        "cooperative",
        action,
        CancellationToken(),
    )
    assert len(observed) == 1
    assert isinstance(observed[0], CancellationToken)
    assert 0 < remaining <= 1


def test_cooperative_action_observes_total_budget():
    middleware = ToolExecutionMiddleware(
        {
            "budgeted": ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=1,
                cooperative_cancellation=True,
                total_budget_seconds=0.01,
            )
        },
        poll_interval_seconds=0.001,
    )

    def action(token):
        while True:
            time.sleep(0.002)
            token.raise_if_cancelled()

    with pytest.raises(ToolExecutionBudgetExceeded):
        middleware.execute(
            "budgeted",
            action,
            CancellationToken(),
        )


def test_turn_request_idempotency_key_is_injected_into_external_tool():
    observed_keys = []

    @tool
    def external_create(config: RunnableConfig) -> str:
        """Create one external resource idempotently."""
        observed_keys.append(
            config["configurable"]["idempotency_key"]
        )
        return "created"

    model = ScriptedModel(
        [
            tool_call_response("external_create", "external-call"),
            [AIMessageChunk(content="done")],
        ]
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[external_create],
        tool_execution_policies={
            "external_create": ToolExecutionPolicy(
                risk="external_side_effect",
                timeout_seconds=None,
                abandon_on_cancel=False,
            )
        },
    )

    list(
        agent.stream_turn(
            "create",
            request_idempotency_key="request-123",
        )
    )

    assert len(observed_keys) == 1
    assert len(observed_keys[0]) == 64
    assert all(
        character in "0123456789abcdef"
        for character in observed_keys[0]
    )

import hashlib
import json
import time
from collections.abc import Callable, Generator
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock, local
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
    messages_from_dict,
    messages_to_dict,
    trim_messages,
)

from async_runtime import (
    bridge_sync_generator,
    iterate_async_synchronously,
    run_coroutine_synchronously,
)
from contracts import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    CitationPolicyEvent,
    CitationValidationEvent,
    ContextTrimmedEvent,
    MemoryUpdatedEvent,
    ModelCallMetricsEvent,
    PreparedToolAction,
    SessionSavedEvent,
    SystemEvent,
    TokenEvent,
    ToolActionConflictError,
    ToolCallEvent,
    ToolResultEvent,
    TurnCancelledEvent,
)
from tool_execution import (
    CancellationToken,
    IDEMPOTENCY_KEY_PATTERN,
    ToolExecutionCancelled,
    ToolExecutionMiddleware,
    ToolExecutionPolicy,
    ToolExecutionTimeout,
)


SYSTEM_PROMPT = (
    "你是一位人工智能老师，也是当前项目的工作区助手。"
    "需要了解项目文件时，请使用工具获取真实信息，不要猜测。"
    "请用通俗、准确的方式回答。"
)
TOOL_RESULT_TRUNCATION_MARKER = "\n[工具结果已截断]"
SNAPSHOT_VERSION = 1
PENDING_APPROVAL_VERSION = 1
REDACTED_ARGUMENT_NAMES = {
    "api_key",
    "content",
    "password",
    "secret",
    "token",
}


class _FrozenDict(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("事件参数不可修改")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass
class _TurnState:
    model_call_count: int = 0
    tool_call_count: int = 0
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    tool_budget_exhausted: bool = False
    tool_result_character_count: int = 0
    tool_result_budget_exhausted: bool = False
    request_idempotency_key: str | None = None


class WorkspaceAgent:
    """单会话 Agent；同一实例同一时间只允许运行一个事件流。"""

    def __init__(
        self,
        model,
        tools,
        max_agent_loops=5,
        max_tool_calls=8,
        max_tool_result_characters=12_000,
        max_context_tokens=6000,
        token_counter="approximate",
        summary_model=None,
        max_summary_characters=2000,
        approval_required_tools: set[str] | None = None,
        approval_previewers: dict[str, Callable[..., str]] | None = None,
        approval_preparers: (
            dict[str, Callable[..., PreparedToolAction]] | None
        ) = None,
        monotonic_clock: Callable[[], float] | None = None,
        citation_validator: Callable | None = None,
        citation_policy: Literal[
            "observe",
            "require_valid",
        ] = "observe",
        citation_guard_tool_names: set[str] | None = None,
        tool_execution_middleware: ToolExecutionMiddleware | None = None,
        tool_execution_policies: (
            dict[str, ToolExecutionPolicy] | None
        ) = None,
        default_tool_timeout_seconds: float | None = 30.0,
    ):
        configured_previewers = dict(approval_previewers or {})
        configured_preparers = dict(approval_preparers or {})
        overlapping_handlers = (
            configured_previewers.keys()
            & configured_preparers.keys()
        )
        if overlapping_handlers:
            tool_names = ", ".join(sorted(overlapping_handlers))
            raise ValueError(
                "同一工具不能同时配置 approval_previewer "
                f"和 approval_preparer：{tool_names}"
            )

        self.model = model
        self.tools = list(tools)
        self.max_agent_loops = max_agent_loops
        self.max_tool_calls = max_tool_calls
        self.max_tool_result_characters = max(
            0,
            max_tool_result_characters,
        )
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter
        self.summary_model = summary_model if summary_model is not None else model
        self.max_summary_characters = max_summary_characters
        self.approval_required_tools = set(approval_required_tools or ())
        self.approval_previewers = configured_previewers
        self.approval_preparers = configured_preparers
        self.monotonic_clock = (
            monotonic_clock
            if monotonic_clock is not None
            else time.monotonic
        )
        if not callable(self.monotonic_clock):
            raise TypeError("monotonic_clock 必须可调用")
        if citation_validator is not None and not callable(
            citation_validator
        ):
            raise TypeError("citation_validator 必须可调用")
        if citation_policy not in {"observe", "require_valid"}:
            raise ValueError("citation_policy 必须是 observe 或 require_valid")
        configured_guard_tool_names = set(
            citation_guard_tool_names or ()
        )
        if any(
            not isinstance(tool_name, str) or not tool_name
            for tool_name in configured_guard_tool_names
        ):
            raise ValueError("citation_guard_tool_names 必须包含非空工具名")
        if citation_policy == "require_valid" and (
            citation_validator is None
            or not configured_guard_tool_names
        ):
            raise ValueError(
                "require_valid 必须配置引用校验器和至少一个门禁工具名"
            )
        self.citation_validator = citation_validator
        self.citation_policy = citation_policy
        self.citation_guard_tool_names = configured_guard_tool_names
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        if len(self.tools_by_name) != len(self.tools):
            raise ValueError("工具名称必须唯一")
        if (
            tool_execution_middleware is not None
            and tool_execution_policies is not None
        ):
            raise ValueError(
                "不能同时配置 tool_execution_middleware "
                "和 tool_execution_policies"
            )
        if tool_execution_middleware is None:
            configured_execution_policies = dict(
                tool_execution_policies or {}
            )
            for tool_name in self.approval_required_tools:
                configured_execution_policies.setdefault(
                    tool_name,
                    ToolExecutionPolicy(
                        risk="workspace_write",
                        timeout_seconds=None,
                        abandon_on_cancel=False,
                    ),
                )
            tool_execution_middleware = ToolExecutionMiddleware(
                configured_execution_policies,
                default_policy=ToolExecutionPolicy(
                    risk="read_only",
                    timeout_seconds=default_tool_timeout_seconds,
                    abandon_on_cancel=True,
                ),
            )
        if not isinstance(
            tool_execution_middleware,
            ToolExecutionMiddleware,
        ):
            raise TypeError(
                "tool_execution_middleware 必须是 ToolExecutionMiddleware"
            )
        self.tool_execution_middleware = tool_execution_middleware
        self.tool_execution_middleware.validate_registered_tools(
            self.tools_by_name
        )
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.memory_summary = ""
        self._turn_lock = Lock()
        self._active_cancellation_token = None
        self._pending_approval = None
        self._execution_context = local()
        self._prefer_async_execution = False

    @property
    def _prefer_async_execution(self) -> bool:
        return bool(
            getattr(
                self._execution_context,
                "prefer_async_execution",
                False,
            )
        )

    @_prefer_async_execution.setter
    def _prefer_async_execution(self, value: bool) -> None:
        self._execution_context.prefer_async_execution = bool(value)

    def export_snapshot(self) -> dict:
        """导出当前已提交会话状态的 JSON 可序列化副本。"""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("事件流活跃时不能导出会话快照")

        try:
            snapshot = {
                "version": SNAPSHOT_VERSION,
                "messages": messages_to_dict(self.messages),
                "memory_summary": self.memory_summary,
            }
            try:
                return json.loads(
                    json.dumps(
                        snapshot,
                        ensure_ascii=False,
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "当前会话包含无法 JSON 序列化的消息"
                ) from error
        finally:
            self._turn_lock.release()

    def restore_snapshot(self, snapshot: dict) -> None:
        """完整验证后原子恢复已提交的会话状态。"""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("事件流活跃时不能恢复会话快照")

        try:
            if self._pending_approval is not None:
                raise RuntimeError(
                    "存在待审批轮次时不能恢复会话快照"
                )
            restored_messages, restored_summary = (
                self._validate_snapshot(snapshot)
            )
            self.messages = restored_messages
            self.memory_summary = restored_summary
        finally:
            self._turn_lock.release()

    def _validate_snapshot(self, snapshot: dict) -> tuple[list, str]:
        if not isinstance(snapshot, dict):
            raise ValueError("会话快照必须是字典")

        try:
            candidate = deepcopy(snapshot)
        except Exception as error:
            raise ValueError("无法复制会话快照") from error

        required_keys = {
            "version",
            "messages",
            "memory_summary",
        }
        if set(candidate) != required_keys:
            raise ValueError("会话快照字段不完整或包含未知字段")
        if (
            type(candidate["version"]) is not int
            or candidate["version"] != SNAPSHOT_VERSION
        ):
            raise ValueError("不支持的会话快照版本")

        restored_summary = candidate["memory_summary"]
        if not isinstance(restored_summary, str):
            raise ValueError("长期摘要必须是字符串")
        if len(restored_summary) > self.max_summary_characters:
            raise ValueError("长期摘要超过允许的长度")

        serialized_messages = candidate["messages"]
        if not isinstance(serialized_messages, list):
            raise ValueError("会话消息必须是列表")
        try:
            restored_messages = messages_from_dict(serialized_messages)
        except Exception as error:
            raise ValueError("会话消息无法反序列化") from error

        if not restored_messages:
            raise ValueError("会话快照缺少系统消息")
        if not self._is_current_system_message(
            restored_messages[0],
            restored_summary,
        ):
            raise ValueError("会话快照的第一条消息不是当前系统消息")
        if any(
            isinstance(message, SystemMessage)
            for message in restored_messages[1:]
        ):
            raise ValueError("会话快照包含额外的系统消息")

        allowed_message_types = (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )
        if any(
            not isinstance(message, allowed_message_types)
            for message in restored_messages
        ):
            raise ValueError("会话快照包含不支持的消息类型")
        if not self._has_complete_tool_groups(restored_messages):
            raise ValueError("会话快照中的工具调用协议不完整")
        if not self._has_only_committed_turns(restored_messages):
            raise ValueError("会话快照包含未完成的对话轮次")

        return deepcopy(restored_messages), restored_summary

    def _is_current_system_message(
        self,
        message,
        memory_summary: str,
    ) -> bool:
        if (
            not isinstance(message, SystemMessage)
            or not isinstance(message.content, str)
        ):
            return False
        if message.content == SYSTEM_PROMPT:
            return True

        summary_header = f"{SYSTEM_PROMPT}\n\n长期记忆摘要：\n"
        if not message.content.startswith(summary_header):
            return False

        embedded_summary = message.content[len(summary_header):]
        allowed_summary = memory_summary[
            : max(0, self.max_summary_characters)
        ]
        return (
            bool(embedded_summary)
            and allowed_summary.startswith(embedded_summary)
        )

    def stream_turn(
        self,
        question: str,
        *,
        request_idempotency_key: str | None = None,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        """运行一轮并产生事件；提前停止消费时应关闭返回的生成器。"""
        if (
            request_idempotency_key is not None
            and (
                not isinstance(request_idempotency_key, str)
                or IDEMPOTENCY_KEY_PATTERN.fullmatch(
                    request_idempotency_key
                )
                is None
            )
        ):
            raise ValueError("request_idempotency_key 非法")
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("同一 WorkspaceAgent 不能同时运行多个对话轮次")
        if self._pending_approval is not None:
            self._turn_lock.release()
            raise RuntimeError("当前存在待恢复审批，不能开始新轮次")

        cancellation_token = CancellationToken()
        self._active_cancellation_token = cancellation_token
        try:
            try:
                yield from self._run_turn_transaction(
                    question,
                    request_idempotency_key=request_idempotency_key,
                )
            except ToolExecutionCancelled:
                yield TurnCancelledEvent(
                    reason=cancellation_token.reason,
                )
        finally:
            self._active_cancellation_token = None
            self._turn_lock.release()

    async def astream_turn(
        self,
        question: str,
        *,
        cancellation_reason: str = "client_disconnect",
        request_idempotency_key: str | None = None,
    ):
        """Asynchronously stream one transactional turn.

        The worker prefers model ``astream`` and tool ``ainvoke`` interfaces
        while preserving the synchronous generator's approval ``asend``
        protocol and rollback behavior.
        """

        def stream_factory():
            self._prefer_async_execution = True
            try:
                yield from self.stream_turn(
                    question,
                    request_idempotency_key=request_idempotency_key,
                )
            finally:
                self._prefer_async_execution = False

        stream = bridge_sync_generator(
            stream_factory,
            cancel_callback=lambda: self.cancel_active_turn(
                cancellation_reason
            ),
        )
        decision = None
        try:
            while True:
                try:
                    event = await stream.asend(decision)
                except StopAsyncIteration:
                    return
                decision = yield event
        finally:
            await stream.aclose()

    def cancel_active_turn(self, reason: str = "user") -> bool:
        """请求取消当前轮次；没有活跃轮次时返回 False。"""
        cancellation_token = self._active_cancellation_token
        if cancellation_token is None:
            return False
        return cancellation_token.cancel(reason)

    def _run_turn_transaction(
        self,
        question: str,
        *,
        request_idempotency_key: str | None = None,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        current_message = HumanMessage(content=question)
        base_system_message = SystemMessage(content=SYSTEM_PROMPT)
        required_messages = [base_system_message, current_message]
        required_context = self._trim_history(required_messages)
        if not self._contains_required_context(
            required_context,
            current_message,
        ):
            yield SystemEvent(
                message=(
                    "系统消息和当前问题超过上下文预算，"
                    "当前任务已停止。"
                )
            )
            return

        working_summary = self.memory_summary
        required_with_summary = self._with_fitting_summary(
            required_messages,
            working_summary,
        )
        candidate_messages = [
            required_with_summary[0],
            *self.messages[1:],
            current_message,
        ]
        working_messages = self._trim_history(candidate_messages)
        if not self._contains_required_context(
            working_messages,
            current_message,
        ):
            yield SystemEvent(
                message=(
                    "系统消息和当前问题超过上下文预算，"
                    "当前任务已停止。"
                )
            )
            return
        working_messages = self._remove_incomplete_tool_groups(
            working_messages,
            current_message,
        )
        removed_message_count = len(candidate_messages) - len(working_messages)
        if removed_message_count > 0:
            yield ContextTrimmedEvent(
                removed_message_count=removed_message_count,
                remaining_message_count=len(working_messages),
            )

        removed_turns = self._extract_removed_turns(working_messages)
        if removed_turns:
            updated_summary = self._update_memory_summary(
                working_summary,
                removed_turns,
            )
            if updated_summary is not None:
                working_summary = updated_summary

        working_messages = self._with_fitting_summary(
            working_messages,
            working_summary,
        )
        current_turn_start_index = len(working_messages) - 1

        state = _TurnState(
            request_idempotency_key=request_idempotency_key,
        )
        yield from self._continue_turn_transaction(
            working_messages=working_messages,
            working_summary=working_summary,
            current_turn_start_index=current_turn_start_index,
            state=state,
            start_step=1,
        )

    def _continue_turn_transaction(
        self,
        *,
        working_messages: list,
        working_summary: str,
        current_turn_start_index: int,
        state: _TurnState,
        start_step: int,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        answered = False
        task_stopped = False

        for step in range(start_step, self.max_agent_loops + 1):
            tools_allowed = (
                step < self.max_agent_loops
                and not state.tool_budget_exhausted
                and not state.tool_result_budget_exhausted
            )
            active_model = self.model_with_tools if tools_allowed else self.model
            state.model_call_count += 1
            guarded_tool_used = self._guarded_tool_was_used(
                working_messages[current_turn_start_index:]
            )
            buffer_candidate = (
                self.citation_policy == "require_valid"
                and guarded_tool_used
            )
            buffered_events = []
            response_stream = self._stream_response(
                active_model,
                working_messages,
                call_index=state.model_call_count,
                cancellation_token=self._active_cancellation_token,
            )
            if buffer_candidate:
                try:
                    while True:
                        buffered_events.append(next(response_stream))
                except StopIteration as stopped:
                    response = stopped.value
                except Exception:
                    for buffered_event in buffered_events:
                        if isinstance(
                            buffered_event,
                            ModelCallMetricsEvent,
                        ):
                            yield buffered_event
                    raise
            else:
                response = yield from response_stream

            if response is None:
                if buffer_candidate:
                    yield from buffered_events
                yield SystemEvent(
                    message="模型未返回任何消息，当前任务已停止。"
                )
                task_stopped = True
                break

            working_messages.append(response)

            if not response.tool_calls:
                citation_event = self._validate_current_turn_citations(
                    working_messages[current_turn_start_index:]
                )
                policy_event = self._citation_policy_event(
                    citation_event,
                    guarded_tool_used=guarded_tool_used,
                )

                if (
                    buffer_candidate
                    and (
                        citation_event is None
                        or citation_event.status != "valid"
                    )
                ):
                    for buffered_event in buffered_events:
                        if isinstance(
                            buffered_event,
                            ModelCallMetricsEvent,
                        ):
                            yield buffered_event
                    if citation_event is not None:
                        yield citation_event
                    if policy_event is not None:
                        yield policy_event
                    yield SystemEvent(
                        message=(
                            "引用校验未通过，候选回答未提交。"
                        )
                    )
                    task_stopped = True
                    break

                if buffer_candidate:
                    yield from buffered_events

                memory_changed = working_summary != self.memory_summary
                self.messages = working_messages
                self.memory_summary = working_summary
                answered = True
                if citation_event is not None:
                    yield citation_event
                if policy_event is not None:
                    yield policy_event
                if memory_changed:
                    yield MemoryUpdatedEvent(
                        character_count=len(self.memory_summary),
                    )
                break

            if buffer_candidate:
                yield from buffered_events

            tool_calls = list(response.tool_calls)
            for tool_index, tool_call in enumerate(tool_calls):
                state_before = self._copy_turn_state(state)
                yield from self._execute_tool_call_with_recovery(
                    tool_call=tool_call,
                    remaining_tool_calls=tool_calls[tool_index + 1 :],
                    step=step,
                    tools_allowed=tools_allowed,
                    state=state,
                    state_before=state_before,
                    working_messages=working_messages,
                    working_summary=working_summary,
                    current_turn_start_index=current_turn_start_index,
                )

        if not answered and not task_stopped:
            yield SystemEvent(
                message=(
                    f"Agent 循环达到 {self.max_agent_loops} 次上限，"
                    "已停止。"
                )
            )

    def _execute_tool_call_with_recovery(
        self,
        *,
        tool_call: dict,
        remaining_tool_calls: list[dict],
        step: int,
        tools_allowed: bool,
        state: _TurnState,
        state_before: _TurnState,
        working_messages: list,
        working_summary: str,
        current_turn_start_index: int,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        tool_stream = self._execute_tool_call(
            tool_call=tool_call,
            step=step,
            tools_allowed=tools_allowed,
            state=state,
            working_messages=working_messages,
        )
        decision = None
        completed = False

        try:
            while True:
                try:
                    event = tool_stream.send(decision)
                except StopIteration:
                    completed = True
                    return

                decision = None
                if isinstance(event, ApprovalRequiredEvent):
                    if tool_call["name"] in self.approval_preparers:
                        self._pending_approval = (
                            self._build_pending_approval_record(
                                working_messages=working_messages,
                                working_summary=working_summary,
                                current_turn_start_index=(
                                    current_turn_start_index
                                ),
                                step=step,
                                tools_allowed=tools_allowed,
                                state=state_before,
                                tool_call=tool_call,
                                remaining_tool_calls=remaining_tool_calls,
                            )
                        )
                elif isinstance(event, ApprovalResolvedEvent):
                    self._pending_approval = None

                decision = yield event
        finally:
            if not completed:
                tool_stream.close()

    @staticmethod
    def _copy_turn_state(state: _TurnState) -> _TurnState:
        return _TurnState(
            model_call_count=state.model_call_count,
            tool_call_count=state.tool_call_count,
            seen_tool_calls=set(state.seen_tool_calls),
            tool_budget_exhausted=state.tool_budget_exhausted,
            tool_result_character_count=(
                state.tool_result_character_count
            ),
            tool_result_budget_exhausted=(
                state.tool_result_budget_exhausted
            ),
            request_idempotency_key=state.request_idempotency_key,
        )

    @staticmethod
    def _serialize_turn_state(state: _TurnState) -> dict:
        return {
            "model_call_count": state.model_call_count,
            "tool_call_count": state.tool_call_count,
            "seen_tool_calls": [
                list(signature)
                for signature in sorted(state.seen_tool_calls)
            ],
            "tool_budget_exhausted": state.tool_budget_exhausted,
            "tool_result_character_count": (
                state.tool_result_character_count
            ),
            "tool_result_budget_exhausted": (
                state.tool_result_budget_exhausted
            ),
            "request_idempotency_key": state.request_idempotency_key,
        }

    @staticmethod
    def _restore_turn_state(payload: dict) -> _TurnState:
        required_keys = {
            "model_call_count",
            "tool_call_count",
            "seen_tool_calls",
            "tool_budget_exhausted",
            "tool_result_character_count",
            "tool_result_budget_exhausted",
        }
        if (
            not isinstance(payload, dict)
            or frozenset(payload)
            not in {
                frozenset(required_keys),
                frozenset(
                    {
                        *required_keys,
                        "request_idempotency_key",
                    }
                ),
            }
        ):
            raise ValueError("待审批轮次状态字段非法")
        request_idempotency_key = payload.get(
            "request_idempotency_key"
        )
        if (
            request_idempotency_key is not None
            and (
                not isinstance(request_idempotency_key, str)
                or IDEMPOTENCY_KEY_PATTERN.fullmatch(
                    request_idempotency_key
                )
                is None
            )
        ):
            raise ValueError("待审批请求幂等键非法")
        integer_fields = (
            "model_call_count",
            "tool_call_count",
            "tool_result_character_count",
        )
        if any(
            type(payload[name]) is not int or payload[name] < 0
            for name in integer_fields
        ):
            raise ValueError("待审批轮次计数非法")
        boolean_fields = (
            "tool_budget_exhausted",
            "tool_result_budget_exhausted",
        )
        if any(
            type(payload[name]) is not bool
            for name in boolean_fields
        ):
            raise ValueError("待审批轮次预算状态非法")

        raw_signatures = payload["seen_tool_calls"]
        if not isinstance(raw_signatures, list):
            raise ValueError("待审批重复调用状态非法")
        signatures = set()
        for raw_signature in raw_signatures:
            if (
                not isinstance(raw_signature, list)
                or len(raw_signature) != 2
                or any(
                    not isinstance(value, str)
                    for value in raw_signature
                )
            ):
                raise ValueError("待审批重复调用签名非法")
            signatures.add(tuple(raw_signature))
        if len(signatures) != len(raw_signatures):
            raise ValueError("待审批重复调用签名重复")

        return _TurnState(
            model_call_count=payload["model_call_count"],
            tool_call_count=payload["tool_call_count"],
            seen_tool_calls=signatures,
            tool_budget_exhausted=payload["tool_budget_exhausted"],
            tool_result_character_count=(
                payload["tool_result_character_count"]
            ),
            tool_result_budget_exhausted=(
                payload["tool_result_budget_exhausted"]
            ),
            request_idempotency_key=request_idempotency_key,
        )

    def _committed_snapshot_hash(self) -> str:
        snapshot = {
            "version": SNAPSHOT_VERSION,
            "messages": messages_to_dict(self.messages),
            "memory_summary": self.memory_summary,
        }
        encoded = json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_pending_approval_record(
        self,
        *,
        working_messages: list,
        working_summary: str,
        current_turn_start_index: int,
        step: int,
        tools_allowed: bool,
        state: _TurnState,
        tool_call: dict,
        remaining_tool_calls: list[dict],
    ) -> dict:
        record = {
            "version": PENDING_APPROVAL_VERSION,
            "base_snapshot_sha256": self._committed_snapshot_hash(),
            "working_messages": messages_to_dict(working_messages),
            "working_summary": working_summary,
            "current_turn_start_index": current_turn_start_index,
            "step": step,
            "tools_allowed": tools_allowed,
            "turn_state": self._serialize_turn_state(state),
            "tool_call": deepcopy(tool_call),
            "remaining_tool_calls": deepcopy(remaining_tool_calls),
        }
        return json.loads(json.dumps(record, ensure_ascii=False))

    def export_pending_approval(self) -> dict | None:
        """导出最近一次可恢复审批；预览需在恢复时重新生成。"""
        if self._pending_approval is None:
            return None
        return deepcopy(self._pending_approval)

    @property
    def has_pending_approval(self) -> bool:
        return self._pending_approval is not None

    def pending_approval_event(self) -> ApprovalRequiredEvent | None:
        """返回不含预览和原始敏感参数的待审批摘要。"""
        if self._pending_approval is None:
            return None
        tool_call = self._pending_approval["tool_call"]
        return ApprovalRequiredEvent(
            tool_call_id=tool_call["id"],
            tool_name=tool_call["name"],
            args=_freeze(
                self._redact_tool_args(tool_call["args"])
            ),
            preview="",
        )

    def restore_pending_approval(self, record: dict) -> None:
        """验证并恢复一个未提交、等待重新审批的轮次。"""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("事件流活跃时不能恢复待审批轮次")
        try:
            restored = self._validate_pending_approval(record)
            self._pending_approval = restored
        finally:
            self._turn_lock.release()

    def _validate_pending_approval(self, record: dict) -> dict:
        if not isinstance(record, dict):
            raise ValueError("待审批记录必须是字典")
        try:
            candidate = deepcopy(record)
        except Exception as error:
            raise ValueError("无法复制待审批记录") from error

        required_keys = {
            "version",
            "base_snapshot_sha256",
            "working_messages",
            "working_summary",
            "current_turn_start_index",
            "step",
            "tools_allowed",
            "turn_state",
            "tool_call",
            "remaining_tool_calls",
        }
        if set(candidate) != required_keys:
            raise ValueError("待审批记录字段不完整或包含未知字段")
        if (
            type(candidate["version"]) is not int
            or candidate["version"] != PENDING_APPROVAL_VERSION
        ):
            raise ValueError("不支持的待审批记录版本")
        base_snapshot_sha256 = candidate["base_snapshot_sha256"]
        if (
            not isinstance(base_snapshot_sha256, str)
            or len(base_snapshot_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in base_snapshot_sha256
            )
            or base_snapshot_sha256
            != self._committed_snapshot_hash()
        ):
            raise ValueError("待审批记录对应的会话状态已经变化")
        summary = candidate["working_summary"]
        if (
            not isinstance(summary, str)
            or len(summary) > self.max_summary_characters
        ):
            raise ValueError("待审批长期摘要非法")

        try:
            working_messages = messages_from_dict(
                candidate["working_messages"]
            )
        except Exception as error:
            raise ValueError("待审批消息无法反序列化") from error
        current_index = candidate["current_turn_start_index"]
        if (
            type(current_index) is not int
            or current_index < 1
            or current_index >= len(working_messages)
            or not isinstance(
                working_messages[current_index],
                HumanMessage,
            )
            or not isinstance(working_messages[0], SystemMessage)
        ):
            raise ValueError("待审批当前轮次位置非法")
        if not self._is_current_system_message(
            working_messages[0],
            summary,
        ):
            raise ValueError("待审批系统消息非法")
        if any(
            not isinstance(
                message,
                (SystemMessage, HumanMessage, AIMessage, ToolMessage),
            )
            for message in working_messages
        ):
            raise ValueError("待审批消息类型非法")
        if any(
            isinstance(message, SystemMessage)
            for message in working_messages[1:]
        ) or any(
            isinstance(message, HumanMessage)
            for message in working_messages[current_index + 1 :]
        ):
            raise ValueError("待审批消息轮次结构非法")

        step = candidate["step"]
        if (
            type(step) is not int
            or step < 1
            or step >= self.max_agent_loops
            or candidate["tools_allowed"] is not True
        ):
            raise ValueError("待审批 Agent 步骤非法")
        state = self._restore_turn_state(candidate["turn_state"])
        if (
            state.model_call_count < step
            or state.tool_call_count > self.max_tool_calls
            or state.tool_result_character_count
            > self.max_tool_result_characters
        ):
            raise ValueError("待审批模型调用计数非法")

        tool_call = self._validate_pending_tool_call(
            candidate["tool_call"],
            require_recoverable=True,
        )
        remaining_tool_calls = [
            self._validate_pending_tool_call(
                pending_call,
                require_recoverable=False,
            )
            for pending_call in candidate["remaining_tool_calls"]
        ] if isinstance(candidate["remaining_tool_calls"], list) else None
        if remaining_tool_calls is None:
            raise ValueError("待审批剩余工具调用非法")

        unresolved_ids = [
            tool_call["id"],
            *(call["id"] for call in remaining_tool_calls),
        ]
        if len(set(unresolved_ids)) != len(unresolved_ids):
            raise ValueError("待审批工具调用 ID 重复")
        matching_ai_message = next(
            (
                message
                for message in reversed(
                    working_messages[current_index + 1 :]
                )
                if (
                    isinstance(message, AIMessage)
                    and message.tool_calls
                    and all(
                        call_id
                        in {
                            call["id"]
                            for call in message.tool_calls
                        }
                        for call_id in unresolved_ids
                    )
                )
            ),
            None,
        )
        if matching_ai_message is None:
            raise ValueError("待审批消息缺少对应工具调用")
        full_calls = [
            {
                "id": call["id"],
                "name": call["name"],
                "args": call["args"],
            }
            for call in matching_ai_message.tool_calls
        ]
        current_call_index = next(
            (
                index
                for index, call in enumerate(full_calls)
                if call["id"] == tool_call["id"]
            ),
            None,
        )
        expected_unresolved_calls = [
            {
                "id": call["id"],
                "name": call["name"],
                "args": call["args"],
            }
            for call in [tool_call, *remaining_tool_calls]
        ]
        if (
            current_call_index is None
            or full_calls[current_call_index:]
            != expected_unresolved_calls
        ):
            raise ValueError("待审批工具调用与消息记录不一致")
        completed_ids = {
            message.tool_call_id
            for message in working_messages
            if isinstance(message, ToolMessage)
        }
        if any(call_id in completed_ids for call_id in unresolved_ids):
            raise ValueError("待审批工具调用已经存在结果")

        candidate["working_messages"] = messages_to_dict(
            working_messages
        )
        candidate["tool_call"] = tool_call
        candidate["remaining_tool_calls"] = remaining_tool_calls
        candidate["turn_state"] = self._serialize_turn_state(state)
        return json.loads(json.dumps(candidate, ensure_ascii=False))

    def _validate_pending_tool_call(
        self,
        tool_call,
        *,
        require_recoverable: bool,
    ) -> dict:
        if (
            not isinstance(tool_call, dict)
            or not isinstance(tool_call.get("id"), str)
            or not tool_call["id"]
            or not isinstance(tool_call.get("name"), str)
            or not tool_call["name"]
            or not isinstance(tool_call.get("args"), dict)
        ):
            raise ValueError("待审批工具调用非法")
        tool_name = tool_call["name"]
        selected_tool = self.tools_by_name.get(tool_name)
        if selected_tool is None:
            raise ValueError("待审批工具不再可用")
        self._validate_tool_arguments(
            selected_tool,
            tool_call["args"],
        )
        if require_recoverable and (
            tool_name not in self.approval_required_tools
            or tool_name not in self.approval_preparers
        ):
            raise ValueError("待审批工具不支持安全恢复")
        return deepcopy(tool_call)

    def stream_resume_pending_approval(
        self,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        """重新准备预览、获取新审批并继续未提交轮次。"""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("同一 WorkspaceAgent 不能同时运行多个对话轮次")
        if self._pending_approval is None:
            self._turn_lock.release()
            raise RuntimeError("当前没有可恢复的待审批轮次")

        cancellation_token = CancellationToken()
        self._active_cancellation_token = cancellation_token
        try:
            try:
                yield from self._run_pending_approval_transaction(
                    deepcopy(self._pending_approval)
                )
            except ToolExecutionCancelled:
                yield TurnCancelledEvent(
                    reason=cancellation_token.reason,
                )
        finally:
            self._active_cancellation_token = None
            self._turn_lock.release()

    def _run_pending_approval_transaction(
        self,
        record: dict,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        record = self._validate_pending_approval(record)
        working_messages = messages_from_dict(
            record["working_messages"]
        )
        working_summary = record["working_summary"]
        current_turn_start_index = record[
            "current_turn_start_index"
        ]
        state = self._restore_turn_state(record["turn_state"])
        step = record["step"]
        tools_allowed = record["tools_allowed"]
        tool_calls = [
            record["tool_call"],
            *record["remaining_tool_calls"],
        ]

        for tool_index, tool_call in enumerate(tool_calls):
            state_before = self._copy_turn_state(state)
            yield from self._execute_tool_call_with_recovery(
                tool_call=tool_call,
                remaining_tool_calls=tool_calls[tool_index + 1 :],
                step=step,
                tools_allowed=tools_allowed,
                state=state,
                state_before=state_before,
                working_messages=working_messages,
                working_summary=working_summary,
                current_turn_start_index=current_turn_start_index,
            )

        yield from self._continue_turn_transaction(
            working_messages=working_messages,
            working_summary=working_summary,
            current_turn_start_index=current_turn_start_index,
            state=state,
            start_step=step + 1,
        )

    def _validate_current_turn_citations(
        self,
        current_turn_messages: list,
    ) -> CitationValidationEvent | None:
        if self.citation_validator is None:
            return None

        try:
            immutable_messages = tuple(deepcopy(current_turn_messages))
            result = self.citation_validator(immutable_messages)
            if not isinstance(result, CitationValidationEvent):
                raise TypeError(
                    "citation_validator 必须返回 CitationValidationEvent"
                )
            allowed_statuses = {
                "valid",
                "missing",
                "unknown",
                "not_applicable",
                "error",
            }
            counts = (
                result.citation_count,
                result.valid_citation_count,
                result.unknown_citation_count,
                result.retrieved_chunk_count,
            )
            if (
                result.status not in allowed_statuses
                or any(type(value) is not int or value < 0 for value in counts)
            ):
                raise ValueError("citation_validator 返回了无效结果")
            return result
        except Exception:
            return CitationValidationEvent(
                status="error",
                citation_count=0,
                valid_citation_count=0,
                unknown_citation_count=0,
                retrieved_chunk_count=0,
            )

    def _citation_policy_event(
        self,
        validation_event: CitationValidationEvent | None,
        *,
        guarded_tool_used: bool,
    ) -> CitationPolicyEvent | None:
        if validation_event is None:
            return None
        if self.citation_policy == "observe":
            action = "observed"
        elif guarded_tool_used and validation_event.status != "valid":
            action = "blocked"
        else:
            action = "allowed"
        return CitationPolicyEvent(
            policy=self.citation_policy,
            action=action,
            validation_status=validation_event.status,
        )

    def _guarded_tool_was_used(
        self,
        current_turn_messages: list,
    ) -> bool:
        return any(
            isinstance(message, ToolMessage)
            and message.name in self.citation_guard_tool_names
            for message in current_turn_messages
        )

    def _trim_history(self, messages: list) -> list:
        return trim_messages(
            messages,
            max_tokens=self.max_context_tokens,
            token_counter=self.token_counter,
            strategy="last",
            include_system=True,
            start_on=HumanMessage,
            allow_partial=False,
        )

    def _with_fitting_summary(
        self,
        messages: list,
        summary: str,
    ) -> list:
        messages_with_summary = list(messages)
        limited_summary = summary[: max(0, self.max_summary_characters)]

        def build_messages(summary_length: int) -> list:
            fitted_messages = list(messages_with_summary)
            fitted_messages[0] = self._build_system_message(
                limited_summary[:summary_length]
            )
            return fitted_messages

        if not limited_summary:
            return build_messages(0)
        if self._messages_fit(build_messages(len(limited_summary))):
            return build_messages(len(limited_summary))

        lower_bound = 0
        upper_bound = len(limited_summary)
        best_length = 0
        while lower_bound <= upper_bound:
            middle = (lower_bound + upper_bound) // 2
            if self._messages_fit(build_messages(middle)):
                best_length = middle
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

        return build_messages(best_length)

    def _messages_fit(self, messages: list) -> bool:
        trimmed_messages = self._trim_history(messages)
        return (
            len(trimmed_messages) == len(messages)
            and self._contains_required_context(
                trimmed_messages,
                messages[-1],
            )
        )

    @staticmethod
    def _build_system_message(summary: str) -> SystemMessage:
        if not summary:
            return SystemMessage(content=SYSTEM_PROMPT)
        return SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                "长期记忆摘要：\n"
                f"{summary}"
            )
        )

    def _extract_removed_turns(self, working_messages: list) -> list[list]:
        retained_message_ids = {id(message) for message in working_messages}
        history_turns = []
        current_turn = []

        for message in self.messages[1:]:
            if isinstance(message, HumanMessage):
                if current_turn:
                    history_turns.append(current_turn)
                current_turn = [message]
            elif current_turn:
                current_turn.append(message)
        if current_turn:
            history_turns.append(current_turn)

        return [
            turn
            for turn in history_turns
            if (
                isinstance(turn[-1], AIMessage)
                and not turn[-1].tool_calls
                and all(id(message) not in retained_message_ids for message in turn)
            )
        ]

    def _update_memory_summary(
        self,
        existing_summary: str,
        removed_turns: list[list],
    ) -> str | None:
        removed_dialogue = self._format_removed_turns(removed_turns)
        if not removed_dialogue:
            return None

        summary_messages = [
            SystemMessage(
                content=(
                    "你负责维护项目对话的长期记忆摘要。"
                    "请重点保留用户偏好、项目目标、已作出的决定、"
                    "未完成事项和重要结论。"
                    "不要补充原对话中不存在的信息，"
                    f"输出不超过 {self.max_summary_characters} 个字符。"
                )
            ),
            HumanMessage(
                content=(
                    "已有摘要：\n"
                    f"{existing_summary or '（无）'}\n\n"
                    "新删除的完整对话轮次：\n"
                    f"{removed_dialogue}"
                )
            ),
        ]

        try:
            summary_parts = []
            cancellation_token = self._active_cancellation_token
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            for chunk in self.summary_model.stream(summary_messages):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                if isinstance(chunk.content, str):
                    text = chunk.content
                else:
                    text = chunk.text
                if text:
                    summary_parts.append(text)
        except ToolExecutionCancelled:
            raise
        except Exception:
            return None

        updated_summary = "".join(summary_parts).strip()
        if not updated_summary:
            return None
        return updated_summary[: max(0, self.max_summary_characters)]

    @classmethod
    def _format_removed_turns(cls, removed_turns: list[list]) -> str:
        formatted_turns = []

        for turn in removed_turns:
            user_text = cls._message_text(turn[0])
            final_answer = ""
            for message in turn[1:]:
                if not isinstance(message, AIMessage):
                    continue
                message_text = cls._message_text(message)
                if message_text:
                    final_answer = message_text

            if user_text and final_answer:
                formatted_turns.append(
                    f"用户：{user_text}\n助手：{final_answer}"
                )

        return "\n\n".join(formatted_turns)

    @staticmethod
    def _message_text(message) -> str:
        if isinstance(message.content, str):
            return message.content
        return message.text

    @staticmethod
    def _contains_required_context(
        messages: list,
        current_message: HumanMessage,
    ) -> bool:
        return (
            len(messages) >= 2
            and isinstance(messages[0], SystemMessage)
            and messages[-1] == current_message
        )

    @classmethod
    def _remove_incomplete_tool_groups(
        cls,
        messages: list,
        current_message: HumanMessage,
    ) -> list:
        valid_messages = list(messages)
        while not cls._has_complete_tool_groups(valid_messages):
            next_turn_index = next(
                (
                    index
                    for index, message in enumerate(valid_messages[2:], start=2)
                    if isinstance(message, HumanMessage)
                ),
                None,
            )
            if next_turn_index is None:
                return [valid_messages[0], current_message]
            valid_messages = [
                valid_messages[0],
                *valid_messages[next_turn_index:],
            ]
        return valid_messages

    @staticmethod
    def _has_complete_tool_groups(messages: list) -> bool:
        pending_tool_call_ids = None

        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                if pending_tool_call_ids:
                    return False
                pending_tool_call_ids = {
                    tool_call["id"] for tool_call in message.tool_calls
                }
                continue

            if isinstance(message, ToolMessage):
                if (
                    pending_tool_call_ids is None
                    or message.tool_call_id not in pending_tool_call_ids
                ):
                    return False
                pending_tool_call_ids.remove(message.tool_call_id)
                continue

            if pending_tool_call_ids:
                return False
            pending_tool_call_ids = None

        return not pending_tool_call_ids

    @staticmethod
    def _has_only_committed_turns(messages: list) -> bool:
        if len(messages) == 1:
            return isinstance(messages[0], SystemMessage)

        index = 1
        while index < len(messages):
            if not isinstance(messages[index], HumanMessage):
                return False
            index += 1

            final_answer_seen = False
            while index < len(messages):
                ai_message = messages[index]
                if not isinstance(ai_message, AIMessage):
                    return False
                index += 1

                if not ai_message.tool_calls:
                    final_answer_seen = True
                    break

                pending_tool_call_ids = {
                    tool_call["id"]
                    for tool_call in ai_message.tool_calls
                }
                while (
                    index < len(messages)
                    and isinstance(messages[index], ToolMessage)
                ):
                    tool_call_id = messages[index].tool_call_id
                    if tool_call_id not in pending_tool_call_ids:
                        return False
                    pending_tool_call_ids.remove(tool_call_id)
                    index += 1
                if pending_tool_call_ids:
                    return False

            if not final_answer_seen:
                return False

        return True

    def _stream_response(
        self,
        active_model,
        working_messages: list,
        call_index: int,
        cancellation_token: CancellationToken,
    ) -> Generator[AgentEvent, None, AIMessage | None]:
        response_chunk = None
        started_at = self.monotonic_clock()
        first_chunk_at = None

        try:
            cancellation_token.raise_if_cancelled()
            stream_method = getattr(active_model, "stream", None)
            async_stream_method = getattr(active_model, "astream", None)
            if (
                self._prefer_async_execution
                and callable(async_stream_method)
            ):
                chunks = iterate_async_synchronously(
                    async_stream_method(working_messages)
                )
            elif callable(stream_method):
                chunks = stream_method(working_messages)
            elif callable(async_stream_method):
                chunks = iterate_async_synchronously(
                    async_stream_method(working_messages)
                )
            else:
                raise TypeError(
                    "模型必须实现 stream() 或 astream()"
                )

            for chunk in chunks:
                cancellation_token.raise_if_cancelled()
                if first_chunk_at is None:
                    first_chunk_at = self.monotonic_clock()

                if isinstance(chunk.content, str):
                    text = chunk.content
                else:
                    text = chunk.text

                if text:
                    yield TokenEvent(text=text)

                if response_chunk is None:
                    response_chunk = chunk
                else:
                    response_chunk = response_chunk + chunk

            response = self._message_from_response_chunk(response_chunk)
            finished_at = self.monotonic_clock()
            input_tokens, output_tokens, total_tokens, token_source = (
                self._extract_usage_metadata(response)
            )
            yield ModelCallMetricsEvent(
                call_index=call_index,
                status="success",
                duration_ms=self._elapsed_ms(
                    started_at,
                    finished_at,
                ),
                first_chunk_ms=(
                    self._elapsed_ms(started_at, first_chunk_at)
                    if first_chunk_at is not None
                    else None
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                token_source=token_source,
            )
            return response
        except Exception as error:
            finished_at = self.monotonic_clock()
            yield ModelCallMetricsEvent(
                call_index=call_index,
                status="error",
                duration_ms=self._elapsed_ms(
                    started_at,
                    finished_at,
                ),
                first_chunk_ms=(
                    self._elapsed_ms(started_at, first_chunk_at)
                    if first_chunk_at is not None
                    else None
                ),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                token_source="unavailable",
                error_type=type(error).__name__,
            )
            raise

    @staticmethod
    def _elapsed_ms(started_at: float, finished_at: float) -> int:
        return int(round(max(0.0, finished_at - started_at) * 1000))

    @classmethod
    def _message_from_response_chunk(cls, response_chunk):
        if response_chunk is None:
            return None

        usage = getattr(response_chunk, "usage_metadata", None)
        if (
            usage is not None
            and cls._validated_usage_metadata(usage) is None
        ):
            response_chunk = response_chunk.model_copy(
                update={"usage_metadata": None}
            )
        return message_chunk_to_message(response_chunk)

    @classmethod
    def _extract_usage_metadata(
        cls,
        response: AIMessage | None,
    ) -> tuple[int | None, int | None, int | None, str]:
        usage = (
            getattr(response, "usage_metadata", None)
            if response is not None
            else None
        )
        token_values = cls._validated_usage_metadata(usage)
        if token_values is None:
            return None, None, None, "unavailable"
        return *token_values, "provider"

    @staticmethod
    def _validated_usage_metadata(
        usage,
    ) -> tuple[int, int, int] | None:
        if not isinstance(usage, dict):
            return None
        token_keys = (
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
        token_values = tuple(usage.get(key) for key in token_keys)
        if any(
            type(value) is not int or value < 0
            for value in token_values
        ):
            return None
        return token_values

    def _execute_tool_call(
        self,
        tool_call,
        step: int,
        tools_allowed: bool,
        state: _TurnState,
        working_messages: list,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        tool_call_id = tool_call["id"]
        tool_name = tool_call["name"]
        internal_args = deepcopy(tool_call["args"])
        event_args = _freeze(self._redact_tool_args(internal_args))
        signature = (
            tool_name,
            json.dumps(
                internal_args,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        yield ToolCallEvent(
            tool_call_id=tool_call_id,
            step=step,
            name=tool_name,
            args=event_args,
        )

        if not tools_allowed:
            detail = "当前轮次禁止调用工具"
            self._append_control_tool_message(
                messages=working_messages,
                content=(
                    "当前轮次不允许调用工具，本次调用未执行。"
                    "请根据已有信息直接回答。"
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        if (
            state.tool_result_budget_exhausted
            or state.tool_result_character_count
            >= self.max_tool_result_characters
        ):
            state.tool_result_budget_exhausted = True
            detail = "工具结果预算已耗尽"
            self._append_control_tool_message(
                messages=working_messages,
                content=(
                    "工具结果预算已耗尽，本次调用未执行。"
                    "请根据已有信息回答。"
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        if signature in state.seen_tool_calls:
            detail = "重复调用"
            self._append_control_tool_message(
                messages=working_messages,
                content="重复工具调用已跳过，请使用之前相同工具调用的结果。",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        state.seen_tool_calls.add(signature)

        if state.tool_call_count >= self.max_tool_calls:
            state.tool_budget_exhausted = True
            detail = "工具预算已耗尽"
            self._append_control_tool_message(
                messages=working_messages,
                content=(
                    "工具预算已耗尽，本次调用未执行。"
                    "请根据已有信息回答。"
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
            )
            return

        selected_tool = self.tools_by_name.get(tool_name)
        if selected_tool is None:
            character_count, truncated = self._append_tool_message(
                messages=working_messages,
                content=f"未知工具：{tool_name}",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                state=state,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
                character_count=character_count,
                detail="未知工具",
                truncated=truncated,
            )
            return

        try:
            self._validate_tool_arguments(
                selected_tool,
                internal_args,
            )
        except Exception as error:
            character_count, truncated = self._append_tool_message(
                messages=working_messages,
                content=f"工具参数非法：{error}",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                state=state,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
                character_count=character_count,
                detail="参数非法",
                truncated=truncated,
            )
            return

        prepared_action = None
        if tool_name in self.approval_required_tools:
            preview = ""
            previewer = self.approval_previewers.get(tool_name)
            preparer = self.approval_preparers.get(tool_name)
            if previewer is not None or preparer is not None:
                try:
                    if preparer is not None:
                        prepared_action = self._invoke_approval_preparer(
                            preparer,
                            internal_args,
                        )
                        preview = prepared_action.preview
                    else:
                        preview = self._invoke_approval_previewer(
                            previewer,
                            internal_args,
                        )
                except Exception as error:
                    if preparer is not None:
                        error_detail = "审批准备失败"
                    else:
                        error_detail = "变更预览失败"
                    preview_error = f"{error_detail}：{error}"
                    character_count, truncated = self._append_tool_message(
                        messages=working_messages,
                        content=preview_error,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        state=state,
                    )
                    yield ToolResultEvent(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                        character_count=character_count,
                        detail=error_detail,
                        truncated=truncated,
                    )
                    return

            decision = yield ApprovalRequiredEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                args=event_args,
                preview=preview,
            )
            approval_outcome = self._classify_approval_decision(
                decision,
                tool_call_id,
            )
            yield ApprovalResolvedEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                outcome=approval_outcome,
            )
            if approval_outcome != "approved":
                detail_by_outcome = {
                    "rejected": "用户未批准",
                    "missing": "缺少审批决定",
                    "mismatched": "审批调用 ID 不匹配",
                    "invalid": "审批决定无效",
                }
                detail = detail_by_outcome[approval_outcome]
                self._append_control_tool_message(
                    messages=working_messages,
                    content=(
                        f"工具 {tool_name} 的审批结果为"
                        f"“{detail}”，"
                        "本次调用未执行。请根据已有信息继续回答。"
                    ),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )
                yield ToolResultEvent(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="skipped",
                    character_count=0,
                    detail=detail,
                )
                return

        state.tool_call_count += 1
        if state.tool_call_count >= self.max_tool_calls:
            state.tool_budget_exhausted = True

        execution_started_at = self.monotonic_clock()
        try:
            execution_policy = (
                self.tool_execution_middleware.policy_for(tool_name)
            )
            idempotency_key = internal_args.get("idempotency_key")
            if (
                idempotency_key is None
                and state.request_idempotency_key is not None
            ):
                idempotency_key = hashlib.sha256(
                    (
                        f"{state.request_idempotency_key}\0"
                        f"{tool_call_id}\0{tool_name}"
                    ).encode("utf-8")
                ).hexdigest()
            if prepared_action is not None:
                action = prepared_action.execute
            elif execution_policy.cooperative_cancellation:
                cooperative_invoke = getattr(
                    selected_tool,
                    "invoke_with_cancellation",
                    None,
                )
                if not callable(cooperative_invoke):
                    raise TypeError(
                        "协作式取消工具必须实现 "
                        "invoke_with_cancellation(args, cancellation_token)"
                    )

                def action(cancellation_token):
                    if execution_policy.risk == "external_side_effect":
                        return cooperative_invoke(
                            deepcopy(internal_args),
                            cancellation_token,
                            idempotency_key=idempotency_key,
                        )
                    return cooperative_invoke(
                        deepcopy(internal_args),
                        cancellation_token,
                    )
            else:
                action = lambda: self._invoke_registered_tool(
                    selected_tool,
                    internal_args,
                    idempotency_key=idempotency_key,
                )
            cancellation_token = self._active_cancellation_token
            if cancellation_token is None:
                raise ToolExecutionCancelled(
                    "The active turn is no longer available."
                )
            raw_tool_result = self.tool_execution_middleware.execute(
                tool_name,
                action,
                cancellation_token,
                idempotency_key=idempotency_key,
            )
        except ToolExecutionCancelled:
            execution_finished_at = self.monotonic_clock()
            self._append_control_tool_message(
                messages=working_messages,
                content="当前轮次已取消，本次工具调用未完成。",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail="轮次已取消",
                duration_ms=self._elapsed_ms(
                    execution_started_at,
                    execution_finished_at,
                ),
                error_type=ToolExecutionCancelled.__name__,
            )
            raise
        except ToolExecutionTimeout:
            execution_finished_at = self.monotonic_clock()
            self._append_control_tool_message(
                messages=working_messages,
                content=(
                    "工具执行超过允许时间，本次调用未获得结果。"
                    "请根据已有信息回答或稍后重试。"
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
                character_count=0,
                detail="工具执行超时",
                duration_ms=self._elapsed_ms(
                    execution_started_at,
                    execution_finished_at,
                ),
                error_type=ToolExecutionTimeout.__name__,
            )
            return
        except ToolActionConflictError as error:
            execution_finished_at = self.monotonic_clock()
            self._append_control_tool_message(
                messages=working_messages,
                content=(
                    f"工具执行冲突：{error}\n"
                    "本次写入未执行，外部修改已保留。"
                    "请根据最新状态重新读取或重新发起写入。"
                ),
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
                character_count=0,
                detail="写入冲突",
                duration_ms=self._elapsed_ms(
                    execution_started_at,
                    execution_finished_at,
                ),
                error_type=type(error).__name__,
            )
            return
        except Exception as error:
            execution_finished_at = self.monotonic_clock()
            tool_result_text = f"工具执行失败：{error}"
            tool_status = "error"
            tool_error_type = type(error).__name__
        else:
            execution_finished_at = self.monotonic_clock()
            try:
                tool_result_text = str(raw_tool_result)
                tool_status = "success"
                tool_error_type = ""
            except Exception as error:
                tool_result_text = f"工具结果转换失败：{error}"
                tool_status = "error"
                tool_error_type = type(error).__name__

        character_count, truncated = self._append_tool_message(
            messages=working_messages,
            content=tool_result_text,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            state=state,
        )
        yield ToolResultEvent(
            tool_call_id=tool_call_id,
            name=tool_name,
            status=tool_status,
            character_count=character_count,
            truncated=truncated,
            duration_ms=self._elapsed_ms(
                execution_started_at,
                execution_finished_at,
            ),
            error_type=tool_error_type,
        )

    @classmethod
    def _redact_tool_args(cls, value):
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                if cls._is_sensitive_argument_name(key):
                    redacted[key] = (
                        f"<{cls._argument_character_count(item)} characters>"
                    )
                else:
                    redacted[key] = cls._redact_tool_args(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact_tool_args(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._redact_tool_args(item) for item in value)
        return deepcopy(value)

    @staticmethod
    def _is_sensitive_argument_name(name) -> bool:
        normalized_name = str(name).lower().replace("-", "_")
        return any(
            sensitive_name in normalized_name
            for sensitive_name in REDACTED_ARGUMENT_NAMES
        )

    @staticmethod
    def _argument_character_count(value) -> int:
        if isinstance(value, str):
            return len(value)
        try:
            return len(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError):
            return len(str(value))

    @staticmethod
    def _invoke_approval_previewer(
        previewer: Callable[..., str],
        tool_args: dict,
    ) -> str:
        preview = WorkspaceAgent._invoke_approval_handler(
            previewer,
            tool_args,
        )
        if not isinstance(preview, str):
            raise TypeError("approval_previewer 必须返回 str")
        return preview

    @staticmethod
    def _invoke_approval_preparer(
        preparer: Callable[..., PreparedToolAction],
        tool_args: dict,
    ) -> PreparedToolAction:
        prepared_action = WorkspaceAgent._invoke_approval_handler(
            preparer,
            tool_args,
        )
        if not isinstance(prepared_action, PreparedToolAction):
            raise TypeError(
                "approval_preparer 必须返回 PreparedToolAction"
            )
        if not isinstance(prepared_action.preview, str):
            raise TypeError(
                "PreparedToolAction.preview 必须是 str"
            )
        if not callable(prepared_action.execute):
            raise TypeError(
                "PreparedToolAction.execute 必须可调用"
            )
        return prepared_action

    @staticmethod
    def _invoke_approval_handler(
        handler: Callable,
        tool_args: dict,
    ):
        handler_args = deepcopy(tool_args)
        if hasattr(handler, "invoke"):
            return handler.invoke(handler_args)
        return handler(**handler_args)

    @staticmethod
    def _classify_approval_decision(
        decision,
        tool_call_id: str,
    ) -> str:
        if decision is None:
            return "missing"
        if not isinstance(decision, ApprovalDecision):
            return "invalid"
        if decision.tool_call_id != tool_call_id:
            return "mismatched"
        if type(decision.approved) is not bool:
            return "invalid"
        return "approved" if decision.approved else "rejected"

    @staticmethod
    def _validate_tool_arguments(selected_tool, tool_args: dict) -> None:
        get_input_schema = getattr(
            selected_tool,
            "get_input_schema",
            None,
        )
        if not callable(get_input_schema):
            return
        input_schema = get_input_schema()
        model_validate = getattr(input_schema, "model_validate", None)
        if callable(model_validate):
            model_validate(tool_args)
            return
        parse_obj = getattr(input_schema, "parse_obj", None)
        if callable(parse_obj):
            parse_obj(tool_args)

    def _invoke_registered_tool(
        self,
        selected_tool,
        tool_args: dict,
        *,
        idempotency_key: str | None = None,
    ):
        coroutine = getattr(selected_tool, "coroutine", None)
        synchronous_function = getattr(selected_tool, "func", None)
        async_invoke = getattr(selected_tool, "ainvoke", None)
        if (
            self._prefer_async_execution
            and callable(async_invoke)
            and callable(coroutine)
        ):
            return run_coroutine_synchronously(
                async_invoke(
                    deepcopy(tool_args),
                    config=self._tool_invocation_config(
                        idempotency_key
                    ),
                )
            )
        if callable(synchronous_function) or not callable(async_invoke):
            return selected_tool.invoke(
                deepcopy(tool_args),
                config=self._tool_invocation_config(
                    idempotency_key
                ),
            )
        return run_coroutine_synchronously(
            async_invoke(
                deepcopy(tool_args),
                config=self._tool_invocation_config(
                    idempotency_key
                ),
            )
        )

    @staticmethod
    def _tool_invocation_config(
        idempotency_key: str | None,
    ) -> dict | None:
        if idempotency_key is None:
            return None
        return {
            "configurable": {
                "idempotency_key": idempotency_key,
            }
        }

    def _append_tool_message(
        self,
        messages: list,
        content: str,
        tool_call_id: str,
        tool_name: str,
        state: _TurnState,
    ) -> tuple[int, bool]:
        remaining_characters = max(
            0,
            self.max_tool_result_characters
            - state.tool_result_character_count,
        )
        limited_content, truncated = self._limit_tool_result(
            content,
            remaining_characters,
        )
        messages.append(
            ToolMessage(
                content=limited_content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )
        character_count = len(limited_content)
        state.tool_result_character_count += character_count
        if (
            state.tool_result_character_count
            >= self.max_tool_result_characters
        ):
            state.tool_result_budget_exhausted = True
        return character_count, truncated

    @staticmethod
    def _append_control_tool_message(
        messages: list,
        content: str,
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    @staticmethod
    def _limit_tool_result(
        content: str,
        remaining_characters: int,
    ) -> tuple[str, bool]:
        if len(content) <= remaining_characters:
            return content, False
        if remaining_characters <= 0:
            return "", True

        marker = TOOL_RESULT_TRUNCATION_MARKER
        if remaining_characters <= len(marker):
            return marker[-remaining_characters:], True

        prefix_length = remaining_characters - len(marker)
        return f"{content[:prefix_length]}{marker}", True

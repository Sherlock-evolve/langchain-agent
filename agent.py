import json
import time
from collections.abc import Callable, Generator
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock

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

from contracts import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequiredEvent,
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
)


SYSTEM_PROMPT = (
    "你是一位人工智能老师，也是当前项目的工作区助手。"
    "需要了解项目文件时，请使用工具获取真实信息，不要猜测。"
    "请用通俗、准确的方式回答。"
)
TOOL_RESULT_TRUNCATION_MARKER = "\n[工具结果已截断]"
SNAPSHOT_VERSION = 1
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
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]
        self.memory_summary = ""
        self._turn_lock = Lock()

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
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        """运行一轮并产生事件；提前停止消费时应关闭返回的生成器。"""
        if not self._turn_lock.acquire(blocking=False):
            raise RuntimeError("同一 WorkspaceAgent 不能同时运行多个对话轮次")

        try:
            yield from self._run_turn_transaction(question)
        finally:
            self._turn_lock.release()

    def _run_turn_transaction(
        self,
        question: str,
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

        state = _TurnState()
        answered = False
        task_stopped = False

        for step in range(1, self.max_agent_loops + 1):
            tools_allowed = (
                step < self.max_agent_loops
                and not state.tool_budget_exhausted
                and not state.tool_result_budget_exhausted
            )
            active_model = self.model_with_tools if tools_allowed else self.model
            state.model_call_count += 1
            response = yield from self._stream_response(
                active_model,
                working_messages,
                call_index=state.model_call_count,
            )

            if response is None:
                yield SystemEvent(
                    message="模型未返回任何消息，当前任务已停止。"
                )
                task_stopped = True
                break

            working_messages.append(response)

            if not response.tool_calls:
                memory_changed = working_summary != self.memory_summary
                self.messages = working_messages
                self.memory_summary = working_summary
                answered = True
                if memory_changed:
                    yield MemoryUpdatedEvent(
                        character_count=len(self.memory_summary),
                    )
                break

            for tool_call in response.tool_calls:
                yield from self._execute_tool_call(
                    tool_call=tool_call,
                    step=step,
                    tools_allowed=tools_allowed,
                    state=state,
                    working_messages=working_messages,
                )

        if not answered and not task_stopped:
            yield SystemEvent(
                message=(
                    f"Agent 循环达到 {self.max_agent_loops} 次上限，"
                    "已停止。"
                )
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
            for chunk in self.summary_model.stream(summary_messages):
                if isinstance(chunk.content, str):
                    text = chunk.content
                else:
                    text = chunk.text
                if text:
                    summary_parts.append(text)
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
    ) -> Generator[AgentEvent, None, AIMessage | None]:
        response_chunk = None
        started_at = self.monotonic_clock()
        first_chunk_at = None

        try:
            for chunk in active_model.stream(working_messages):
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
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="skipped",
                character_count=0,
                detail=detail,
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
            if (
                not isinstance(decision, ApprovalDecision)
                or decision.tool_call_id != tool_call_id
                or decision.approved is not True
            ):
                if (
                    isinstance(decision, ApprovalDecision)
                    and decision.tool_call_id != tool_call_id
                ):
                    detail = "审批调用 ID 不匹配"
                else:
                    detail = "用户未批准"
                self._append_control_tool_message(
                    messages=working_messages,
                    content=(
                        f"用户未批准执行工具 {tool_name}，"
                        "本次调用未执行。请根据已有信息继续回答。"
                    ),
                    tool_call_id=tool_call_id,
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

        try:
            if prepared_action is not None:
                tool_result_text = str(prepared_action.execute())
            else:
                selected_tool = self.tools_by_name[tool_name]
                tool_result_text = str(selected_tool.invoke(internal_args))
            tool_status = "success"
        except ToolActionConflictError as error:
            self._append_control_tool_message(
                messages=working_messages,
                content=(
                    f"工具执行冲突：{error}\n"
                    "本次写入未执行，外部修改已保留。"
                    "请根据最新状态重新读取或重新发起写入。"
                ),
                tool_call_id=tool_call_id,
            )
            yield ToolResultEvent(
                tool_call_id=tool_call_id,
                name=tool_name,
                status="error",
                character_count=0,
                detail="写入冲突",
            )
            return
        except Exception as error:
            tool_result_text = f"工具执行失败：{error}"
            tool_status = "error"

        character_count, truncated = self._append_tool_message(
            messages=working_messages,
            content=tool_result_text,
            tool_call_id=tool_call_id,
            state=state,
        )
        yield ToolResultEvent(
            tool_call_id=tool_call_id,
            name=tool_name,
            status=tool_status,
            character_count=character_count,
            truncated=truncated,
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

    def _append_tool_message(
        self,
        messages: list,
        content: str,
        tool_call_id: str,
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
    ) -> None:
        messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
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

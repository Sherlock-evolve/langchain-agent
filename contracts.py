from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TokenEvent:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    tool_call_id: str
    step: int
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResultEvent:
    tool_call_id: str
    name: str
    status: Literal["success", "error", "skipped"]
    character_count: int
    detail: str = ""
    truncated: bool = False


@dataclass(frozen=True)
class ApprovalRequiredEvent:
    tool_call_id: str
    tool_name: str
    args: dict
    preview: str = ""


@dataclass(frozen=True)
class ApprovalDecision:
    tool_call_id: str
    approved: bool


@dataclass(frozen=True)
class PreparedToolAction:
    preview: str
    execute: Callable[[], str]


class ToolActionConflictError(RuntimeError):
    """工具预览后的目标状态发生变化。"""


@dataclass(frozen=True)
class SystemEvent:
    message: str


@dataclass(frozen=True)
class ContextTrimmedEvent:
    removed_message_count: int
    remaining_message_count: int


@dataclass(frozen=True)
class MemoryUpdatedEvent:
    character_count: int


@dataclass(frozen=True)
class SessionSavedEvent:
    session_id: str


AgentEvent = (
    TokenEvent
    | ToolCallEvent
    | ToolResultEvent
    | ApprovalRequiredEvent
    | SystemEvent
    | ContextTrimmedEvent
    | MemoryUpdatedEvent
    | SessionSavedEvent
)


@dataclass(frozen=True)
class EventEnvelope:
    session_id: str
    turn_id: str
    sequence: int
    event: AgentEvent

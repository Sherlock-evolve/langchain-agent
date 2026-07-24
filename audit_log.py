import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import session_store
from contracts import (
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


AUDIT_SCHEMA_VERSION = 1
AUDIT_DIRECTORY_NAME = ".agent_audit"
AUDIT_LOG_ROOT = Path(__file__).resolve().parent / AUDIT_DIRECTORY_NAME
AUDIT_FILE_SUFFIX = ".jsonl"
MAX_AUDIT_RECORD_BYTES = 64 * 1024
MAX_AUDIT_LOG_BYTES = 10 * 1024 * 1024
MAX_TURN_ID_LENGTH = 128


class AuditLogError(Exception):
    """审计记录无法安全编码或写入。"""


class UnsupportedAuditEventError(AuditLogError, TypeError):
    """事件类型不在审计白名单中。"""


class AuditLogLimitError(AuditLogError):
    """审计记录或日志超过大小限制。"""


class JsonlAuditLogger:
    """将事件信封按白名单编码后安全追加到独立 JSONL 日志。"""

    _process_lock = Lock()

    def __init__(
        self,
        root: Path | str | None = None,
        timestamp_factory=None,
        max_record_bytes: int = MAX_AUDIT_RECORD_BYTES,
        max_log_bytes: int = MAX_AUDIT_LOG_BYTES,
    ):
        if timestamp_factory is not None and not callable(timestamp_factory):
            raise TypeError("timestamp_factory 必须可调用")
        for name, value in (
            ("max_record_bytes", max_record_bytes),
            ("max_log_bytes", max_log_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是正整数")

        self.root = (
            Path(root)
            if root is not None
            else AUDIT_LOG_ROOT
        )
        self.timestamp_factory = (
            timestamp_factory
            if timestamp_factory is not None
            else self._utc_now
        )
        self.max_record_bytes = max_record_bytes
        self.max_log_bytes = max_log_bytes

    def record(self, envelope: EventEnvelope) -> None:
        try:
            record = self._build_record(envelope)
            encoded_line = self._encode_record(record)
        except AuditLogError:
            raise
        except Exception as error:
            raise AuditLogError("审计事件字段非法") from error
        with self._process_lock:
            self._append_line(envelope.session_id, encoded_line)

    def _build_record(self, envelope: EventEnvelope) -> dict:
        if not isinstance(envelope, EventEnvelope):
            raise AuditLogError("审计器只接受 EventEnvelope")
        self._validate_session_id(envelope.session_id)
        if (
            not isinstance(envelope.turn_id, str)
            or not envelope.turn_id.strip()
            or len(envelope.turn_id) > MAX_TURN_ID_LENGTH
        ):
            raise AuditLogError("turn_id 非法")
        if type(envelope.sequence) is not int or envelope.sequence < 1:
            raise AuditLogError("sequence 必须是正整数")
        if (
            isinstance(envelope.event, SessionSavedEvent)
            and envelope.event.session_id != envelope.session_id
        ):
            raise AuditLogError("保存事件与信封会话 ID 不一致")

        event_type, data = self._serialize_event(envelope.event)
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "recorded_at": self._recorded_at(),
            "session_id": envelope.session_id,
            "turn_id": envelope.turn_id,
            "sequence": envelope.sequence,
            "event_type": event_type,
            "data": data,
        }

    @staticmethod
    def _serialize_event(event) -> tuple[str, dict]:
        if isinstance(event, TokenEvent):
            return "TokenEvent", {
                "character_count": len(event.text),
            }
        if isinstance(event, ToolCallEvent):
            return "ToolCallEvent", {
                "tool_call_id": event.tool_call_id,
                "step": event.step,
                "name": event.name,
                "argument_count": len(event.args),
            }
        if isinstance(event, ToolResultEvent):
            return "ToolResultEvent", {
                "tool_call_id": event.tool_call_id,
                "name": event.name,
                "status": event.status,
                "character_count": event.character_count,
                "truncated": event.truncated,
                "duration_ms": event.duration_ms,
                "error_type": event.error_type,
            }
        if isinstance(event, ApprovalRequiredEvent):
            return "ApprovalRequiredEvent", {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "argument_count": len(event.args),
                "has_preview": bool(event.preview),
                "preview_character_count": len(event.preview),
            }
        if isinstance(event, ApprovalResolvedEvent):
            if event.outcome not in {
                "approved",
                "rejected",
                "missing",
                "mismatched",
                "invalid",
            }:
                raise AuditLogError("审批结果不在允许范围内")
            return "ApprovalResolvedEvent", {
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "outcome": event.outcome,
            }
        if isinstance(event, SystemEvent):
            return "SystemEvent", {
                "character_count": len(event.message),
            }
        if isinstance(event, ContextTrimmedEvent):
            return "ContextTrimmedEvent", {
                "removed_message_count": event.removed_message_count,
                "remaining_message_count": event.remaining_message_count,
            }
        if isinstance(event, MemoryUpdatedEvent):
            return "MemoryUpdatedEvent", {
                "character_count": event.character_count,
            }
        if isinstance(event, CitationValidationEvent):
            if event.status not in {
                "valid",
                "missing",
                "unknown",
                "not_applicable",
                "error",
            }:
                raise AuditLogError("引用校验状态不在允许范围内")
            counts = (
                event.citation_count,
                event.valid_citation_count,
                event.unknown_citation_count,
                event.retrieved_chunk_count,
            )
            if any(
                type(value) is not int or value < 0
                for value in counts
            ):
                raise AuditLogError("引用校验计数必须是非负整数")
            return "CitationValidationEvent", {
                "status": event.status,
                "citation_count": event.citation_count,
                "valid_citation_count": event.valid_citation_count,
                "unknown_citation_count": event.unknown_citation_count,
                "retrieved_chunk_count": event.retrieved_chunk_count,
            }
        if isinstance(event, CitationPolicyEvent):
            if event.policy not in {"observe", "require_valid"}:
                raise AuditLogError("引用策略不在允许范围内")
            if event.action not in {
                "observed",
                "allowed",
                "blocked",
            }:
                raise AuditLogError("引用策略动作不在允许范围内")
            if event.validation_status not in {
                "valid",
                "missing",
                "unknown",
                "not_applicable",
                "error",
            }:
                raise AuditLogError("引用策略校验状态不在允许范围内")
            return "CitationPolicyEvent", {
                "policy": event.policy,
                "action": event.action,
                "validation_status": event.validation_status,
            }
        if isinstance(event, ModelCallMetricsEvent):
            return "ModelCallMetricsEvent", {
                "call_index": event.call_index,
                "status": event.status,
                "duration_ms": event.duration_ms,
                "first_chunk_ms": event.first_chunk_ms,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "total_tokens": event.total_tokens,
                "token_source": event.token_source,
                "error_type": event.error_type,
            }
        if isinstance(event, SessionSavedEvent):
            return "SessionSavedEvent", {
                "saved": True,
            }
        raise UnsupportedAuditEventError(
            "事件类型不在审计白名单中"
        )

    def _encode_record(self, record: dict) -> bytes:
        try:
            encoded_line = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise AuditLogError("审计记录无法编码为 JSON") from error
        if len(encoded_line) > self.max_record_bytes:
            raise AuditLogLimitError("单条审计记录超过大小限制")
        return encoded_line

    def _append_line(
        self,
        session_id: str,
        encoded_line: bytes,
    ) -> None:
        directory_fd = self._open_audit_directory()
        audit_fd = None
        try:
            filename = f"{session_id}{AUDIT_FILE_SUFFIX}"
            file_existed = self._validate_existing_file(
                directory_fd,
                filename,
            )
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                audit_fd = os.open(
                    filename,
                    flags,
                    0o600,
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise AuditLogError(
                    "无法安全打开审计日志文件"
                ) from error

            file_stat = os.fstat(audit_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise AuditLogError("审计日志路径不是常规文件")
            if file_stat.st_size > self.max_log_bytes:
                raise AuditLogLimitError("审计日志已超过总大小限制")
            if file_stat.st_size + len(encoded_line) > self.max_log_bytes:
                raise AuditLogLimitError("追加后将超过审计日志总大小限制")

            os.fchmod(audit_fd, 0o600)
            with os.fdopen(audit_fd, "ab") as audit_file:
                audit_fd = None
                audit_file.write(encoded_line)
                audit_file.flush()
                os.fsync(audit_file.fileno())
            if not file_existed:
                os.fsync(directory_fd)
        except AuditLogError:
            raise
        except OSError as error:
            raise AuditLogError("写入审计日志失败") from error
        finally:
            if audit_fd is not None:
                os.close(audit_fd)
            os.close(directory_fd)

    def _open_audit_directory(self) -> int:
        try:
            root_mode = self.root.lstat().st_mode
        except FileNotFoundError:
            try:
                self.root.mkdir(mode=0o700)
            except OSError as error:
                raise AuditLogError("无法创建审计日志目录") from error
        else:
            if stat.S_ISLNK(root_mode):
                raise AuditLogError("审计日志目录不能是符号链接")
            if not stat.S_ISDIR(root_mode):
                raise AuditLogError("审计日志根路径不是目录")

        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            directory_fd = os.open(self.root, flags)
        except OSError as error:
            raise AuditLogError("无法安全打开审计日志目录") from error
        try:
            directory_stat = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_stat.st_mode):
                raise AuditLogError("审计日志根路径不是目录")
            os.fchmod(directory_fd, 0o700)
        except Exception:
            os.close(directory_fd)
            raise
        return directory_fd

    @staticmethod
    def _validate_existing_file(
        directory_fd: int,
        filename: str,
    ) -> bool:
        try:
            file_stat = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        except OSError as error:
            raise AuditLogError(
                "无法检查审计日志文件"
            ) from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise AuditLogError("审计日志文件不能是符号链接")
        if not stat.S_ISREG(file_stat.st_mode):
            raise AuditLogError("审计日志路径不是常规文件")
        return True

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if (
            not isinstance(session_id, str)
            or session_store.SESSION_ID_PATTERN.fullmatch(session_id) is None
        ):
            raise AuditLogError("会话 ID 非法")

    def _recorded_at(self) -> str:
        try:
            value = self.timestamp_factory()
        except Exception as error:
            raise AuditLogError("无法生成 UTC 审计时间戳") from error

        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise AuditLogError("审计时间戳必须包含 UTC 时区")
            return (
                value.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError as error:
                raise AuditLogError("审计时间戳格式非法") from error
            if (
                parsed.tzinfo is None
                or parsed.utcoffset() != timezone.utc.utcoffset(None)
            ):
                raise AuditLogError("审计时间戳必须使用 UTC")
            return value
        raise AuditLogError("审计时间戳必须是 datetime 或字符串")

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

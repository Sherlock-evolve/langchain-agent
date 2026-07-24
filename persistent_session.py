from collections.abc import Callable, Generator
from copy import deepcopy
from threading import Lock
from uuid import uuid4

import session_store
from agent import WorkspaceAgent
from async_runtime import bridge_sync_generator
from contracts import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    EventEnvelope,
    SessionSavedEvent,
)

MAX_TURN_ID_LENGTH = 128


class PersistentSessionError(Exception):
    """持久化会话运行失败。"""


class PersistentSessionOpenError(PersistentSessionError):
    """持久化会话无法安全打开。"""


class PersistentSessionSaveError(PersistentSessionError):
    """已提交状态暂时无法保存。"""


class PersistentSession:
    """组合 WorkspaceAgent 与本地快照仓库的单会话运行器。"""

    def __init__(
        self,
        session_id: str,
        agent: WorkspaceAgent,
        turn_id_factory: Callable[[], str] | None = None,
        store_backend=None,
    ):
        if turn_id_factory is not None and not callable(turn_id_factory):
            raise TypeError("turn_id_factory 必须可调用")

        self.session_id = session_id
        self.agent = agent
        self.store_backend = (
            store_backend
            if store_backend is not None
            else session_store
        )
        for method_name in (
            "save",
            "save_pending",
            "load",
            "load_pending",
            "delete_pending",
        ):
            if not callable(
                getattr(self.store_backend, method_name, None)
            ):
                raise TypeError(
                    f"store_backend 缺少 {method_name}()"
                )
        self.turn_id_factory = (
            turn_id_factory
            if turn_id_factory is not None
            else self._default_turn_id
        )
        self._dirty = False
        self._pending_snapshot = None
        self._operation_lock = Lock()

    @classmethod
    def open(
        cls,
        session_id: str,
        agent_factory: Callable[[], WorkspaceAgent],
        turn_id_factory: Callable[[], str] | None = None,
        store_backend=None,
    ) -> "PersistentSession":
        backend = (
            store_backend
            if store_backend is not None
            else session_store
        )
        try:
            snapshot = backend.load(session_id)
        except session_store.SessionNotFoundError:
            snapshot = None
        except session_store.SessionStoreError as error:
            raise PersistentSessionOpenError(
                f"无法加载会话 {session_id}：{error}"
            ) from error
        try:
            pending_approval = backend.load_pending(session_id)
        except session_store.SessionStoreError as error:
            raise PersistentSessionOpenError(
                f"无法加载会话 {session_id} 的待审批轮次：{error}"
            ) from error

        agent = agent_factory()
        if not isinstance(agent, WorkspaceAgent):
            raise TypeError("agent_factory 必须返回 WorkspaceAgent")

        if snapshot is not None:
            try:
                agent.restore_snapshot(snapshot)
            except Exception as error:
                raise PersistentSessionOpenError(
                    f"会话 {session_id} 的快照语义无效：{error}"
                ) from error
        if pending_approval is not None:
            try:
                agent.restore_pending_approval(pending_approval)
            except Exception as error:
                raise PersistentSessionOpenError(
                    f"会话 {session_id} 的待审批轮次语义无效：{error}"
                ) from error

        return cls(
            session_id=session_id,
            agent=agent,
            turn_id_factory=turn_id_factory,
            store_backend=backend,
        )

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def has_pending_approval(self) -> bool:
        return self.agent.has_pending_approval

    def pending_approval_event(self) -> ApprovalRequiredEvent | None:
        return self.agent.pending_approval_event()

    def stream_turn(
        self,
        question: str,
        *,
        request_idempotency_key: str | None = None,
    ) -> Generator[EventEnvelope, ApprovalDecision | None, None]:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("持久化会话已有正在进行的操作")

        try:
            if self._dirty:
                raise PersistentSessionSaveError(
                    "会话存在尚未保存的状态，请先调用 flush()"
                )
            if self.has_pending_approval:
                raise PersistentSessionError(
                    "会话存在待恢复审批，请先恢复或拒绝该审批"
                )
            turn_id = self._create_turn_id()
            events = self._stream_and_persist(
                question,
                request_idempotency_key=request_idempotency_key,
            )
            yield from self._envelope_events(events, turn_id)
        finally:
            self._operation_lock.release()

    async def astream_turn(
        self,
        question: str,
        *,
        cancellation_reason: str = "client_disconnect",
        request_idempotency_key: str | None = None,
    ):
        """Asynchronously stream a persisted turn with approval ``asend``."""

        def stream_factory():
            self.agent._prefer_async_execution = True
            try:
                yield from self.stream_turn(
                    question,
                    request_idempotency_key=request_idempotency_key,
                )
            finally:
                self.agent._prefer_async_execution = False

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
                    envelope = await stream.asend(decision)
                except StopAsyncIteration:
                    return
                decision = yield envelope
        finally:
            await stream.aclose()

    def stream_resume_pending_approval(
        self,
    ) -> Generator[EventEnvelope, ApprovalDecision | None, None]:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("持久化会话已有正在进行的操作")

        try:
            if self._dirty:
                raise PersistentSessionSaveError(
                    "会话存在尚未保存的状态，请先调用 flush()"
                )
            if not self.has_pending_approval:
                raise PersistentSessionError("当前没有可恢复的待审批轮次")
            turn_id = self._create_turn_id()
            events = self._resume_and_persist()
            yield from self._envelope_events(events, turn_id)
        finally:
            self._operation_lock.release()

    async def astream_resume_pending_approval(
        self,
        *,
        cancellation_reason: str = "client_disconnect",
    ):
        """Asynchronously resume a safely persisted approval transaction."""

        def stream_factory():
            self.agent._prefer_async_execution = True
            try:
                yield from self.stream_resume_pending_approval()
            finally:
                self.agent._prefer_async_execution = False

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
                    envelope = await stream.asend(decision)
                except StopAsyncIteration:
                    return
                decision = yield envelope
        finally:
            await stream.aclose()

    def cancel_active_turn(self, reason: str = "user") -> bool:
        return self.agent.cancel_active_turn(reason)

    @staticmethod
    def _default_turn_id() -> str:
        return str(uuid4())

    def _create_turn_id(self) -> str:
        turn_id = self.turn_id_factory()
        if not isinstance(turn_id, str):
            raise ValueError("turn_id_factory 必须返回字符串")
        if not turn_id.strip():
            raise ValueError("turn_id 不能为空")
        if len(turn_id) > MAX_TURN_ID_LENGTH:
            raise ValueError(
                f"turn_id 不能超过 {MAX_TURN_ID_LENGTH} 个字符"
            )
        return turn_id

    def _envelope_events(
        self,
        events: Generator[AgentEvent, ApprovalDecision | None, None],
        turn_id: str,
    ) -> Generator[EventEnvelope, ApprovalDecision | None, None]:
        sequence = 0
        decision = None
        completed = False

        try:
            while True:
                try:
                    event = events.send(decision)
                except StopIteration:
                    completed = True
                    return

                sequence += 1
                decision = yield EventEnvelope(
                    session_id=self.session_id,
                    turn_id=turn_id,
                    sequence=sequence,
                    event=event,
                )
        finally:
            if not completed:
                events.close()

    def _stream_and_persist(
        self,
        question: str,
        *,
        request_idempotency_key: str | None = None,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        agent_stream = self.agent.stream_turn(
            question,
            request_idempotency_key=request_idempotency_key,
        )
        yield from self._persist_agent_stream(agent_stream)

    def _resume_and_persist(
        self,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        agent_stream = self.agent.stream_resume_pending_approval()
        yield from self._persist_agent_stream(agent_stream)

    def _persist_agent_stream(
        self,
        agent_stream: Generator[
            AgentEvent,
            ApprovalDecision | None,
            None,
        ],
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        before_snapshot = self.agent.export_snapshot()
        stream_completed = False
        decision = None

        try:
            while True:
                try:
                    event = agent_stream.send(decision)
                except StopIteration:
                    stream_completed = True
                    break

                if isinstance(event, ApprovalRequiredEvent):
                    pending_record = self.agent.export_pending_approval()
                    if pending_record is not None:
                        try:
                            self.store_backend.save_pending(
                                self.session_id,
                                pending_record,
                            )
                        except Exception as error:
                            raise PersistentSessionSaveError(
                                "待审批轮次无法安全保存"
                            ) from error
                elif isinstance(event, ApprovalResolvedEvent):
                    try:
                        self.store_backend.delete_pending(
                            self.session_id,
                            missing_ok=True,
                        )
                    except Exception as error:
                        raise PersistentSessionSaveError(
                            "待审批记录无法在工具执行前清除"
                        ) from error

                decision = yield event
        finally:
            if not stream_completed:
                agent_stream.close()
                current_snapshot = self.agent.export_snapshot()
                if current_snapshot != before_snapshot:
                    self._mark_dirty(current_snapshot)

        after_snapshot = self.agent.export_snapshot()
        if after_snapshot == before_snapshot:
            return

        try:
            self.store_backend.save(self.session_id, after_snapshot)
        except Exception as error:
            self._mark_dirty(after_snapshot)
            raise PersistentSessionSaveError(
                f"会话状态已提交但保存失败：{error}"
            ) from error

        yield SessionSavedEvent(session_id=self.session_id)

    def flush(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("持久化会话已有正在进行的操作")

        try:
            if not self._dirty:
                return
            pending_snapshot = deepcopy(self._pending_snapshot)
            try:
                self.store_backend.save(
                    self.session_id,
                    pending_snapshot,
                )
            except Exception as error:
                raise PersistentSessionSaveError(
                    f"重试保存会话失败：{error}"
                ) from error

            self._dirty = False
            self._pending_snapshot = None
        finally:
            self._operation_lock.release()

    def _mark_dirty(self, snapshot: dict) -> None:
        self._dirty = True
        self._pending_snapshot = deepcopy(snapshot)

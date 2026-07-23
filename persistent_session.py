from collections.abc import Callable, Generator
from copy import deepcopy
from threading import Lock

import session_store
from agent import WorkspaceAgent
from contracts import (
    AgentEvent,
    ApprovalDecision,
    SessionSavedEvent,
)


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
    ):
        self.session_id = session_id
        self.agent = agent
        self._dirty = False
        self._pending_snapshot = None
        self._operation_lock = Lock()

    @classmethod
    def open(
        cls,
        session_id: str,
        agent_factory: Callable[[], WorkspaceAgent],
    ) -> "PersistentSession":
        try:
            snapshot = session_store.load(session_id)
        except session_store.SessionNotFoundError:
            snapshot = None
        except session_store.SessionStoreError as error:
            raise PersistentSessionOpenError(
                f"无法加载会话 {session_id}：{error}"
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

        return cls(
            session_id=session_id,
            agent=agent,
        )

    @property
    def dirty(self) -> bool:
        return self._dirty

    def stream_turn(
        self,
        question: str,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("持久化会话已有正在进行的操作")

        try:
            if self._dirty:
                raise PersistentSessionSaveError(
                    "会话存在尚未保存的状态，请先调用 flush()"
                )
            yield from self._stream_and_persist(question)
        finally:
            self._operation_lock.release()

    def _stream_and_persist(
        self,
        question: str,
    ) -> Generator[AgentEvent, ApprovalDecision | None, None]:
        before_snapshot = self.agent.export_snapshot()
        agent_stream = self.agent.stream_turn(question)
        stream_completed = False
        decision = None

        try:
            while True:
                try:
                    event = agent_stream.send(decision)
                except StopIteration:
                    stream_completed = True
                    break

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
            session_store.save(self.session_id, after_snapshot)
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
                session_store.save(
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

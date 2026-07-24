"""Authenticated FastAPI + SSE service for the workspace agent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

import session_store
from agent import WorkspaceAgent
from async_runtime import run_sync_asynchronously
from audit_log import JsonlAuditLogger
from contracts import (
    ApprovalDecision,
    ApprovalRequiredEvent,
    EventEnvelope,
    ModelCallMetricsEvent,
)
from persistent_session import PersistentSession
from knowledge_runtime import create_knowledge_runtime
from production_runtime import (
    ApiKeyAuthenticator,
    ApprovalBroker,
    AuthenticationError,
    ConcurrencyLimiter,
    FaultInjector,
    IdempotencyConflictError,
    MetricsRegistry,
    Principal,
    RequestIdempotencyStore,
    RequestLimitError,
    ServiceLimits,
)
from storage_security import SnapshotCipher, TenantPaths
from tool_execution import (
    ToolExecutionMiddleware,
    ToolExecutionPolicy,
)
from tools import create_workspace_tool_bundle


class ClientDisconnected(RuntimeError):
    """The SSE consumer disconnected before the turn completed."""


class RequestSizeLimitMiddleware:
    """Reject both fixed-length and chunked oversized request bodies."""

    def __init__(self, app, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.lower(): value
            for key, value in scope.get("headers", ())
        }
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await self._reject(send, 400, "invalid Content-Length")
                return
            if content_length > self.max_bytes:
                await self._reject(
                    send,
                    413,
                    "request size limit exceeded",
                )
                return

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > self.max_bytes:
                    raise RequestLimitError(
                        "request size limit exceeded"
                    )
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestLimitError as error:
            await self._reject(send, 413, str(error))

    @staticmethod
    async def _reject(send, status_code: int, detail: str) -> None:
        body = json.dumps(
            {"detail": detail},
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (
                        b"content-length",
                        str(len(body)).encode("ascii"),
                    ),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    message: str


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approved: bool


@dataclass(frozen=True)
class WebServiceConfig:
    storage_root: Path
    api_keys: dict[str, str]
    encryption_key: str
    limits: ServiceLimits = ServiceLimits()
    disconnect_poll_seconds: float = 0.05
    enable_knowledge: bool = False
    knowledge_directory: str = "docs"

    def __post_init__(self) -> None:
        root = Path(self.storage_root)
        if root.is_symlink():
            raise ValueError("storage_root cannot be a symbolic link")
        if (
            not isinstance(self.disconnect_poll_seconds, (int, float))
            or isinstance(self.disconnect_poll_seconds, bool)
            or self.disconnect_poll_seconds <= 0
        ):
            raise ValueError(
                "disconnect_poll_seconds must be positive"
            )
        if type(self.enable_knowledge) is not bool:
            raise ValueError("enable_knowledge must be a boolean")
        if (
            not isinstance(self.knowledge_directory, str)
            or not self.knowledge_directory
        ):
            raise ValueError(
                "knowledge_directory must be a non-empty string"
            )
        object.__setattr__(self, "storage_root", root.resolve())

    @classmethod
    def from_environment(cls) -> "WebServiceConfig":
        storage_root = os.getenv("AGENT_SERVICE_ROOT")
        api_keys = os.getenv("AGENT_API_KEYS")
        encryption_key = os.getenv("AGENT_ENCRYPTION_KEY")
        if not storage_root:
            raise ValueError("AGENT_SERVICE_ROOT is required")
        if not api_keys:
            raise ValueError("AGENT_API_KEYS is required")
        if not encryption_key:
            raise ValueError("AGENT_ENCRYPTION_KEY is required")
        try:
            parsed_keys = json.loads(api_keys)
        except json.JSONDecodeError:
            raise ValueError(
                "AGENT_API_KEYS must be a JSON object"
            ) from None
        def integer(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        def floating(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        limits = ServiceLimits(
            max_request_bytes=integer(
                "AGENT_MAX_REQUEST_BYTES",
                64 * 1024,
            ),
            max_message_characters=integer(
                "AGENT_MAX_MESSAGE_CHARACTERS",
                20_000,
            ),
            max_input_tokens=integer(
                "AGENT_MAX_INPUT_TOKENS",
                8_000,
            ),
            max_output_tokens=integer(
                "AGENT_MAX_OUTPUT_TOKENS",
                2_000,
            ),
            max_concurrent_global=integer(
                "AGENT_MAX_CONCURRENT_GLOBAL",
                32,
            ),
            max_concurrent_per_user=integer(
                "AGENT_MAX_CONCURRENT_PER_USER",
                4,
            ),
            max_cost_usd_per_turn=floating(
                "AGENT_MAX_COST_USD_PER_TURN",
                1.0,
            ),
            input_cost_per_million_tokens=floating(
                "AGENT_INPUT_COST_PER_MILLION_TOKENS",
                0.0,
            ),
            output_cost_per_million_tokens=floating(
                "AGENT_OUTPUT_COST_PER_MILLION_TOKENS",
                0.0,
            ),
        )
        return cls(
            storage_root=Path(storage_root),
            api_keys=parsed_keys,
            encryption_key=encryption_key,
            limits=limits,
            enable_knowledge=(
                os.getenv("AGENT_ENABLE_KNOWLEDGE", "")
                .strip()
                .lower()
                in {"1", "true", "yes"}
            ),
            knowledge_directory=os.getenv(
                "AGENT_KNOWLEDGE_DIRECTORY",
                "docs",
            ),
        )


class TenantSessionPool:
    """Owns user-scoped session instances and per-session async locks."""

    def __init__(
        self,
        config: WebServiceConfig,
        cipher: SnapshotCipher,
        session_factory: Callable | None = None,
    ) -> None:
        self.config = config
        self.cipher = cipher
        self.session_factory = session_factory
        self._lock = asyncio.Lock()
        self._sessions: dict[tuple[str, str, str], PersistentSession] = {}
        self._session_locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    async def lock_for(self, scope: tuple[str, str, str]) -> asyncio.Lock:
        async with self._lock:
            return self._session_locks.setdefault(scope, asyncio.Lock())

    async def get(
        self,
        user_id: str,
        workspace_id: str,
        session_id: str,
    ) -> PersistentSession:
        scope = (user_id, workspace_id, session_id)
        async with self._lock:
            existing = self._sessions.get(scope)
            if existing is not None:
                return existing

        paths = TenantPaths(
            self.config.storage_root,
            user_id,
            workspace_id,
        )
        await run_sync_asynchronously(paths.prepare)
        if self.session_factory is not None:
            created = await run_sync_asynchronously(
                self.session_factory,
                paths,
                session_id,
            )
        else:
            created = await run_sync_asynchronously(
                self._default_session,
                paths,
                session_id,
            )
        if not isinstance(created, PersistentSession):
            raise TypeError(
                "session_factory must return PersistentSession"
            )
        async with self._lock:
            return self._sessions.setdefault(scope, created)

    async def discard(self, scope: tuple[str, str, str]) -> None:
        async with self._lock:
            self._sessions.pop(scope, None)
            self._session_locks.pop(scope, None)

    def _default_session(
        self,
        paths: TenantPaths,
        session_id: str,
    ) -> PersistentSession:
        bundle = create_workspace_tool_bundle(paths.workspace)
        registered_tools = list(bundle.tools)
        knowledge_runtime = None
        if self.config.enable_knowledge:
            knowledge_runtime = create_knowledge_runtime(
                workspace_root=paths.workspace,
                knowledge_directory=self.config.knowledge_directory,
            )
            registered_tools.append(knowledge_runtime.search_tool)
        model = ChatOpenAI(
            model=os.getenv("ZHIPU_MODEL"),
            api_key=os.getenv("ZHIPU_API_KEY"),
            base_url=os.getenv("ZHIPU_BASE_URL"),
            temperature=0,
            max_tokens=self.config.limits.max_output_tokens,
        )
        policies = {
            registered_tool.name: ToolExecutionPolicy(
                risk="read_only",
                timeout_seconds=30.0,
                abandon_on_cancel=True,
                max_attempts=2,
                initial_backoff_seconds=0.1,
                max_backoff_seconds=1.0,
                total_budget_seconds=45.0,
            )
            for registered_tool in registered_tools
        }
        policies["write_file"] = ToolExecutionPolicy(
            risk="workspace_write",
            timeout_seconds=None,
            abandon_on_cancel=False,
        )

        def agent_factory() -> WorkspaceAgent:
            return WorkspaceAgent(
                model=model,
                tools=registered_tools,
                approval_required_tools={"write_file"},
                approval_preparers=bundle.approval_preparers,
                citation_validator=(
                    knowledge_runtime.citation_validator
                    if knowledge_runtime is not None
                    else None
                ),
                citation_guard_tool_names=(
                    set(knowledge_runtime.citation_guard_tool_names)
                    if knowledge_runtime is not None
                    else set()
                ),
                tool_execution_middleware=ToolExecutionMiddleware(
                    policies,
                    require_registered_policies=True,
                ),
            )

        backend = session_store.SessionStoreBackend(
            paths.sessions,
            cipher=self.cipher,
            aad_namespace=f"{paths.user_id}/{paths.workspace_id}",
        )
        return PersistentSession.open(
            session_id,
            agent_factory,
            store_backend=backend,
        )


def _json_safe(value):
    if is_dataclass(value):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def envelope_payload(envelope: EventEnvelope) -> dict:
    return {
        "session_id": envelope.session_id,
        "turn_id": envelope.turn_id,
        "sequence": envelope.sequence,
        "event": {
            "type": type(envelope.event).__name__,
            **_json_safe(envelope.event),
        },
    }


def encode_sse(envelope: EventEnvelope) -> bytes:
    event_name = type(envelope.event).__name__
    event_id = f"{envelope.turn_id}:{envelope.sequence}"
    data = json.dumps(
        envelope_payload(envelope),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        f"id: {event_id}\n"
        f"event: {event_name}\n"
        f"data: {data}\n\n"
    ).encode("utf-8")


def encode_sse_error(error: BaseException) -> bytes:
    detail = (
        str(error)
        if isinstance(
            error,
            (
                RequestLimitError,
                ClientDisconnected,
                IdempotencyConflictError,
            ),
        )
        else "turn failed safely"
    )
    data = json.dumps(
        {
            "error": type(error).__name__,
            "detail": detail,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: error\ndata: {data}\n\n".encode("utf-8")


async def _await_with_disconnect(
    awaitable,
    *,
    request: Request,
    session: PersistentSession,
    poll_seconds: float,
):
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait(
                {task},
                timeout=poll_seconds,
            )
            if task in done:
                return task.result()
            if await request.is_disconnected():
                session.cancel_active_turn("client_disconnect")
                task.cancel()
                raise ClientDisconnected(
                    "SSE client disconnected"
                )
    except asyncio.CancelledError:
        session.cancel_active_turn("client_disconnect")
        task.cancel()
        raise


def create_app(
    config: WebServiceConfig,
    *,
    session_factory: Callable | None = None,
    fault_injector: FaultInjector | None = None,
) -> FastAPI:
    """Build a production-scoped ASGI application."""

    if not isinstance(config, WebServiceConfig):
        raise TypeError("config must be WebServiceConfig")
    authenticator = ApiKeyAuthenticator(config.api_keys)
    cipher = SnapshotCipher.from_base64(config.encryption_key)
    pool = TenantSessionPool(
        config,
        cipher,
        session_factory=session_factory,
    )
    limiter = ConcurrencyLimiter(config.limits)
    idempotency = RequestIdempotencyStore()
    approvals = ApprovalBroker()
    metrics = MetricsRegistry()
    faults = fault_injector or FaultInjector()

    app = FastAPI(
        title="Workspace Agent API",
        version="0.2.0",
    )
    app.state.session_pool = pool
    app.state.metrics = metrics
    app.state.approvals = approvals
    app.state.idempotency = idempotency
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=config.limits.max_request_bytes,
    )

    async def principal_dependency(
        authorization: str | None = Header(default=None),
    ) -> Principal:
        try:
            return authenticator.authenticate(authorization)
        except AuthenticationError as error:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(error),
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

    def validate_scope(
        principal: Principal,
        workspace_id: str,
        session_id: str,
    ) -> tuple[str, str, str]:
        try:
            TenantPaths(
                config.storage_root,
                principal.user_id,
                workspace_id,
            )
            session_store._validate_session_id(session_id)
        except (ValueError, session_store.SessionStoreError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        return principal.user_id, workspace_id, session_id

    async def turn_response(
        *,
        request: Request,
        principal: Principal,
        workspace_id: str,
        session_id: str,
        idempotency_key: str | None,
        message: str | None,
        resume: bool,
    ) -> Response:
        scope = validate_scope(
            principal,
            workspace_id,
            session_id,
        )
        try:
            key = RequestIdempotencyStore.validate_key(
                idempotency_key
            )
            if not resume:
                config.limits.validate_message(message)
        except RequestLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        request_digest = hashlib.sha256(
            json.dumps(
                {
                    "resume": resume,
                    "message": message,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        try:
            mode, entry = await idempotency.begin(
                scope,
                key,
                request_digest,
            )
        except IdempotencyConflictError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error

        async def replay():
            chunks = (
                entry.response_chunks
                if mode == "replay"
                else await asyncio.shield(entry.future)
            )
            for chunk in chunks:
                yield chunk

        if mode != "new":
            return StreamingResponse(
                replay(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Idempotent-Replay": "true",
                },
            )

        async def generate():
            chunks: list[bytes] = []
            started_at = time.monotonic()
            turn_failed = False
            turn_cancelled = False
            cost_usd = 0.0
            registered_approval_id = None
            stream = None
            session = None
            try:
                async with limiter.slot(principal.user_id):
                    session_lock = await pool.lock_for(scope)
                    async with session_lock:
                        session = await pool.get(*scope)
                        faults.trigger("before_turn")
                        stream = (
                            session.astream_resume_pending_approval()
                            if resume
                            else session.astream_turn(
                                message,
                                request_idempotency_key=key,
                            )
                        )
                        logger = JsonlAuditLogger(
                            root=TenantPaths(
                                config.storage_root,
                                principal.user_id,
                                workspace_id,
                            ).audit,
                            rotation_count=5,
                            retention_days=30,
                        )
                        decision = None
                        while True:
                            try:
                                envelope = await _await_with_disconnect(
                                    stream.asend(decision),
                                    request=request,
                                    session=session,
                                    poll_seconds=(
                                        config.disconnect_poll_seconds
                                    ),
                                )
                            except StopAsyncIteration:
                                break
                            decision = None
                            if not isinstance(envelope, EventEnvelope):
                                raise TypeError(
                                    "session emitted an invalid envelope"
                                )
                            metrics.record_envelope(envelope)
                            if isinstance(
                                envelope.event,
                                ModelCallMetricsEvent,
                            ):
                                cost_usd += config.limits.model_cost(
                                    envelope.event
                                )
                                if (
                                    cost_usd
                                    > config.limits.max_cost_usd_per_turn
                                ):
                                    session.cancel_active_turn("shutdown")
                                    raise RequestLimitError(
                                        "turn cost limit exceeded"
                                    )
                            await run_sync_asynchronously(
                                logger.record,
                                envelope,
                            )
                            chunk = encode_sse(envelope)
                            chunks.append(chunk)
                            yield chunk

                            if isinstance(
                                envelope.event,
                                ApprovalRequiredEvent,
                            ):
                                registered_approval_id = (
                                    envelope.event.tool_call_id
                                )
                                approval_future = (
                                    await approvals.register(
                                        scope,
                                        registered_approval_id,
                                    )
                                )
                                try:
                                    approved = (
                                        await _await_with_disconnect(
                                            approval_future,
                                            request=request,
                                            session=session,
                                            poll_seconds=(
                                                config
                                                .disconnect_poll_seconds
                                            ),
                                        )
                                    )
                                finally:
                                    await approvals.unregister(
                                        scope,
                                        registered_approval_id,
                                    )
                                    registered_approval_id = None
                                decision = ApprovalDecision(
                                    tool_call_id=(
                                        envelope.event.tool_call_id
                                    ),
                                    approved=approved,
                                )
                        faults.trigger("after_turn")
                await idempotency.complete(entry, chunks)
            except ClientDisconnected as error:
                turn_cancelled = True
                await idempotency.fail(
                    scope,
                    key,
                    entry,
                    error,
                )
                return
            except asyncio.CancelledError as error:
                turn_cancelled = True
                if session is not None:
                    session.cancel_active_turn("client_disconnect")
                await idempotency.fail(
                    scope,
                    key,
                    entry,
                    error,
                )
                raise
            except BaseException as error:
                turn_failed = True
                error_chunk = encode_sse_error(error)
                chunks.append(error_chunk)
                yield error_chunk
                await idempotency.complete(entry, chunks)
            finally:
                if registered_approval_id is not None:
                    await approvals.unregister(
                        scope,
                        registered_approval_id,
                    )
                if stream is not None:
                    await stream.aclose()
                metrics.record_turn(
                    duration_ms=int(
                        max(0.0, time.monotonic() - started_at)
                        * 1000
                    ),
                    failed=turn_failed,
                    cancelled=turn_cancelled,
                    cost_usd=cost_usd,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics(
        principal: Principal = Depends(principal_dependency),
    ):
        return metrics.render_prometheus()

    @app.post(
        "/v1/workspaces/{workspace_id}/sessions/{session_id}/turns"
    )
    async def create_turn(
        workspace_id: str,
        session_id: str,
        payload: TurnRequest,
        request: Request,
        principal: Principal = Depends(principal_dependency),
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
        ),
    ):
        return await turn_response(
            request=request,
            principal=principal,
            workspace_id=workspace_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            message=payload.message,
            resume=False,
        )

    @app.post(
        "/v1/workspaces/{workspace_id}/sessions/{session_id}/resume"
    )
    async def resume_turn(
        workspace_id: str,
        session_id: str,
        request: Request,
        principal: Principal = Depends(principal_dependency),
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
        ),
    ):
        return await turn_response(
            request=request,
            principal=principal,
            workspace_id=workspace_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            message=None,
            resume=True,
        )

    @app.post(
        "/v1/workspaces/{workspace_id}/sessions/{session_id}"
        "/approvals/{tool_call_id}"
    )
    async def submit_approval(
        workspace_id: str,
        session_id: str,
        tool_call_id: str,
        payload: ApprovalRequest,
        principal: Principal = Depends(principal_dependency),
    ):
        scope = validate_scope(
            principal,
            workspace_id,
            session_id,
        )
        resolved = await approvals.resolve(
            scope,
            tool_call_id,
            payload.approved,
        )
        if not resolved:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="matching pending approval was not found",
            )
        return {
            "tool_call_id": tool_call_id,
            "accepted": True,
        }

    @app.get(
        "/v1/workspaces/{workspace_id}/sessions"
    )
    async def list_owned_sessions(
        workspace_id: str,
        principal: Principal = Depends(principal_dependency),
    ):
        paths = TenantPaths(
            config.storage_root,
            principal.user_id,
            workspace_id,
        )
        backend = session_store.SessionStoreBackend(
            paths.sessions,
            cipher=cipher,
            aad_namespace=f"{principal.user_id}/{workspace_id}",
        )
        return {
            "sessions": await run_sync_asynchronously(
                backend.list_sessions
            )
        }

    @app.delete(
        "/v1/workspaces/{workspace_id}/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_owned_session(
        workspace_id: str,
        session_id: str,
        principal: Principal = Depends(principal_dependency),
    ):
        scope = validate_scope(
            principal,
            workspace_id,
            session_id,
        )
        lock = await pool.lock_for(scope)
        if lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="session has an active transaction",
            )
        async with lock:
            paths = TenantPaths(
                config.storage_root,
                principal.user_id,
                workspace_id,
            )
            backend = session_store.SessionStoreBackend(
                paths.sessions,
                cipher=cipher,
                aad_namespace=f"{principal.user_id}/{workspace_id}",
            )
            try:
                await run_sync_asynchronously(
                    backend.delete,
                    session_id,
                )
            except session_store.SessionNotFoundError as error:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(error),
                ) from error
            audit_logger = JsonlAuditLogger(
                root=paths.audit,
                rotation_count=5,
                retention_days=30,
            )
            await run_sync_asynchronously(
                audit_logger.delete_session_logs,
                session_id,
            )
            await pool.discard(scope)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Workspace Agent API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    app = create_app(WebServiceConfig.from_environment())
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        workers=1,
    )


if __name__ == "__main__":
    main()

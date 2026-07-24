"""Production boundaries shared by the FastAPI service and tests."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import time
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock

from contracts import (
    CitationPolicyEvent,
    CitationValidationEvent,
    EventEnvelope,
    ModelCallMetricsEvent,
    ToolResultEvent,
)


IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class AuthenticationError(RuntimeError):
    """An API credential was missing or invalid."""


class RequestLimitError(RuntimeError):
    """A configured request, token, cost or concurrency limit was exceeded."""


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for a different request."""


@dataclass(frozen=True)
class Principal:
    user_id: str


class ApiKeyAuthenticator:
    """Constant-time API key authentication without retaining raw keys."""

    def __init__(self, api_keys: dict[str, str]) -> None:
        if not isinstance(api_keys, dict) or not api_keys:
            raise ValueError("at least one API key must be configured")
        hashed = {}
        for api_key, user_id in api_keys.items():
            if (
                not isinstance(api_key, str)
                or len(api_key) < 16
                or not isinstance(user_id, str)
                or not user_id
            ):
                raise ValueError("API key mapping is invalid")
            digest = hashlib.sha256(api_key.encode("utf-8")).digest()
            if digest in hashed:
                raise ValueError("duplicate API key")
            hashed[digest] = user_id
        self._hashed_keys = hashed

    @classmethod
    def from_json(cls, value: str) -> "ApiKeyAuthenticator":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            raise ValueError("AGENT_API_KEYS must be a JSON object") from None
        return cls(payload)

    def authenticate(self, authorization: str | None) -> Principal:
        if (
            not isinstance(authorization, str)
            or not authorization.startswith("Bearer ")
        ):
            raise AuthenticationError("Bearer authentication is required")
        api_key = authorization[7:]
        if not api_key:
            raise AuthenticationError("Bearer authentication is required")
        candidate = hashlib.sha256(api_key.encode("utf-8")).digest()
        matched_user = None
        for expected, user_id in self._hashed_keys.items():
            if hmac.compare_digest(candidate, expected):
                matched_user = user_id
        if matched_user is None:
            raise AuthenticationError("API key is invalid")
        return Principal(user_id=matched_user)


@dataclass(frozen=True)
class ServiceLimits:
    max_request_bytes: int = 64 * 1024
    max_message_characters: int = 20_000
    max_input_tokens: int = 8_000
    max_output_tokens: int = 2_000
    max_concurrent_global: int = 32
    max_concurrent_per_user: int = 4
    max_cost_usd_per_turn: float = 1.0
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_request_bytes",
            "max_message_characters",
            "max_input_tokens",
            "max_output_tokens",
            "max_concurrent_global",
            "max_concurrent_per_user",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_cost_usd_per_turn",
            "input_cost_per_million_tokens",
            "output_cost_per_million_tokens",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative and finite")

    def validate_message(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise RequestLimitError("message must be non-empty")
        if len(message) > self.max_message_characters:
            raise RequestLimitError("message character limit exceeded")
        estimated_tokens = max(1, math.ceil(len(message) / 4))
        if estimated_tokens > self.max_input_tokens:
            raise RequestLimitError("input token limit exceeded")

    def model_cost(
        self,
        event: ModelCallMetricsEvent,
    ) -> float:
        if event.token_source != "provider":
            return 0.0
        input_tokens = event.input_tokens or 0
        output_tokens = event.output_tokens or 0
        return (
            input_tokens * self.input_cost_per_million_tokens
            + output_tokens * self.output_cost_per_million_tokens
        ) / 1_000_000


class ConcurrencyLimiter:
    """Fail-fast global and per-user concurrency limits."""

    def __init__(self, limits: ServiceLimits) -> None:
        self.limits = limits
        self._lock = asyncio.Lock()
        self._global_active = 0
        self._user_active: dict[str, int] = defaultdict(int)

    @asynccontextmanager
    async def slot(self, user_id: str):
        async with self._lock:
            if self._global_active >= self.limits.max_concurrent_global:
                raise RequestLimitError(
                    "global concurrency limit exceeded"
                )
            if (
                self._user_active[user_id]
                >= self.limits.max_concurrent_per_user
            ):
                raise RequestLimitError(
                    "user concurrency limit exceeded"
                )
            self._global_active += 1
            self._user_active[user_id] += 1
        try:
            yield
        finally:
            async with self._lock:
                self._global_active -= 1
                self._user_active[user_id] -= 1
                if self._user_active[user_id] == 0:
                    del self._user_active[user_id]


class MetricsRegistry:
    """Low-cardinality in-memory counters suitable for scraping."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._turn_latency_sum_ms = 0.0
        self._turn_latency_count = 0

    def record_envelope(self, envelope: EventEnvelope) -> None:
        event = envelope.event
        with self._lock:
            self._counters["agent_events_total"] += 1
            if isinstance(event, ModelCallMetricsEvent):
                self._counters["model_calls_total"] += 1
                self._counters["model_duration_ms_total"] += (
                    event.duration_ms
                )
                if event.status == "error":
                    self._counters["model_failures_total"] += 1
                if event.total_tokens is not None:
                    self._counters["model_tokens_total"] += (
                        event.total_tokens
                    )
            elif isinstance(event, ToolResultEvent):
                self._counters["tool_calls_total"] += 1
                if event.status == "error":
                    self._counters["tool_failures_total"] += 1
            elif isinstance(event, CitationValidationEvent):
                self._counters["citation_validations_total"] += 1
                if event.status not in {"valid", "not_applicable"}:
                    self._counters[
                        "citation_validation_failures_total"
                    ] += 1
            elif isinstance(event, CitationPolicyEvent):
                if event.action == "blocked":
                    self._counters["citation_blocks_total"] += 1

    def record_turn(
        self,
        *,
        duration_ms: int,
        failed: bool,
        cancelled: bool,
        cost_usd: float,
    ) -> None:
        with self._lock:
            self._counters["turns_total"] += 1
            if failed:
                self._counters["turn_failures_total"] += 1
            if cancelled:
                self._counters["turn_cancellations_total"] += 1
            self._counters["estimated_cost_usd_total"] += cost_usd
            self._turn_latency_sum_ms += max(0, duration_ms)
            self._turn_latency_count += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                f"workspace_agent_{name} {value}"
                for name, value in sorted(self._counters.items())
            ]
            lines.extend(
                [
                    "workspace_agent_turn_latency_ms_sum "
                    f"{self._turn_latency_sum_ms}",
                    "workspace_agent_turn_latency_ms_count "
                    f"{self._turn_latency_count}",
                ]
            )
        return "\n".join(lines) + "\n"


@dataclass
class _IdempotencyEntry:
    request_digest: str
    future: asyncio.Future
    response_chunks: tuple[bytes, ...] | None = None
    completed_at: float | None = None


class RequestIdempotencyStore:
    """Bounded request replay for completed and in-flight SSE turns."""

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        ttl_seconds: float = 24 * 60 * 60,
        monotonic_clock=None,
    ) -> None:
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(float(ttl_seconds))
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be positive and finite")
        self.max_entries = max_entries
        self.ttl_seconds = float(ttl_seconds)
        self.monotonic_clock = monotonic_clock or time.monotonic
        self._lock = asyncio.Lock()
        self._entries: OrderedDict[tuple, _IdempotencyEntry] = OrderedDict()

    @staticmethod
    def validate_key(key: str | None) -> str:
        if (
            not isinstance(key, str)
            or IDEMPOTENCY_KEY_PATTERN.fullmatch(key) is None
        ):
            raise RequestLimitError(
                "a valid Idempotency-Key header is required"
            )
        return key

    async def begin(
        self,
        scope: tuple,
        key: str,
        request_digest: str,
    ) -> tuple[str, _IdempotencyEntry]:
        async with self._lock:
            self._prune_locked()
            scoped_key = (*scope, key)
            existing = self._entries.get(scoped_key)
            if existing is not None:
                if not hmac.compare_digest(
                    existing.request_digest,
                    request_digest,
                ):
                    raise IdempotencyConflictError(
                        "idempotency key was reused with another request"
                    )
                self._entries.move_to_end(scoped_key)
                return (
                    "replay"
                    if existing.response_chunks is not None
                    else "wait",
                    existing,
                )
            while len(self._entries) >= self.max_entries:
                evictable_key = next(
                    (
                        candidate_key
                        for candidate_key, candidate in self._entries.items()
                        if candidate.future.done()
                    ),
                    None,
                )
                if evictable_key is None:
                    raise RequestLimitError(
                        "idempotency store capacity exceeded"
                    )
                self._entries.pop(evictable_key)
            future = asyncio.get_running_loop().create_future()
            entry = _IdempotencyEntry(
                request_digest=request_digest,
                future=future,
            )
            self._entries[scoped_key] = entry
            return "new", entry

    async def complete(
        self,
        entry: _IdempotencyEntry,
        chunks: list[bytes],
    ) -> None:
        immutable_chunks = tuple(chunks)
        entry.response_chunks = immutable_chunks
        entry.completed_at = self.monotonic_clock()
        if not entry.future.done():
            entry.future.set_result(immutable_chunks)

    async def fail(
        self,
        scope: tuple,
        key: str,
        entry: _IdempotencyEntry,
        error: BaseException,
    ) -> None:
        async with self._lock:
            self._entries.pop((*scope, key), None)
        if not entry.future.done():
            entry.future.set_exception(error)
            entry.future.add_done_callback(
                lambda completed: completed.exception()
            )

    def _prune_locked(self) -> None:
        now = self.monotonic_clock()
        expired = [
            key
            for key, entry in self._entries.items()
            if (
                entry.completed_at is not None
                and now - entry.completed_at > self.ttl_seconds
            )
        ]
        for key in expired:
            self._entries.pop(key, None)


class ApprovalBroker:
    """User/session-scoped rendezvous for independent approval requests."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pending: dict[tuple, asyncio.Future] = {}

    async def register(self, scope: tuple, tool_call_id: str) -> asyncio.Future:
        key = (*scope, tool_call_id)
        async with self._lock:
            if key in self._pending:
                raise RuntimeError("approval is already pending")
            future = asyncio.get_running_loop().create_future()
            self._pending[key] = future
            return future

    async def resolve(
        self,
        scope: tuple,
        tool_call_id: str,
        approved: bool,
    ) -> bool:
        key = (*scope, tool_call_id)
        async with self._lock:
            future = self._pending.get(key)
            if future is None or future.done():
                return False
            future.set_result(approved)
            return True

    async def unregister(
        self,
        scope: tuple,
        tool_call_id: str,
    ) -> None:
        key = (*scope, tool_call_id)
        async with self._lock:
            future = self._pending.pop(key, None)
            if future is not None and not future.done():
                future.cancel()


class FaultInjector:
    """Deterministic, opt-in failure hooks for resilience testing."""

    def __init__(self, failpoints: set[str] | None = None) -> None:
        self.failpoints = set(failpoints or ())

    @classmethod
    def from_environment(cls) -> "FaultInjector":
        raw = os.getenv("AGENT_FAILPOINTS", "")
        return cls(
            {
                item.strip()
                for item in raw.split(",")
                if item.strip()
            }
        )

    def trigger(self, name: str) -> None:
        if name in self.failpoints:
            raise RuntimeError(f"injected failure: {name}")

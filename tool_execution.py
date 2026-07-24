"""Policy-driven tool execution with cancellation, retry and circuit breaking.

The middleware owns the execution boundary only. Conversation budgets,
duplicate-call detection and human approval remain Agent concerns.
"""

from __future__ import annotations

import math
import queue
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Literal, Protocol, runtime_checkable


ToolRisk = Literal[
    "read_only",
    "workspace_write",
    "external_side_effect",
]
CANCELLATION_REASONS = frozenset(
    {"user", "client_disconnect", "shutdown"}
)
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ToolExecutionError(RuntimeError):
    """Base class for safe execution-control failures."""


class ToolExecutionTimeout(ToolExecutionError):
    """A cancellable tool exceeded its configured execution deadline."""


class ToolExecutionBudgetExceeded(ToolExecutionTimeout):
    """A tool exhausted its total execution and retry budget."""


class ToolExecutionCancelled(ToolExecutionError):
    """The caller cancelled the active turn."""


class ToolCircuitOpen(ToolExecutionError):
    """A failing tool is temporarily blocked by its circuit breaker."""


class ToolPolicyNotRegistered(ToolExecutionError):
    """A tool was used without an explicit risk policy."""


class ToolIdempotencyKeyRequired(ToolExecutionError):
    """A side effect or retry was attempted without an idempotency key."""


class CancellationToken:
    """A thread-safe, one-way cancellation signal with an optional deadline.

    Cooperative tools receive a child token. Calling ``raise_if_cancelled()``
    inside long-running loops observes both turn cancellation and the tool's
    execution deadline.
    """

    def __init__(
        self,
        *,
        parent: "CancellationToken | None" = None,
        deadline: float | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if parent is not None and not isinstance(parent, CancellationToken):
            raise TypeError("parent must be a CancellationToken")
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise ValueError("deadline must be a finite monotonic timestamp")
        if monotonic_clock is not None and not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._event = Event()
        self._reason = "user"
        self._lock = Lock()
        self._parent = parent
        self._deadline = (
            float(deadline) if deadline is not None else None
        )
        self._monotonic_clock = monotonic_clock or time.monotonic

    def cancel(self, reason: str = "user") -> bool:
        if reason not in CANCELLATION_REASONS:
            raise ValueError("cancellation reason is not supported")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason
            self._event.set()
            return True

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (
            self._parent is not None and self._parent.cancelled
        )

    @property
    def reason(self) -> str:
        if self._parent is not None and self._parent.cancelled:
            return self._parent.reason
        with self._lock:
            return self._reason

    @property
    def deadline(self) -> float | None:
        return self._deadline

    @property
    def remaining_seconds(self) -> float | None:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._monotonic_clock())

    def child(
        self,
        *,
        deadline: float | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> "CancellationToken":
        return CancellationToken(
            parent=self,
            deadline=deadline,
            monotonic_clock=monotonic_clock or self._monotonic_clock,
        )

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ToolExecutionCancelled("The active turn was cancelled.")
        if (
            self._deadline is not None
            and self._monotonic_clock() >= self._deadline
        ):
            raise ToolExecutionBudgetExceeded(
                "The tool exhausted its total execution budget."
            )


@runtime_checkable
class CooperativeCancellationTool(Protocol):
    """Injection protocol required by cooperative long-running tools."""

    name: str

    def invoke_with_cancellation(
        self,
        args: dict,
        cancellation_token: CancellationToken,
        *,
        idempotency_key: str | None = None,
    ) -> object:
        """Execute while periodically checking ``cancellation_token``."""


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """Immutable execution policy attached to one registered tool."""

    risk: ToolRisk = "read_only"
    timeout_seconds: float | None = 30.0
    abandon_on_cancel: bool = True
    cooperative_cancellation: bool = False
    max_attempts: int = 1
    initial_backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0
    total_budget_seconds: float | None = None
    retry_exception_types: tuple[type[BaseException], ...] = (Exception,)
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.risk not in {
            "read_only",
            "workspace_write",
            "external_side_effect",
        }:
            raise ValueError("invalid tool risk")
        self._validate_optional_seconds(
            "timeout_seconds",
            self.timeout_seconds,
        )
        self._validate_optional_seconds(
            "total_budget_seconds",
            self.total_budget_seconds,
        )
        for name, value, allow_zero in (
            ("initial_backoff_seconds", self.initial_backoff_seconds, True),
            ("backoff_multiplier", self.backoff_multiplier, False),
            ("max_backoff_seconds", self.max_backoff_seconds, True),
            ("circuit_recovery_seconds", self.circuit_recovery_seconds, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (value < 0 if allow_zero else value <= 0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} finite number")
        if type(self.abandon_on_cancel) is not bool:
            raise ValueError("abandon_on_cancel must be a boolean")
        if type(self.cooperative_cancellation) is not bool:
            raise ValueError("cooperative_cancellation must be a boolean")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if (
            type(self.circuit_failure_threshold) is not int
            or self.circuit_failure_threshold < 1
        ):
            raise ValueError(
                "circuit_failure_threshold must be a positive integer"
            )
        if (
            not isinstance(self.retry_exception_types, tuple)
            or not self.retry_exception_types
            or any(
                not isinstance(error_type, type)
                or not issubclass(error_type, BaseException)
                for error_type in self.retry_exception_types
            )
        ):
            raise ValueError(
                "retry_exception_types must contain exception classes"
            )
        if self.risk != "read_only" and self.abandon_on_cancel:
            raise ValueError(
                "side-effecting tools cannot be abandoned after execution starts"
            )
        if (
            self.risk != "read_only"
            and not self.cooperative_cancellation
            and self.timeout_seconds is not None
        ):
            raise ValueError(
                "non-cooperative side effects require timeout_seconds=None"
            )
        if (
            self.risk != "read_only"
            and not self.cooperative_cancellation
            and self.total_budget_seconds is not None
        ):
            raise ValueError(
                "non-cooperative side effects cannot enforce a total budget"
            )

    @staticmethod
    def _validate_optional_seconds(
        name: str,
        value: float | None,
    ) -> None:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


class ToolExecutionMiddleware:
    """Execute registered actions under immutable per-tool policies."""

    def __init__(
        self,
        policies: Mapping[str, ToolExecutionPolicy] | None = None,
        *,
        default_policy: ToolExecutionPolicy | None = None,
        require_registered_policies: bool = False,
        poll_interval_seconds: float = 0.02,
        monotonic_clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        configured_policies = dict(policies or {})
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(policy, ToolExecutionPolicy)
            for name, policy in configured_policies.items()
        ):
            raise ValueError(
                "tool execution policies require names and ToolExecutionPolicy values"
            )
        if default_policy is None:
            default_policy = ToolExecutionPolicy()
        if not isinstance(default_policy, ToolExecutionPolicy):
            raise TypeError("default_policy must be ToolExecutionPolicy")
        if type(require_registered_policies) is not bool:
            raise TypeError("require_registered_policies must be a boolean")
        if (
            isinstance(poll_interval_seconds, bool)
            or not isinstance(poll_interval_seconds, (int, float))
            or not math.isfinite(float(poll_interval_seconds))
            or poll_interval_seconds <= 0
        ):
            raise ValueError(
                "poll_interval_seconds must be a positive finite number"
            )
        if monotonic_clock is not None and not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        if sleeper is not None and not callable(sleeper):
            raise TypeError("sleeper must be callable")

        self._policies = configured_policies
        self.default_policy = default_policy
        self.require_registered_policies = require_registered_policies
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._circuit_lock = Lock()
        self._circuits: dict[str, _CircuitState] = {}

    def validate_registered_tools(self, tool_names) -> None:
        names = set(tool_names)
        missing = names - self._policies.keys()
        if self.require_registered_policies and missing:
            raise ToolPolicyNotRegistered(
                "missing execution policies for tools: "
                + ", ".join(sorted(missing))
            )

    def policy_for(self, tool_name: str) -> ToolExecutionPolicy:
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty string")
        policy = self._policies.get(tool_name)
        if policy is None:
            if self.require_registered_policies:
                raise ToolPolicyNotRegistered(
                    f"tool execution policy is not registered: {tool_name}"
                )
            return self.default_policy
        return policy

    def execute(
        self,
        tool_name: str,
        action: Callable[..., object],
        cancellation_token: CancellationToken,
        *,
        idempotency_key: str | None = None,
    ) -> object:
        """Run one action under retry, deadline and circuit-breaker controls.

        When ``cooperative_cancellation`` is enabled, ``action`` is invoked as
        ``action(cancellation_token)``. Otherwise it is invoked with no
        arguments. External side effects always require an idempotency key.
        Other side effects require one only when retries are enabled.
        """

        if not callable(action):
            raise TypeError("tool action must be callable")
        if not isinstance(cancellation_token, CancellationToken):
            raise TypeError("cancellation_token must be CancellationToken")

        policy = self.policy_for(tool_name)
        self._validate_idempotency(policy, idempotency_key)
        cancellation_token.raise_if_cancelled()
        self._before_circuit_call(tool_name, policy)

        started_at = self.monotonic_clock()
        total_deadline = (
            started_at + float(policy.total_budget_seconds)
            if policy.total_budget_seconds is not None
            else None
        )
        tool_token = cancellation_token.child(
            deadline=total_deadline,
            monotonic_clock=self.monotonic_clock,
        )

        try:
            for attempt in range(1, policy.max_attempts + 1):
                tool_token.raise_if_cancelled()
                try:
                    result = self._execute_attempt(
                        action,
                        tool_token,
                        policy,
                        total_deadline,
                    )
                except ToolExecutionCancelled:
                    raise
                except BaseException as error:
                    if not self._can_retry(
                        error,
                        attempt,
                        policy,
                        idempotency_key,
                    ):
                        raise
                    self._backoff(
                        attempt,
                        policy,
                        tool_token,
                        total_deadline,
                    )
                else:
                    self._record_circuit_success(tool_name)
                    return result
        except ToolExecutionCancelled:
            self._release_circuit_probe(tool_name)
            raise
        except BaseException:
            self._record_circuit_failure(tool_name, policy)
            raise

        raise ToolExecutionError("The tool exhausted its execution attempts.")

    @staticmethod
    def _validate_idempotency(
        policy: ToolExecutionPolicy,
        idempotency_key: str | None,
    ) -> None:
        key_required = (
            policy.risk == "external_side_effect"
            or (policy.risk != "read_only" and policy.max_attempts > 1)
        )
        if key_required and (
            not isinstance(idempotency_key, str)
            or IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key) is None
        ):
            raise ToolIdempotencyKeyRequired(
                "A valid idempotency key is required for this tool."
            )
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key) is None
        ):
            raise ValueError("idempotency_key is invalid")

    def _execute_attempt(
        self,
        action: Callable[..., object],
        cancellation_token: CancellationToken,
        policy: ToolExecutionPolicy,
        total_deadline: float | None,
    ) -> object:
        if policy.risk != "read_only" and not policy.cooperative_cancellation:
            return action()
        return self._execute_cancellable(
            action,
            cancellation_token,
            policy,
            total_deadline,
        )

    def _execute_cancellable(
        self,
        action: Callable[..., object],
        cancellation_token: CancellationToken,
        policy: ToolExecutionPolicy,
        total_deadline: float | None,
    ) -> object:
        outcomes: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def run_action() -> None:
            try:
                if policy.cooperative_cancellation:
                    value = action(cancellation_token)
                else:
                    value = action()
                outcomes.put((True, value))
            except BaseException as error:
                outcomes.put((False, error))

        worker = Thread(
            target=run_action,
            name="workspace-agent-tool",
            daemon=True,
        )
        worker.start()

        attempt_started_at = self.monotonic_clock()
        attempt_deadline = (
            attempt_started_at + float(policy.timeout_seconds)
            if policy.timeout_seconds is not None
            else None
        )
        deadline_candidates = [
            value
            for value in (attempt_deadline, total_deadline)
            if value is not None
        ]
        effective_deadline = (
            min(deadline_candidates) if deadline_candidates else None
        )

        while True:
            if cancellation_token.cancelled and policy.abandon_on_cancel:
                raise ToolExecutionCancelled("The active turn was cancelled.")

            wait_seconds = self.poll_interval_seconds
            if effective_deadline is not None:
                remaining = effective_deadline - self.monotonic_clock()
                if remaining <= 0:
                    if (
                        total_deadline is not None
                        and effective_deadline == total_deadline
                    ):
                        raise ToolExecutionBudgetExceeded(
                            "The tool exhausted its total execution budget."
                        )
                    raise ToolExecutionTimeout(
                        "The tool exceeded its execution deadline."
                    )
                wait_seconds = min(wait_seconds, remaining)

            try:
                succeeded, value = outcomes.get(timeout=wait_seconds)
            except queue.Empty:
                continue

            if succeeded:
                return value
            if isinstance(value, BaseException):
                raise value
            raise ToolExecutionError("The tool returned an invalid outcome.")

    @staticmethod
    def _can_retry(
        error: BaseException,
        attempt: int,
        policy: ToolExecutionPolicy,
        idempotency_key: str | None,
    ) -> bool:
        if attempt >= policy.max_attempts:
            return False
        if isinstance(error, (ToolExecutionCancelled, KeyboardInterrupt)):
            return False
        if not isinstance(error, policy.retry_exception_types):
            return False
        return (
            policy.risk == "read_only"
            or idempotency_key is not None
        )

    def _backoff(
        self,
        attempt: int,
        policy: ToolExecutionPolicy,
        cancellation_token: CancellationToken,
        total_deadline: float | None,
    ) -> None:
        delay = min(
            float(policy.max_backoff_seconds),
            float(policy.initial_backoff_seconds)
            * (float(policy.backoff_multiplier) ** (attempt - 1)),
        )
        if delay <= 0:
            return
        if total_deadline is not None:
            remaining = total_deadline - self.monotonic_clock()
            if remaining <= 0 or delay > remaining:
                raise ToolExecutionBudgetExceeded(
                    "Retry backoff exceeds the tool's total budget."
                )

        slept = 0.0
        while slept < delay:
            cancellation_token.raise_if_cancelled()
            interval = min(self.poll_interval_seconds, delay - slept)
            self.sleeper(interval)
            slept += interval
        cancellation_token.raise_if_cancelled()

    def _before_circuit_call(
        self,
        tool_name: str,
        policy: ToolExecutionPolicy,
    ) -> None:
        now = self.monotonic_clock()
        with self._circuit_lock:
            state = self._circuits.setdefault(tool_name, _CircuitState())
            if state.opened_at is None:
                return
            if now - state.opened_at < policy.circuit_recovery_seconds:
                raise ToolCircuitOpen(
                    f"tool circuit is open: {tool_name}"
                )
            if state.probe_in_flight:
                raise ToolCircuitOpen(
                    f"tool circuit recovery probe is active: {tool_name}"
                )
            state.probe_in_flight = True

    def _record_circuit_success(self, tool_name: str) -> None:
        with self._circuit_lock:
            self._circuits[tool_name] = _CircuitState()

    def _record_circuit_failure(
        self,
        tool_name: str,
        policy: ToolExecutionPolicy,
    ) -> None:
        with self._circuit_lock:
            state = self._circuits.setdefault(tool_name, _CircuitState())
            state.probe_in_flight = False
            state.consecutive_failures += 1
            if (
                state.consecutive_failures
                >= policy.circuit_failure_threshold
            ):
                state.opened_at = self.monotonic_clock()

    def _release_circuit_probe(self, tool_name: str) -> None:
        with self._circuit_lock:
            state = self._circuits.get(tool_name)
            if state is not None:
                state.probe_in_flight = False

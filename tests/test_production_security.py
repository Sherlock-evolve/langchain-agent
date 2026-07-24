import asyncio

import pytest

import session_store
from contracts import EventEnvelope, ModelCallMetricsEvent
from production_runtime import (
    ApiKeyAuthenticator,
    AuthenticationError,
    ConcurrencyLimiter,
    FaultInjector,
    MetricsRegistry,
    RequestIdempotencyStore,
    RequestLimitError,
    ServiceLimits,
)
from storage_security import SnapshotCipher, TenantPaths


def test_session_snapshots_and_pending_approvals_are_encrypted_at_rest(
    tmp_path,
):
    key = SnapshotCipher.generate_key()
    cipher = SnapshotCipher.from_base64(key)
    backend = session_store.SessionStoreBackend(
        tmp_path / "sessions",
        cipher=cipher,
    )
    snapshot = {"private": "committed secret"}
    pending = {"private": "pending secret"}

    backend.save("demo", snapshot)
    backend.save_pending("demo", pending)

    snapshot_bytes = (tmp_path / "sessions" / "demo.json").read_bytes()
    pending_bytes = (
        tmp_path / "sessions" / "demo.pending.json"
    ).read_bytes()
    assert b"committed secret" not in snapshot_bytes
    assert b"pending secret" not in pending_bytes
    assert backend.load("demo") == snapshot
    assert backend.load_pending("demo") == pending

    wrong_backend = session_store.SessionStoreBackend(
        tmp_path / "sessions",
        cipher=SnapshotCipher.from_base64(
            SnapshotCipher.generate_key()
        ),
    )
    with pytest.raises(session_store.CorruptSessionError):
        wrong_backend.load("demo")

    wrong_tenant = session_store.SessionStoreBackend(
        tmp_path / "sessions",
        cipher=cipher,
        aad_namespace="another-user/project",
    )
    with pytest.raises(session_store.CorruptSessionError):
        wrong_tenant.load("demo")


def test_tenant_paths_keep_users_and_workspaces_disjoint(tmp_path):
    alice = TenantPaths(tmp_path, "alice", "project")
    bob = TenantPaths(tmp_path, "bob", "project")
    other_workspace = TenantPaths(tmp_path, "alice", "other")
    alice.prepare()
    bob.prepare()
    other_workspace.prepare()

    assert alice.root != bob.root
    assert alice.root != other_workspace.root
    assert alice.workspace.is_dir()
    assert alice.sessions.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError):
        TenantPaths(tmp_path, "../escape", "project")


def test_api_key_authentication_and_request_limits():
    authenticator = ApiKeyAuthenticator(
        {
            "alice-secret-key": "alice",
            "bob-secret-key-12": "bob",
        }
    )
    assert (
        authenticator.authenticate(
            "Bearer alice-secret-key"
        ).user_id
        == "alice"
    )
    with pytest.raises(AuthenticationError):
        authenticator.authenticate(None)
    with pytest.raises(AuthenticationError):
        authenticator.authenticate("Bearer invalid-secret")

    limits = ServiceLimits(
        max_message_characters=8,
        max_input_tokens=2,
    )
    limits.validate_message("12345678")
    with pytest.raises(RequestLimitError, match="character"):
        limits.validate_message("123456789")


def test_concurrency_cost_metrics_and_fault_injection():
    limits = ServiceLimits(
        max_concurrent_global=1,
        max_concurrent_per_user=1,
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=2.0,
    )
    limiter = ConcurrencyLimiter(limits)

    async def exercise_limit():
        async with limiter.slot("alice"):
            with pytest.raises(RequestLimitError):
                async with limiter.slot("alice"):
                    pass

    asyncio.run(exercise_limit())

    model_event = ModelCallMetricsEvent(
        call_index=1,
        status="success",
        duration_ms=20,
        first_chunk_ms=5,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        token_source="provider",
    )
    assert limits.model_cost(model_event) == pytest.approx(0.0002)

    metrics = MetricsRegistry()
    metrics.record_envelope(
        EventEnvelope(
            session_id="demo",
            turn_id="turn",
            sequence=1,
            event=model_event,
        )
    )
    metrics.record_turn(
        duration_ms=30,
        failed=False,
        cancelled=False,
        cost_usd=0.0002,
    )
    rendered = metrics.render_prometheus()
    assert "workspace_agent_model_calls_total 1.0" in rendered
    assert "workspace_agent_turn_latency_ms_count 1" in rendered

    injector = FaultInjector({"before_turn"})
    with pytest.raises(RuntimeError, match="injected failure"):
        injector.trigger("before_turn")
    injector.trigger("after_turn")


def test_request_idempotency_store_is_hard_bounded():
    async def exercise():
        store = RequestIdempotencyStore(
            max_entries=1,
            ttl_seconds=60,
        )
        scope = ("alice", "project", "demo")
        mode, first = await store.begin(scope, "request-1", "digest-1")
        assert mode == "new"

        with pytest.raises(RequestLimitError, match="capacity"):
            await store.begin(scope, "request-2", "digest-2")

        await store.complete(first, [b"first"])
        mode, second = await store.begin(
            scope,
            "request-2",
            "digest-2",
        )
        assert mode == "new"
        await store.complete(second, [b"second"])

    asyncio.run(exercise())

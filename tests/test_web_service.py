import asyncio
from collections import deque

import httpx
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool

import session_store
from agent import WorkspaceAgent
from persistent_session import PersistentSession
from production_runtime import ServiceLimits
from storage_security import SnapshotCipher
from tool_execution import ToolExecutionPolicy
from web_service import WebServiceConfig, create_app


class AsyncScriptedModel:
    def __init__(self, responses, *, tools_enabled=False):
        self.responses = (
            responses
            if isinstance(responses, deque)
            else deque(responses)
        )
        self.tools_enabled = tools_enabled

    def bind_tools(self, tools):
        return AsyncScriptedModel(
            self.responses,
            tools_enabled=True,
        )

    async def astream(self, messages):
        await asyncio.sleep(0)
        for chunk in self.responses.popleft():
            yield chunk


def tool_call_response(name, call_id, args="{}"):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": name,
                    "args": args,
                    "id": call_id,
                    "index": 0,
                }
            ],
        )
    ]


def test_authenticated_sse_turn_is_encrypted_isolated_and_replayable(
    tmp_path,
):
    encryption_key = SnapshotCipher.generate_key()
    cipher = SnapshotCipher.from_base64(encryption_key)
    created = []

    def session_factory(paths, session_id):
        created.append((paths.user_id, paths.workspace_id, session_id))
        model = AsyncScriptedModel(
            [[AIMessageChunk(content=f"hello-{paths.user_id}")]]
        )
        agent = WorkspaceAgent(model=model, tools=[])
        backend = session_store.SessionStoreBackend(
            paths.sessions,
            cipher=cipher,
        )
        return PersistentSession(
            session_id,
            agent,
            store_backend=backend,
        )

    app = create_app(
        WebServiceConfig(
            storage_root=tmp_path,
            api_keys={
                "alice-secret-key": "alice",
                "bob-secret-key-12": "bob",
            },
            encryption_key=encryption_key,
        ),
        session_factory=session_factory,
    )

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            unauthenticated = await client.post(
                "/v1/workspaces/project/sessions/demo/turns",
                headers={"Idempotency-Key": "request-unauth"},
                json={"message": "private question"},
            )
            assert unauthenticated.status_code == 401

            headers = {
                "Authorization": "Bearer alice-secret-key",
                "Idempotency-Key": "request-1",
            }
            first = await client.post(
                "/v1/workspaces/project/sessions/demo/turns",
                headers=headers,
                json={"message": "private question"},
            )
            assert first.status_code == 200
            assert first.headers["content-type"].startswith(
                "text/event-stream"
            )
            assert "event: TokenEvent" in first.text
            assert "hello-alice" in first.text

            replay = await client.post(
                "/v1/workspaces/project/sessions/demo/turns",
                headers=headers,
                json={"message": "private question"},
            )
            assert replay.status_code == 200
            assert replay.headers["x-idempotent-replay"] == "true"
            assert replay.content == first.content

            conflict = await client.post(
                "/v1/workspaces/project/sessions/demo/turns",
                headers=headers,
                json={"message": "different"},
            )
            assert conflict.status_code == 409

            bob = await client.post(
                "/v1/workspaces/project/sessions/demo/turns",
                headers={
                    "Authorization": "Bearer bob-secret-key-12",
                    "Idempotency-Key": "request-1",
                },
                json={"message": "bob question"},
            )
            assert bob.status_code == 200
            assert "hello-bob" in bob.text

            metrics = await client.get(
                "/metrics",
                headers={
                    "Authorization": "Bearer alice-secret-key"
                },
            )
            assert metrics.status_code == 200
            assert "workspace_agent_turns_total 2.0" in metrics.text

    asyncio.run(exercise())

    assert created == [
        ("alice", "project", "demo"),
        ("bob", "project", "demo"),
    ]
    alice_snapshot = (
        tmp_path
        / "users"
        / "alice"
        / "workspaces"
        / "project"
        / "sessions"
        / "demo.json"
    ).read_bytes()
    bob_snapshot = (
        tmp_path
        / "users"
        / "bob"
        / "workspaces"
        / "project"
        / "sessions"
        / "demo.json"
    ).read_bytes()
    assert b"private question" not in alice_snapshot
    assert b"bob question" not in bob_snapshot
    assert alice_snapshot != bob_snapshot


def test_independent_approval_api_is_owner_scoped(tmp_path):
    encryption_key = SnapshotCipher.generate_key()
    cipher = SnapshotCipher.from_base64(encryption_key)
    executions = []

    @tool
    def approved_operation(value: str) -> str:
        """Perform an approval-protected operation."""
        executions.append(value)
        return "approved"

    def session_factory(paths, session_id):
        model = AsyncScriptedModel(
            [
                tool_call_response(
                    "approved_operation",
                    "approval-1",
                    '{"value":"once"}',
                ),
                [AIMessageChunk(content="done")],
            ]
        )
        agent = WorkspaceAgent(
            model=model,
            tools=[approved_operation],
            approval_required_tools={"approved_operation"},
            tool_execution_policies={
                "approved_operation": ToolExecutionPolicy(
                    risk="workspace_write",
                    timeout_seconds=None,
                    abandon_on_cancel=False,
                )
            },
        )
        return PersistentSession(
            session_id,
            agent,
            store_backend=session_store.SessionStoreBackend(
                paths.sessions,
                cipher=cipher,
            ),
        )

    app = create_app(
        WebServiceConfig(
            storage_root=tmp_path,
            api_keys={
                "alice-secret-key": "alice",
                "bob-secret-key-12": "bob",
            },
            encryption_key=encryption_key,
            disconnect_poll_seconds=0.005,
        ),
        session_factory=session_factory,
    )

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            turn_task = asyncio.create_task(
                client.post(
                    "/v1/workspaces/project/sessions/demo/turns",
                    headers={
                        "Authorization": "Bearer alice-secret-key",
                        "Idempotency-Key": "approval-request",
                    },
                    json={"message": "approve"},
                )
            )

            approval_path = (
                "/v1/workspaces/project/sessions/demo"
                "/approvals/approval-1"
            )
            accepted = None
            for _ in range(100):
                wrong_owner = await client.post(
                    approval_path,
                    headers={
                        "Authorization": "Bearer bob-secret-key-12"
                    },
                    json={"approved": True},
                )
                assert wrong_owner.status_code == 404
                accepted = await client.post(
                    approval_path,
                    headers={
                        "Authorization": "Bearer alice-secret-key"
                    },
                    json={"approved": True},
                )
                if accepted.status_code == 200:
                    break
                await asyncio.sleep(0.005)

            assert accepted is not None
            assert accepted.status_code == 200
            response = await asyncio.wait_for(turn_task, timeout=2)
            assert response.status_code == 200
            assert "event: ApprovalRequiredEvent" in response.text
            assert "event: ApprovalResolvedEvent" in response.text
            assert '"outcome":"approved"' in response.text

    asyncio.run(exercise())
    assert executions == ["once"]


def test_web_service_rejects_oversized_body_before_session_creation(
    tmp_path,
):
    encryption_key = SnapshotCipher.generate_key()
    created = []
    app = create_app(
        WebServiceConfig(
            storage_root=tmp_path,
            api_keys={"alice-secret-key": "alice"},
            encryption_key=encryption_key,
            limits=ServiceLimits(max_request_bytes=20),
        ),
        session_factory=lambda paths, session_id: created.append(
            session_id
        ),
    )

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/workspaces/project/sessions/demo/turns",
                headers={
                    "Authorization": "Bearer alice-secret-key",
                    "Idempotency-Key": "oversized",
                },
                json={"message": "x" * 100},
            )
            assert response.status_code == 413

    asyncio.run(exercise())
    assert created == []

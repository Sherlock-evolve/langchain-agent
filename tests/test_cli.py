from collections import deque

from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool

import main as cli
import session_store
from agent import WorkspaceAgent
from persistent_session import PersistentSession


CLI_TOOL_EXECUTIONS = []


@tool
def cli_echo_test(value: str) -> str:
    """记录并返回 CLI 测试输入。"""
    CLI_TOOL_EXECUTIONS.append(value)
    return value


class ScriptedModel:
    def __init__(
        self,
        responses,
        *,
        tools_enabled=False,
        call_log=None,
    ):
        self.responses = (
            responses if isinstance(responses, deque) else deque(responses)
        )
        self.tools_enabled = tools_enabled
        self.call_log = call_log if call_log is not None else []

    def bind_tools(self, tools):
        return ScriptedModel(
            self.responses,
            tools_enabled=True,
            call_log=self.call_log,
        )

    def stream(self, messages):
        self.call_log.append(self.tools_enabled)
        if not self.responses:
            raise AssertionError("CLI 测试模型响应队列已耗尽")
        yield from self.responses.popleft()


def tool_call_response(tool_call_id, value):
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": cli_echo_test.name,
                    "args": f'{{"value":"{value}"}}',
                    "id": tool_call_id,
                    "index": 0,
                }
            ],
        )
    ]


def set_inputs(monkeypatch, values):
    responses = iter(values)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(responses),
    )


def test_cli_uses_default_and_named_session_ids(
    monkeypatch,
):
    opened_session_ids = []

    class ExitOnlySession:
        dirty = False

        def __init__(self, session_id):
            self.session_id = session_id

    def fake_open(cls, session_id, agent_factory):
        opened_session_ids.append(session_id)
        return ExitOnlySession(session_id)

    monkeypatch.setattr(
        cli.PersistentSession,
        "open",
        classmethod(fake_open),
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    set_inputs(monkeypatch, ["exit"])
    default_status = cli.main([])
    set_inputs(monkeypatch, ["exit"])
    named_status = cli.main(["--session", "learning"])

    assert default_status == 0
    assert named_status == 0
    assert opened_session_ids == ["default", "learning"]


def test_cli_saves_new_session_and_restores_it_on_next_start(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    first_model = ScriptedModel(
        [[AIMessageChunk(content="持久化 CLI 回答。")]]
    )
    first_agent = WorkspaceAgent(model=first_model, tools=[])
    monkeypatch.setattr(
        cli,
        "create_workspace_agent",
        lambda: first_agent,
    )
    set_inputs(monkeypatch, ["学习问题", "exit"])

    first_status = cli.main(["--session", "learning"])
    first_output = capsys.readouterr().out

    assert first_status == 0
    assert "[会话] 已保存：learning" in first_output
    assert session_store.list_sessions() == ["learning"]

    restored_agents = []

    def restored_factory():
        agent = WorkspaceAgent(
            model=ScriptedModel([]),
            tools=[],
        )
        restored_agents.append(agent)
        return agent

    monkeypatch.setattr(cli, "create_workspace_agent", restored_factory)
    set_inputs(monkeypatch, ["exit"])

    second_status = cli.main(["--session", "learning"])

    assert second_status == 0
    assert len(restored_agents) == 1
    assert [
        message.content
        for message in restored_agents[0].messages
        if isinstance(message, HumanMessage)
    ] == ["学习问题"]
    assert restored_agents[0].messages[-1].content == "持久化 CLI 回答。"


def test_cli_startup_failure_is_nonzero_and_never_enters_input_loop(
    tmp_path,
    monkeypatch,
    capsys,
):
    store_root = tmp_path / ".agent_sessions"
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        store_root,
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    factory_calls = []

    def factory():
        factory_calls.append("called")
        return WorkspaceAgent(model=ScriptedModel([]), tools=[])

    monkeypatch.setattr(cli, "create_workspace_agent", factory)

    def input_must_not_run(prompt=""):
        raise AssertionError("启动失败后不应进入输入循环")

    monkeypatch.setattr("builtins.input", input_must_not_run)

    invalid_status = cli.main(["--session", "../invalid"])
    assert invalid_status != 0
    assert factory_calls == []

    store_root.mkdir(mode=0o700)
    corrupt_file = store_root / "corrupt.json"
    corrupt_file.write_text("{invalid-json", encoding="utf-8")
    corrupt_bytes = corrupt_file.read_bytes()

    corrupt_status = cli.main(["--session", "corrupt"])
    assert corrupt_status != 0
    assert corrupt_file.read_bytes() == corrupt_bytes
    assert factory_calls == []

    semantic_snapshot = {
        "version": 2,
        "messages": [],
        "memory_summary": "",
    }
    session_store.save("semantic", semantic_snapshot)
    semantic_file = store_root / "semantic.json"
    semantic_bytes = semantic_file.read_bytes()

    semantic_status = cli.main(["--session", "semantic"])
    output = capsys.readouterr().out

    assert semantic_status != 0
    assert semantic_file.read_bytes() == semantic_bytes
    assert factory_calls == ["called"]
    assert "[启动失败]" in output
    assert "Traceback" not in output


def test_cli_dirty_mode_retries_without_replaying_model_or_tool(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    CLI_TOOL_EXECUTIONS.clear()
    model = ScriptedModel(
        [
            tool_call_response("cli-dirty-call", "side-effect"),
            [AIMessageChunk(content="保存失败前的回答。")],
        ]
    )
    agent = WorkspaceAgent(model=model, tools=[cli_echo_test])
    monkeypatch.setattr(
        cli,
        "create_workspace_agent",
        lambda: agent,
    )
    real_save = session_store.save
    save_attempts = []

    def flaky_save(session_id, snapshot):
        save_attempts.append(session_id)
        if len(save_attempts) == 1:
            raise session_store.SessionStoreError("模拟 CLI 保存失败")
        real_save(session_id, snapshot)

    monkeypatch.setattr(session_store, "save", flaky_save)
    set_inputs(
        monkeypatch,
        [
            "执行工具",
            "这条问题必须被阻止",
            ":retry",
            "exit",
        ],
    )

    status = cli.main(["--session", "dirty-cli"])
    output = capsys.readouterr().out

    assert status == 0
    assert CLI_TOOL_EXECUTIONS == ["side-effect"]
    assert model.call_log == [True, True]
    assert save_attempts == ["dirty-cli", "dirty-cli"]
    assert "[保存失败]" in output
    assert "存在未保存状态" in output
    assert "[会话] 重试保存成功：dirty-cli" in output
    assert "Traceback" not in output
    assert session_store.load("dirty-cli") == agent.export_snapshot()


def test_cli_closes_active_approval_stream_on_exit_signals(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    signal_cases = [
        ("eof", EOFError(), 0),
        ("keyboard", KeyboardInterrupt(), 130),
        ("exit", "exit", 0),
    ]

    for label, signal, expected_status in signal_cases:
        monkeypatch.setattr(
            session_store,
            "SESSION_STORE_ROOT",
            tmp_path / label / ".agent_sessions",
        )
        CLI_TOOL_EXECUTIONS.clear()
        model = ScriptedModel(
            [
                tool_call_response(f"{label}-approval", label),
                [AIMessageChunk(content="锁已释放。")],
            ]
        )
        agent = WorkspaceAgent(
            model=model,
            tools=[cli_echo_test],
            approval_required_tools={cli_echo_test.name},
        )
        monkeypatch.setattr(
            cli,
            "create_workspace_agent",
            lambda agent=agent: agent,
        )
        input_calls = 0

        def interrupted_input(prompt=""):
            nonlocal input_calls
            input_calls += 1
            if input_calls == 1:
                return "请求审批"
            if isinstance(signal, BaseException):
                raise signal
            return signal

        monkeypatch.setattr("builtins.input", interrupted_input)

        status = cli.main(["--session", f"signal-{label}"])

        assert status == expected_status
        assert CLI_TOOL_EXECUTIONS == []
        assert session_store.list_sessions() == []
        next_events = list(agent.stream_turn("验证锁释放"))
        assert [
            event
            for event in next_events
            if not isinstance(event, cli.ModelCallMetricsEvent)
        ] == [
            cli.TokenEvent(text="锁已释放。")
        ]


def test_cli_shows_current_and_sorted_session_list(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    snapshot = WorkspaceAgent(
        model=ScriptedModel([]),
        tools=[],
    ).export_snapshot()
    session_store.save("zeta", snapshot)
    session_store.save("alpha", snapshot)

    current = PersistentSession(
        "middle",
        WorkspaceAgent(model=ScriptedModel([]), tools=[]),
    )
    current._dirty = True
    set_inputs(monkeypatch, [":session", ":sessions", "exit"])

    status = cli.run_cli(
        current,
        lambda: WorkspaceAgent(model=ScriptedModel([]), tools=[]),
    )
    output = capsys.readouterr().out

    assert status == 1
    assert "[会话] 当前：middle（dirty：是）" in output
    assert "* middle（当前，dirty）" in output
    list_output = output.split("[会话列表]", 1)[1]
    assert (
        list_output.index("alpha")
        < list_output.index("middle")
        < list_output.index("zeta")
    )


def test_cli_switches_existing_and_new_sessions_with_independent_agents(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    saved_agent = WorkspaceAgent(
        model=ScriptedModel(
            [[AIMessageChunk(content="旧会话回答")]]
        ),
        tools=[],
    )
    list(saved_agent.stream_turn("旧会话问题"))
    session_store.save("saved", saved_agent.export_snapshot())

    original_agent = WorkspaceAgent(
        model=ScriptedModel([]),
        tools=[],
    )
    original = PersistentSession("original", original_agent)
    created_agents = []

    def agent_factory():
        agent = WorkspaceAgent(model=ScriptedModel([]), tools=[])
        created_agents.append(agent)
        return agent

    set_inputs(
        monkeypatch,
        [":switch saved", ":switch fresh", "exit"],
    )

    status = cli.run_cli(original, agent_factory)
    output = capsys.readouterr().out

    assert status == 0
    assert "[会话] 已切换：saved" in output
    assert "[会话] 已切换：fresh" in output
    assert len(created_agents) == 2
    assert [
        message.content
        for message in created_agents[0].messages
        if isinstance(message, HumanMessage)
    ] == ["旧会话问题"]
    assert [type(message) for message in created_agents[1].messages] == [
        SystemMessage
    ]
    assert original_agent is not created_agents[0]
    assert created_agents[0] is not created_agents[1]


def test_cli_failed_switch_keeps_current_session(
    tmp_path,
    monkeypatch,
    capsys,
):
    store_root = tmp_path / ".agent_sessions"
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        store_root,
    )
    store_root.mkdir(mode=0o700)
    corrupt_path = store_root / "broken.json"
    corrupt_path.write_text("{broken", encoding="utf-8")
    corrupt_bytes = corrupt_path.read_bytes()

    current_model = ScriptedModel(
        [[AIMessageChunk(content="仍由原会话回答")]]
    )
    current_agent = WorkspaceAgent(model=current_model, tools=[])
    current = PersistentSession("current", current_agent)
    candidate_factory_calls = []

    def candidate_factory():
        candidate_factory_calls.append("called")
        return WorkspaceAgent(model=ScriptedModel([]), tools=[])

    set_inputs(
        monkeypatch,
        [":switch broken", "继续原会话", "exit"],
    )

    status = cli.run_cli(current, candidate_factory)
    output = capsys.readouterr().out

    assert status == 0
    assert "[切换失败]" in output
    assert "仍由原会话回答" in output
    assert candidate_factory_calls == []
    assert current_model.call_log == [True]
    assert current_agent.messages[-2].content == "继续原会话"
    assert session_store.load("current") == current_agent.export_snapshot()
    assert corrupt_path.read_bytes() == corrupt_bytes


def test_cli_delete_requires_confirmation_and_never_deletes_current(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    agent = WorkspaceAgent(model=ScriptedModel([]), tools=[])
    snapshot = agent.export_snapshot()
    session_store.save("current", snapshot)
    session_store.save("target", snapshot)
    current = PersistentSession("current", agent)
    inputs = deque(
        [
            ":delete target",
            EOFError(),
            ":delete target",
            KeyboardInterrupt(),
            ":delete target",
            "n",
            ":delete target",
            "yes",
            ":delete current",
            "exit",
        ]
    )

    def scripted_input(prompt=""):
        value = inputs.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", scripted_input)

    status = cli.run_cli(
        current,
        lambda: WorkspaceAgent(model=ScriptedModel([]), tools=[]),
    )
    output = capsys.readouterr().out

    assert status == 0
    assert output.count("[会话] 已取消删除") == 3
    assert "[会话] 已删除：target" in output
    assert "[会话] 不能删除当前会话" in output
    assert session_store.list_sessions() == ["current"]


def test_cli_dirty_mode_blocks_mutations_and_bad_commands_show_help(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        session_store,
        "SESSION_STORE_ROOT",
        tmp_path / ".agent_sessions",
    )
    snapshot = WorkspaceAgent(
        model=ScriptedModel([]),
        tools=[],
    ).export_snapshot()
    session_store.save("other", snapshot)

    model = ScriptedModel([])
    current = PersistentSession(
        "current",
        WorkspaceAgent(model=model, tools=[]),
    )
    current._dirty = True
    factory_calls = []
    delete_calls = []
    real_delete = session_store.delete

    def tracked_factory():
        factory_calls.append("called")
        return WorkspaceAgent(model=ScriptedModel([]), tools=[])

    def tracked_delete(session_id):
        delete_calls.append(session_id)
        real_delete(session_id)

    monkeypatch.setattr(session_store, "delete", tracked_delete)
    set_inputs(
        monkeypatch,
        [
            ":switch other",
            ":delete other",
            ":session",
            ":sessions",
            ":switch",
            ":delete a b",
            ":switch ../bad",
            ":unknown",
            ":help",
            "不能发送给模型",
            "exit",
        ],
    )

    status = cli.run_cli(current, tracked_factory)
    output = capsys.readouterr().out

    assert status == 1
    assert factory_calls == []
    assert delete_calls == []
    assert session_store.list_sessions() == ["other"]
    assert model.call_log == []
    assert "不能切换会话" in output
    assert "不能删除会话" in output
    assert "[会话] 当前：current（dirty：是）" in output
    assert output.count("可用命令：") >= 5
    assert "存在未保存状态，请先输入 :retry 或退出" in output

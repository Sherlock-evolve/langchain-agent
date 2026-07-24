import argparse
import json
import os
from collections.abc import Callable, Iterable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

import session_store
from agent import WorkspaceAgent
from audit_log import JsonlAuditLogger
from contracts import (
    AgentEvent,
    ApprovalDecision,
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
    TurnCancelledEvent,
)
from knowledge_runtime import (
    KnowledgeRuntime,
    create_knowledge_runtime,
)
from persistent_session import (
    PersistentSession,
    PersistentSessionSaveError,
)
from tool_execution import (
    ToolExecutionMiddleware,
    ToolExecutionPolicy,
)
from tools import (
    list_files,
    prepare_write_file,
    read_file,
    search_text,
    write_file,
)


ai_label_printed = False
cursor_at_line_start = True
STATUS_LABELS = {
    "success": "成功",
    "error": "失败",
}
APPROVAL_OUTCOME_LABELS = {
    "approved": "已批准",
    "rejected": "已拒绝",
    "missing": "缺少决定",
    "mismatched": "调用 ID 不匹配",
    "invalid": "无效决定",
}
EXIT_COMMANDS = {"exit", "quit", "退出"}
HELP_TEXT = """可用命令：
  :session              显示当前会话状态
  :sessions             按名称列出会话
  :switch <session_id>  切换或新建会话
  :delete <session_id>  删除非当前会话
  :pending              显示等待恢复的审批
  :resume               恢复并重新确认审批
  :retry                重试保存未保存状态
  :help                 显示此帮助
  exit / quit / 退出    退出程序"""


class _ExitRequested(Exception):
    pass


def start_turn() -> None:
    global ai_label_printed, cursor_at_line_start

    print()
    ai_label_printed = False
    cursor_at_line_start = True


def finish_turn() -> None:
    _ensure_line_start()


def render_event(event: AgentEvent) -> None:
    global ai_label_printed, cursor_at_line_start

    if isinstance(event, TokenEvent):
        if not ai_label_printed:
            print("AI：", end="", flush=True)
            ai_label_printed = True
        print(event.text, end="", flush=True)
        cursor_at_line_start = event.text.endswith("\n")
        return

    _ensure_line_start()

    if isinstance(event, ToolCallEvent):
        args_text = json.dumps(
            event.args,
            ensure_ascii=False,
            sort_keys=True,
        )
        print(f"[第 {event.step} 轮工具调用] {event.name} {args_text}")
    elif isinstance(event, ToolResultEvent):
        duration_label = (
            f"，耗时 {event.duration_ms} ms"
            if event.duration_ms is not None
            else ""
        )
        if event.status == "skipped":
            print(f"[工具跳过] {event.detail}{duration_label}")
        else:
            status_label = STATUS_LABELS[event.status]
            truncated_label = "（已截断）" if event.truncated else ""
            print(
                f"[工具结果] {status_label}，"
                f"返回 {event.character_count} 个字符"
                f"{truncated_label}"
                f"{duration_label}"
            )
    elif isinstance(event, ApprovalRequiredEvent):
        print(f"[审批] 工具 {event.tool_name} 等待用户确认")
        if event.preview:
            print(event.preview)
    elif isinstance(event, ApprovalResolvedEvent):
        print(
            "[审批结果] "
            f"{event.tool_name}："
            f"{APPROVAL_OUTCOME_LABELS[event.outcome]}"
        )
    elif isinstance(event, SystemEvent):
        print(f"[系统] {event.message}")
    elif isinstance(event, ContextTrimmedEvent):
        print(f"[上下文] 已移除 {event.removed_message_count} 条旧消息")
    elif isinstance(event, MemoryUpdatedEvent):
        print(f"[记忆] 长期摘要已更新，共 {event.character_count} 个字符")
    elif isinstance(event, CitationValidationEvent):
        if event.status == "valid":
            print(
                f"[引用] 有效，引用 {event.citation_count}，"
                f"未知 {event.unknown_citation_count}，"
                f"可用资料 {event.retrieved_chunk_count}"
            )
        elif event.status == "missing":
            print(
                "[引用] 缺少引用，"
                f"可用资料 {event.retrieved_chunk_count}"
            )
        elif event.status == "unknown":
            print(
                "[引用] 检测到 "
                f"{event.unknown_citation_count} 个未知引用"
            )
        elif event.status == "not_applicable":
            print("[引用] 本轮未使用知识检索")
        else:
            print("[引用] 校验失败")
    elif isinstance(event, CitationPolicyEvent):
        if event.action == "observed":
            print("[引用策略] 已观测")
        elif event.action == "allowed":
            print("[引用策略] 校验通过，允许提交")
        else:
            print("[引用策略] 校验未通过，回答未提交")
    elif isinstance(event, ModelCallMetricsEvent):
        status_label = STATUS_LABELS[event.status]
        first_chunk = (
            f"{event.first_chunk_ms} ms"
            if event.first_chunk_ms is not None
            else "无"
        )
        tokens = (
            f"{event.total_tokens}"
            if event.token_source == "provider"
            else "不可用"
        )
        error_label = (
            f"，错误类型 {event.error_type}"
            if event.error_type
            else ""
        )
        print(
            f"[模型调用 #{event.call_index}] {status_label}，"
            f"耗时 {event.duration_ms} ms，"
            f"首块 {first_chunk}，总 tokens {tokens}"
            f"{error_label}"
        )
    elif isinstance(event, SessionSavedEvent):
        print(f"[会话] 已保存：{event.session_id}")
    elif isinstance(event, TurnCancelledEvent):
        print(f"[取消] 当前轮次已取消：{event.reason}")


def _ensure_line_start() -> None:
    global cursor_at_line_start

    if not cursor_at_line_start:
        print()
        cursor_at_line_start = True


def create_workspace_agent(
    extra_tools: Iterable[BaseTool] | None = None,
    citation_validator: Callable | None = None,
    citation_policy: str = "observe",
    citation_guard_tool_names: set[str] | frozenset[str] | None = None,
) -> WorkspaceAgent:
    model = ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL"),
        temperature=0,
    )
    registered_tools = [
        list_files,
        read_file,
        search_text,
        write_file,
    ]
    if extra_tools is not None:
        registered_tools.extend(tuple(extra_tools))
    execution_policies = {
        registered_tool.name: ToolExecutionPolicy(
            risk="read_only",
            timeout_seconds=30.0,
            abandon_on_cancel=True,
        )
        for registered_tool in registered_tools
    }
    execution_policies[write_file.name] = ToolExecutionPolicy(
        risk="workspace_write",
        timeout_seconds=None,
        abandon_on_cancel=False,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=registered_tools,
        approval_required_tools={write_file.name},
        approval_preparers={write_file.name: prepare_write_file},
        citation_validator=citation_validator,
        citation_policy=citation_policy,
        citation_guard_tool_names=set(citation_guard_tool_names or ()),
        tool_execution_middleware=ToolExecutionMiddleware(
            execution_policies,
            require_registered_policies=True,
        ),
    )
    return agent


def create_workspace_agent_factory(
    extra_tools: Iterable[BaseTool],
    citation_validator: Callable | None = None,
    citation_policy: str = "observe",
    citation_guard_tool_names: set[str] | frozenset[str] | None = None,
) -> Callable[[], WorkspaceAgent]:
    shared_tools = tuple(extra_tools)
    shared_guard_tool_names = frozenset(
        citation_guard_tool_names or ()
    )

    def agent_factory() -> WorkspaceAgent:
        return create_workspace_agent(
            extra_tools=shared_tools,
            citation_validator=citation_validator,
            citation_policy=citation_policy,
            citation_guard_tool_names=shared_guard_tool_names,
        )

    return agent_factory


def _show_knowledge_runtime(runtime: KnowledgeRuntime) -> None:
    corpus_prefix = runtime.corpus_id[:12]
    print(
        "[知识库] "
        f"已索引文件 {runtime.indexed_file_count}，"
        f"分块 {runtime.chunk_count}，"
        f"跳过文件 {runtime.skipped_file_count}，"
        f"复用向量 {runtime.reused_embedding_count}，"
        f"新增向量 {runtime.created_embedding_count}，"
        f"corpus_id {corpus_prefix}"
    )


def _drive_turn(
    session: PersistentSession,
    question: str,
    audit_logger: JsonlAuditLogger | None = None,
) -> None:
    stream = session.stream_turn(question)
    _drive_event_stream(
        stream,
        audit_logger=audit_logger,
    )


def _drive_pending_approval(
    session: PersistentSession,
    audit_logger: JsonlAuditLogger | None = None,
) -> None:
    stream = session.stream_resume_pending_approval()
    _drive_event_stream(
        stream,
        audit_logger=audit_logger,
    )


def _drive_event_stream(
    stream,
    audit_logger: JsonlAuditLogger | None = None,
) -> None:
    start_turn()
    decision = None
    audit_warning_shown = False

    try:
        while True:
            try:
                envelope = stream.send(decision)
            except StopIteration:
                break

            decision = None
            if not isinstance(envelope, EventEnvelope):
                raise TypeError("持久会话返回了无效的事件信封")
            if audit_logger is not None:
                try:
                    audit_logger.record(envelope)
                except Exception:
                    if not audit_warning_shown:
                        _ensure_line_start()
                        print("[审计警告] 当前轮审计日志写入失败")
                        audit_warning_shown = True
            event = envelope.event
            render_event(event)

            if isinstance(event, ApprovalRequiredEvent):
                answer = input("是否允许执行？[y/N] ")
                normalized_answer = answer.strip().lower()
                if normalized_answer in EXIT_COMMANDS:
                    raise _ExitRequested
                decision = ApprovalDecision(
                    tool_call_id=event.tool_call_id,
                    approved=normalized_answer in {"y", "yes"},
                )
    except (EOFError, KeyboardInterrupt, _ExitRequested):
        raise
    except PersistentSessionSaveError as error:
        _ensure_line_start()
        print(f"[保存失败] {error}")
    except Exception as error:
        _ensure_line_start()
        print(f"[运行失败] {error}")
    finally:
        stream.close()
        finish_turn()


def _exit_status(
    session: PersistentSession,
    *,
    interrupted: bool = False,
) -> int:
    if session.dirty:
        print("[警告] 当前会话仍有未保存状态")
        if not interrupted:
            return 1
    return 130 if interrupted else 0


def _print_help() -> None:
    print(HELP_TEXT)


def _valid_session_id(session_id: str) -> bool:
    return session_store.SESSION_ID_PATTERN.fullmatch(session_id) is not None


def _show_current_session(session: PersistentSession) -> None:
    dirty_state = "是" if session.dirty else "否"
    print(f"[会话] 当前：{session.session_id}（dirty：{dirty_state}）")
    if session.has_pending_approval:
        print("[会话] 当前存在待恢复审批")


def _show_sessions(session: PersistentSession) -> None:
    try:
        session_ids = set(session_store.list_sessions())
    except session_store.SessionStoreError as error:
        print(f"[会话列表失败] {error}")
        return

    session_ids.add(session.session_id)
    print("[会话列表]")
    for session_id in sorted(session_ids):
        if session_id == session.session_id:
            dirty_label = "，dirty" if session.dirty else ""
            pending_label = (
                "，待审批" if session.has_pending_approval else ""
            )
            print(
                f"* {session_id}"
                f"（当前{dirty_label}{pending_label}）"
            )
        else:
            print(f"  {session_id}")


def _retry_save(session: PersistentSession) -> None:
    if not session.dirty:
        print("[会话] 当前没有未保存状态")
        return

    try:
        session.flush()
    except PersistentSessionSaveError as error:
        print(f"[保存失败] {error}")
    except Exception as error:
        print(f"[保存失败] {error}")
    else:
        print(f"[会话] 重试保存成功：{session.session_id}")


def _show_pending_approval(session: PersistentSession) -> None:
    event = session.pending_approval_event()
    if event is None:
        print("[审批恢复] 当前没有待审批轮次")
        return
    args_text = json.dumps(
        event.args,
        ensure_ascii=False,
        sort_keys=True,
    )
    print(
        "[审批恢复] "
        f"工具 {event.tool_name}，参数 {args_text}；"
        "输入 :resume 重新生成预览并确认"
    )


def _switch_session(
    session: PersistentSession,
    session_id: str,
    agent_factory: Callable[[], WorkspaceAgent],
) -> PersistentSession:
    if session.dirty:
        print("[会话] 存在未保存状态，不能切换会话；请先输入 :retry")
        return session
    if session_id == session.session_id:
        print(f"[会话] 已经位于：{session_id}")
        return session

    try:
        candidate = PersistentSession.open(session_id, agent_factory)
    except Exception as error:
        print(f"[切换失败] {error}")
        return session

    print(f"[会话] 已切换：{session_id}")
    return candidate


def _delete_session(
    session: PersistentSession,
    session_id: str,
) -> None:
    if session.dirty:
        print("[会话] 存在未保存状态，不能删除会话；请先输入 :retry")
        return
    if session_id == session.session_id:
        print("[会话] 不能删除当前会话")
        return

    try:
        answer = input(f"确认删除会话 {session_id}？[y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        print("[会话] 已取消删除")
        return

    if answer.strip().lower() not in {"y", "yes"}:
        print("[会话] 已取消删除")
        return

    try:
        session_store.delete(session_id)
    except session_store.SessionStoreError as error:
        print(f"[删除失败] {error}")
    except Exception as error:
        print(f"[删除失败] {error}")
    else:
        print(f"[会话] 已删除：{session_id}")


def _handle_command(
    session: PersistentSession,
    command_text: str,
    agent_factory: Callable[[], WorkspaceAgent],
    audit_logger: JsonlAuditLogger | None = None,
) -> tuple[bool, PersistentSession]:
    if not command_text.startswith(":"):
        return False, session

    parts = command_text.split()
    command = parts[0].lower()
    arguments = parts[1:]

    if command == ":session" and not arguments:
        _show_current_session(session)
    elif command == ":sessions" and not arguments:
        _show_sessions(session)
    elif command == ":retry" and not arguments:
        _retry_save(session)
    elif command == ":pending" and not arguments:
        _show_pending_approval(session)
    elif command == ":resume" and not arguments:
        if session.dirty:
            print("[会话] 存在未保存状态，请先输入 :retry")
        elif not session.has_pending_approval:
            print("[审批恢复] 当前没有待审批轮次")
        else:
            _drive_pending_approval(
                session,
                audit_logger=audit_logger,
            )
    elif command == ":switch" and len(arguments) == 1:
        session_id = arguments[0]
        if not _valid_session_id(session_id):
            _print_help()
        else:
            session = _switch_session(
                session,
                session_id,
                agent_factory,
            )
    elif command == ":delete" and len(arguments) == 1:
        session_id = arguments[0]
        if not _valid_session_id(session_id):
            _print_help()
        else:
            _delete_session(session, session_id)
    elif command == ":help" and not arguments:
        _print_help()
    else:
        _print_help()

    return True, session


def run_cli(
    session: PersistentSession,
    agent_factory: Callable[[], WorkspaceAgent] | None = None,
    audit_logger: JsonlAuditLogger | None = None,
) -> int:
    if agent_factory is None:
        agent_factory = create_workspace_agent

    try:
        while True:
            question = input("\n你：").strip()

            if question.lower() in EXIT_COMMANDS:
                return _exit_status(session)

            if not question:
                continue

            handled, session = _handle_command(
                session,
                question,
                agent_factory,
                audit_logger=audit_logger,
            )
            if handled:
                continue

            if session.dirty:
                print("[会话] 存在未保存状态，请先输入 :retry 或退出")
                continue
            if session.has_pending_approval:
                print(
                    "[审批恢复] 当前会话存在待审批轮次，"
                    "请先输入 :resume 或切换会话"
                )
                continue

            _drive_turn(
                session,
                question,
                audit_logger=audit_logger,
            )
    except EOFError:
        return _exit_status(session)
    except KeyboardInterrupt:
        print()
        return _exit_status(session, interrupted=True)
    except _ExitRequested:
        return _exit_status(session)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="工作区 Agent",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="持久会话 ID（默认：default）",
    )
    parser.add_argument(
        "--enable-knowledge",
        action="store_true",
        help="显式启用 docs 语义检索（可能调用外部 Embeddings 服务）",
    )
    parser.add_argument(
        "--knowledge-directory",
        default="docs",
        help="工作区内的知识文档目录（默认：docs）",
    )
    parser.add_argument(
        "--citation-policy",
        choices=("observe", "require-valid"),
        default="observe",
        help="引用策略（默认：observe）",
    )
    return parser


def main(
    argv=None,
    knowledge_runtime_factory=None,
) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    load_dotenv()
    citation_policy = arguments.citation_policy.replace("-", "_")

    if (
        citation_policy == "require_valid"
        and not arguments.enable_knowledge
    ):
        print("[启动失败] require-valid 引用策略必须启用知识库")
        return 2

    runtime = None
    agent_factory = create_workspace_agent
    if arguments.enable_knowledge:
        if knowledge_runtime_factory is None:
            knowledge_runtime_factory = create_knowledge_runtime
        try:
            runtime = knowledge_runtime_factory(
                knowledge_directory=arguments.knowledge_directory,
            )
            if not isinstance(runtime, KnowledgeRuntime):
                raise TypeError("invalid knowledge runtime")
            agent_factory = create_workspace_agent_factory(
                [runtime.search_tool],
                citation_validator=runtime.citation_validator,
                citation_policy=citation_policy,
                citation_guard_tool_names=(
                    runtime.citation_guard_tool_names
                ),
            )
        except Exception:
            print("[知识库启动失败] 无法安全初始化知识库")
            return 2

    try:
        session = PersistentSession.open(
            arguments.session,
            agent_factory,
        )
    except Exception as error:
        print(f"[启动失败] {error}")
        return 2

    if runtime is not None:
        _show_knowledge_runtime(runtime)
    if getattr(session, "has_pending_approval", False):
        _show_pending_approval(session)

    return run_cli(
        session,
        agent_factory,
        audit_logger=JsonlAuditLogger(),
    )


if __name__ == "__main__":
    raise SystemExit(main())

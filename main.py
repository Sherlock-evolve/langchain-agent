import argparse
import json
import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import session_store
from agent import WorkspaceAgent
from contracts import (
    AgentEvent,
    ApprovalDecision,
    ApprovalRequiredEvent,
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
from persistent_session import (
    PersistentSession,
    PersistentSessionSaveError,
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
EXIT_COMMANDS = {"exit", "quit", "退出"}
HELP_TEXT = """可用命令：
  :session              显示当前会话状态
  :sessions             按名称列出会话
  :switch <session_id>  切换或新建会话
  :delete <session_id>  删除非当前会话
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
        if event.status == "skipped":
            print(f"[工具跳过] {event.detail}")
        else:
            status_label = STATUS_LABELS[event.status]
            truncated_label = "（已截断）" if event.truncated else ""
            print(
                f"[工具结果] {status_label}，"
                f"返回 {event.character_count} 个字符"
                f"{truncated_label}"
            )
    elif isinstance(event, ApprovalRequiredEvent):
        print(f"[审批] 工具 {event.tool_name} 等待用户确认")
        if event.preview:
            print(event.preview)
    elif isinstance(event, SystemEvent):
        print(f"[系统] {event.message}")
    elif isinstance(event, ContextTrimmedEvent):
        print(f"[上下文] 已移除 {event.removed_message_count} 条旧消息")
    elif isinstance(event, MemoryUpdatedEvent):
        print(f"[记忆] 长期摘要已更新，共 {event.character_count} 个字符")
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


def _ensure_line_start() -> None:
    global cursor_at_line_start

    if not cursor_at_line_start:
        print()
        cursor_at_line_start = True


def create_workspace_agent() -> WorkspaceAgent:
    model = ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL"),
        temperature=0,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[list_files, read_file, search_text, write_file],
        approval_required_tools={write_file.name},
        approval_preparers={write_file.name: prepare_write_file},
    )
    return agent


def _drive_turn(
    session: PersistentSession,
    question: str,
) -> None:
    start_turn()
    stream = session.stream_turn(question)
    decision = None

    try:
        while True:
            try:
                envelope = stream.send(decision)
            except StopIteration:
                break

            decision = None
            if not isinstance(envelope, EventEnvelope):
                raise TypeError("持久会话返回了无效的事件信封")
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
            print(f"* {session_id}（当前{dirty_label}）")
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
            )
            if handled:
                continue

            if session.dirty:
                print("[会话] 存在未保存状态，请先输入 :retry 或退出")
                continue

            _drive_turn(session, question)
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
    return parser


def main(argv=None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    load_dotenv()

    try:
        session = PersistentSession.open(
            arguments.session,
            create_workspace_agent,
        )
    except Exception as error:
        print(f"[启动失败] {error}")
        return 2

    return run_cli(session, create_workspace_agent)


if __name__ == "__main__":
    raise SystemExit(main())

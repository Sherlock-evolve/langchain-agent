import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agent import (
    AgentEvent,
    ContextTrimmedEvent,
    MemoryUpdatedEvent,
    SystemEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkspaceAgent,
)
from tools import list_files, read_file, search_text


ai_label_printed = False
cursor_at_line_start = True
STATUS_LABELS = {
    "success": "成功",
    "error": "失败",
}


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
    elif isinstance(event, SystemEvent):
        print(f"[系统] {event.message}")
    elif isinstance(event, ContextTrimmedEvent):
        print(f"[上下文] 已移除 {event.removed_message_count} 条旧消息")
    elif isinstance(event, MemoryUpdatedEvent):
        print(f"[记忆] 长期摘要已更新，共 {event.character_count} 个字符")


def _ensure_line_start() -> None:
    global cursor_at_line_start

    if not cursor_at_line_start:
        print()
        cursor_at_line_start = True


def main() -> None:
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("ZHIPU_MODEL"),
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL"),
        temperature=0,
    )
    agent = WorkspaceAgent(
        model=model,
        tools=[list_files, read_file, search_text],
    )

    while True:
        question = input("\n你：").strip()

        if question.lower() in {"exit", "quit", "退出"}:
            break

        if not question:
            continue

        start_turn()
        for event in agent.stream_turn(question):
            render_event(event)
        finish_turn()


if __name__ == "__main__":
    main()

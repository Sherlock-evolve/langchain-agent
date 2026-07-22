import json
import os

from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_openai import ChatOpenAI

from tools import list_files, read_file, search_text


load_dotenv()

model = ChatOpenAI(
    model=os.getenv("ZHIPU_MODEL"),
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url=os.getenv("ZHIPU_BASE_URL"),
    temperature=0,
)
tools = [list_files, read_file, search_text]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

MAX_AGENT_LOOPS = 5

messages = [
    SystemMessage(
        content=(
            "你是一位人工智能老师，也是当前项目的工作区助手。"
            "需要了解项目文件时，请使用工具获取真实信息，不要猜测。"
            "请用通俗、准确的方式回答。"
        )
    )
]

while True:
    question = input("\n你：").strip()

    if question.lower() in {"exit", "quit", "退出"}:
        break

    if not question:
        continue

    messages.append(HumanMessage(content=question))

    print()
    ai_label_printed = False
    cursor_at_line_start = True
    answered = False
    task_stopped = False

    for step in range(1, MAX_AGENT_LOOPS + 1):
        response_chunk = None

        for chunk in model_with_tools.stream(messages):
            if chunk.content:
                if not ai_label_printed:
                    print("AI：", end="", flush=True)
                    ai_label_printed = True
                print(chunk.content, end="", flush=True)
                cursor_at_line_start = chunk.content.endswith("\n")

            if response_chunk is None:
                response_chunk = chunk
            else:
                response_chunk = response_chunk + chunk

        if response_chunk is None:
            if not cursor_at_line_start:
                print()
                cursor_at_line_start = True
            print("[系统] 模型未返回任何消息，当前任务已停止。")
            task_stopped = True
            break

        response = message_chunk_to_message(response_chunk)
        messages.append(response)

        if not response.tool_calls:
            if not cursor_at_line_start:
                print()
                cursor_at_line_start = True
            answered = True
            break

        if not cursor_at_line_start:
            print()
            cursor_at_line_start = True

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args_text = json.dumps(
                tool_call["args"],
                ensure_ascii=False,
                sort_keys=True,
            )
            print(f"[第 {step} 轮工具调用] {tool_name} {args_text}")

            try:
                selected_tool = tools_by_name[tool_name]
                tool_result_text = str(selected_tool.invoke(tool_call["args"]))
                tool_succeeded = True
            except Exception as error:
                tool_result_text = f"工具执行失败：{error}"
                tool_succeeded = False

            result_status = "成功" if tool_succeeded else "失败"
            print(
                f"[工具结果] {result_status}，"
                f"返回 {len(tool_result_text)} 个字符"
            )

            messages.append(
                ToolMessage(
                    content=tool_result_text,
                    tool_call_id=tool_call["id"],
                )
            )

    if not answered:
        if not task_stopped:
            if not cursor_at_line_start:
                print()
            print(f"[系统] Agent 循环达到 {MAX_AGENT_LOOPS} 次上限，已停止。")
        continue

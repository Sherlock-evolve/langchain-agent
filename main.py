import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import read_note


load_dotenv()

model = ChatOpenAI(
    model=os.getenv("ZHIPU_MODEL"),
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url=os.getenv("ZHIPU_BASE_URL"),
    temperature=0.7,
)
model_with_tools = model.bind_tools([read_note])
tools_by_name = {read_note.name: read_note}

MAX_AGENT_LOOPS = 5

messages = [
    SystemMessage(
        content="你是一位人工智能老师，请用通俗、准确的方式回答。"
    )
]

while True:
    question = input("\n你：").strip()

    if question.lower() in {"exit", "quit", "退出"}:
        break

    if not question:
        continue

    messages.append(HumanMessage(content=question))

    answered = False
    for _ in range(MAX_AGENT_LOOPS):
        response = model_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\nAI：{response.content}")
            answered = True
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            selected_tool = tools_by_name[tool_name]

            print(f"[工具调用] {tool_name}")
            tool_result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    if not answered:
        print(f"\nAI：Agent 循环达到 {MAX_AGENT_LOOPS} 次上限，已停止。")
        break

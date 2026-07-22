from pathlib import Path

from langchain_core.tools import tool


NOTE_PATH = Path(__file__).resolve().parent / "docs" / "note.txt"


@tool
def read_note() -> str:
    """读取项目中 docs/note.txt 的完整内容；仅当用户询问这份笔记的内容时使用。"""
    return NOTE_PATH.read_text(encoding="utf-8")

# schema.py
from typing import TypedDict, List, Optional


class PDFPage(TypedDict):
    page_number: int
    content: str


class AgentState(TypedDict):
    file_path: str

    # 变更 1: 存储分页内容，而不是纯文本
    pdf_pages: List[PDFPage]

    # 变更 2: 增加思考日志，用于"可视化思考"
    thought_log: List[str]

    summary: str
    key_concepts: List[str]
    search_results: dict
    final_report: str

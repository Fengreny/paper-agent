# schema.py
from typing import TypedDict, List, Optional


class PDFPage(TypedDict):
    page_number: int
    content: str


class AgentState(TypedDict):
    file_path: str

    
    pdf_pages: List[PDFPage]

    
    thought_log: List[str]

    summary: str
    key_concepts: List[str]
    search_results: dict
    final_report: str

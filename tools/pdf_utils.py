# tools/pdf_utils.py
from pathlib import Path
from typing import List
from pypdf import PdfReader
from schema import PDFPage


def read_pdf_with_pages(path: str) -> List[PDFPage]:
    """读取 PDF 并保留页码信息"""
    reader = PdfReader(path)
    pages_data = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # 清理一下多余的空白字符，但保留段落结构
            clean_text = text.strip()
            pages_data.append({
                "page_number": i + 1,  # 人类习惯从第1页开始
                "content": clean_text
            })

    return pages_data

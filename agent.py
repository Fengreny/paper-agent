# agent.py
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from schema import AgentState
from tools.pdf_utils import read_pdf_with_pages
from tools.web_search import search_web

from dotenv import load_dotenv
load_dotenv()

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 或者你的 deepseek 模型


# ==========================================
# 1. Reader Node 
# ==========================================
def reader_node(state: AgentState):
    file_path = state["file_path"]

    # 读取 PDF (带页码)
    pages = read_pdf_with_pages(file_path)

    # 为了让 LLM 提取概念，我们还是需要拼一个全文，但这次只是为了提取概念
    # 真正的引用在 Writer 阶段做
    full_text_for_summary = "\n".join([p["content"] for p in pages[:5]])  # 只读前5页做摘要，节省 token，或者读全文

    # 记录日志
    logs = [f"✅ 成功读取 PDF，共 {len(pages)} 页。"]

    # 提取核心概念的 Prompt
    summary_prompt = ChatPromptTemplate.from_template(
        """
        你是一个专业的 AI 论文阅读助手。
        请阅读以下论文片段，提取出 3-5 个最关键的技术术语或核心概念（特别是那些可能需要联网搜索才能深入理解的）。

        输出格式必须是 JSON:
        {{
            "summary": "一句话概括论文主旨",
            "key_concepts": ["概念1", "概念2", "概念3"]
        }}

        论文片段:
        {text}
        """
    )

    chain = summary_prompt | llm | JsonOutputParser()
    result = chain.invoke({"text": full_text_for_summary})

    logs.append(f"🧠 提取到核心概念: {', '.join(result['key_concepts'])}")

    return {
        "pdf_pages": pages,
        "summary": result["summary"],
        "key_concepts": result["key_concepts"],
        "thought_log": logs
    }


# ==========================================
# 2. Researcher Node
# ==========================================
def researcher_node(state: AgentState):
    concepts = state["key_concepts"]
    search_results = {}
    logs = state.get("thought_log", [])

    logs.append("🌐 开始联网搜索背景知识...")

    for concept in concepts:
        # 简单搜索
        query = f"{concept} explanation machine learning"
        result = search_web(query)
        search_results[concept] = result
        logs.append(f"   -> 已搜索 '{concept}'，获取了相关资料。")

    return {
        "search_results": search_results,
        "thought_log": logs
    }


# ==========================================
# 3. Writer Node
# ==========================================


def writer_node(state: AgentState):
    pages = state["pdf_pages"]
    search_data = state["search_results"]
    summary = state["summary"]
    logs = state.get("thought_log", [])

    logs.append("✍️ 正在撰写最终报告...")

    # 构造上下文
    context_with_pages = ""
    for p in pages:
        content_preview = p['content'][:2000]
        context_with_pages += f"\n=== Page {p['page_number']} ===\n{content_preview}\n"

    writer_prompt = ChatPromptTemplate.from_template(
        """
        你是一个高级算法工程师专家。请根据提供的论文内容和联网搜索补充的知识，撰写一份深度技术报告。

        【输入素材】
        1. 论文全文（带页码标记）：
        {context}

        2. 联网搜索补充知识（用于解释复杂概念）：
        {search_data}

        【写作要求】
        1. **结构化图表**：开头包含 Mermaid 思维导图。
        2. **严格的来源区分（关键）**：
           - 凡是引用论文原文的，必须在句尾标注 `[Page X]`。
           - 凡是引用**联网搜索**补充的内容（如背景介绍、公式解释、竞品对比），必须使用引用块格式，并标注🌐图标。

           格式示例：
           DeepSeek-R1 采用了 GRPO 算法 [Page 5]。
           > 🌐 **网络补充 / 背景知识**：
           > GRPO (Group Relative Policy Optimization) 是一种不需要价值网络（Value Network）的强化学习算法，它通过...（此处写搜索到的补充内容）。

        3. **代码展示**：展示核心算法的 Python 伪代码。
        4. **深度解析**：利用搜索到的知识，解释论文中未详细展开的术语。

        【输出格式】
        Markdown 格式。
        """
    )

    chain = writer_prompt | llm | StrOutputParser()
    report = chain.invoke({
        "context": context_with_pages,
        "search_data": str(search_data)
    })

    # 追加日志
    final_report = report
    if "Agent 思考日志" not in report:
        log_str = "\n\n## 🕵️ Agent 思考日志\n" + "\n".join([f"- {log}" for log in logs])
        final_report += log_str

    return {
        "final_report": final_report,
        "thought_log": logs
    }


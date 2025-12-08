# tools/web_search.py
from langchain_community.tools import DuckDuckGoSearchRun

def search_web(query: str) -> str:
    """
    执行网络搜索，返回摘要结果。
    """
    try:
        search = DuckDuckGoSearchRun()
        # 限制一下返回长度，防止 token 爆炸
        results = search.invoke(query)
        return f"【搜索关键词】: {query}\n【搜索结果】: {results}\n"
    except Exception as e:
        return f"搜索失败: {e}"

if __name__ == "__main__":
    # 测试一下
    print(search_web("LangGraph 是什么?"))

import os
import argparse
from pathlib import Path
from langgraph.graph import StateGraph, START, END

# 导入定义的 Schema 和 Nodes
from schema import AgentState
from agent import reader_node, researcher_node, writer_node


# ==========================================
# 构建图 
# ==========================================
def build_graph():
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("reader", reader_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)

    # 定义边 (串行流程)
    workflow.add_edge(START, "reader")
    workflow.add_edge("reader", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()


# ==========================================
# 主程序 
# ==========================================
def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="AI Agent 论文阅读助手 (LangGraph版)")
    parser.add_argument(
        "--paper",
        type=str,
        default="examples/paper.pdf",  # 默认值，方便测试
        help="论文 PDF 文件路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/final_report.md",
        help="输出 markdown 文件路径",
    )
    args = parser.parse_args()

    # 2. 检查输入文件
    paper_path = Path(args.paper)
    if not paper_path.exists():
        print(f"❌ 错误: 找不到文件 {paper_path}")
        return

    # 3. 初始化并运行 Graph
    print(f"🔥 启动 Agent 工作流，正在处理: {paper_path}")
    app = build_graph()

    # 初始状态只给路径，让 Reader Node 去负责读取
    initial_state = {"file_path": str(paper_path)}

    try:
        # 运行图
        final_state = app.invoke(initial_state)

        # 4. 保存结果
        report_content = final_state.get("final_report", "生成失败，无内容。")

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_content, encoding="utf-8")

        print(f"\n✅ 任务完成！报告已保存至：{out_path}")

    except Exception as e:
        print(f"\n❌ 运行过程中发生错误: {e}")
        # 打印详细错误方便调试
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

import streamlit as st
def show_instructions():
    st.title("使用说明")
    st.markdown("""
    1. **数据导入**：在"数据概览"页面上传您的 CSV、Excel 或 JSON 文件。
    2. **数据清洗**：使用"数据清洗"页面处理缺失值、删除重复行或列，并支持交互式数据编辑。
    3. **数据分析**：在"数据分析"页面查看描述性统计和相关性分析。
    4. **数据可视化**：使用"可视化"页面创建散点图、线图、柱状图等多种图表。
    5. **高级分析**：在"高级分析"页面进行分组聚合等更深入的数据探索。
    6. **AI数据分析**：在"AI数据分析"页面调用大语言模型，对上传的数据进行智能分析并回答问题。
    7. **机器学习**：在"机器学习"页面选择目标变量和特征列，训练模型并进行预测。

    如需更多帮助，请参阅 [GitHub 仓库](https://github.com/lixinchen-031016/data-visualization-tool_for_smart_farm)。
    """)

def render_ui():
    show_instructions()
import base64
import json
from io import BytesIO

from openai import OpenAI
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.colors import n_colors
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_option_menu import option_menu

# 设置页面配置
st.set_page_config(layout="wide", page_title="数据分析工具", page_icon="📊")


client = OpenAI(
    api_key="sk-6e4e147032d54b8e8951f712b1e0b305",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取文件函数
def read_file(file):
    try:
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            data = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(file)
        elif file_extension == 'json':
            data = pd.read_json(file)
        else:
            st.error(f"不支持的文件格式：{file_extension}")
            return None
        
        # 自动检测并转换日期时间列
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_datetime(data[col])
                except:
                    pass
        
        return data
    except Exception as e:
        st.error(f"读取文件时出错：{str(e)}")
        return None

# 主函数
def main():
    # 侧边栏导航
    with st.sidebar:
        selected = option_menu(
            menu_title="主菜单",
            options=["数据概览", "数据清洗", "数据分析", "可视化", "高级分析", "AI数据分析","使用说明"],
            icons=["table", "tools", "bar-chart", "graph-up", "gear-fill","tools", "question-circle"],
            menu_icon="cast",
            default_index=0,
        )
    
    # 主内容区
    if selected == "数据概览":
        data_overview()
    elif selected == "数据清洗":
        data_cleaning()
    elif selected == "数据分析":
        data_analysis()
    elif selected == "可视化":
        data_visualization()
    elif selected == "高级分析":
        advanced_analysis()
    elif selected == "AI数据分析":
        ai_data_analysis()
    elif selected == "使用说明":
        show_instructions()

# 数据概览函数
def data_overview():
    st.title("数据概览")
    uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        data = read_file(uploaded_file)
        if data is not None:
            st.success("文件读取成功")
            st.session_state['data'] = data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", data.shape[0])
            with col2:
                st.metric("列数", data.shape[1])
            with col3:
                st.metric("缺失值数", data.isnull().sum().sum())
            
            style_metric_cards()
            
            st.subheader("数据预览")
            st.dataframe(data.head())
            
            st.subheader("数据类型")
            st.dataframe(data.dtypes)
            
            # 数据导出
            st.subheader("数据导出")
            export_format = st.radio("选择导出格式", ["CSV", "Excel"])
            if st.button("导出数据"):
                if export_format == "CSV":
                    csv = data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="exported_data.csv">下载 CSV 文件</a>'
                else:
                    towrite = BytesIO()
                    data.to_excel(towrite, index=False, engine="openpyxl")
                    towrite.seek(0)
                    b64 = base64.b64encode(towrite.read()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exported_data.xlsx">下载 Excel 文件</a>'
                st.markdown(href, unsafe_allow_html=True)

# 数据清洗函数
def data_cleaning():
    st.title("数据清洗")
    if 'data' not in st.session_state:
        st.warning("请先在数据概览页面上传数据")
        return
    
    data = st.session_state['data']
    
    st.subheader("删除重复行")
    if st.button("删除重复行"):
        original_rows = data.shape[0]
        data = data.drop_duplicates()
        st.success(f"删除了 {original_rows - data.shape[0]} 行重复数据")
    
    st.subheader("处理缺失值")
    missing_columns = data.columns[data.isnull().any()].tolist()
    for column in missing_columns:
        method = st.selectbox(f"选择处理 {column} 缺失值的方法", ["保持不变", "删除", "填充平均值", "填充中位数", "填充众数"])
        if method == "删除":
            data = data.dropna(subset=[column])
        elif method == "填充平均值":
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == "填充中位数":
            data[column].fillna(data[column].median(), inplace=True)
        elif method == "填充众数":
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    st.session_state['data'] = data
    st.success("数据清洗完成")
    
    # 添加交互式数据编辑功能
    st.subheader("交互式数据编辑")
    edited_df = st.data_editor(st.session_state['data'])
    if st.button("保存编辑"):
        st.session_state['data'] = edited_df
        st.success("数据编辑已保存")

# 数据分析函数
def data_analysis():
    st.title("数据分析")
    if 'data' not in st.session_state:
        st.warning("请先在数据概览页面上传数据")
        return
    
    data = st.session_state['data']
    
    st.subheader("描述性统计")
    st.dataframe(data.describe())
    
    st.subheader("相关性分析")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) < 2:
        st.warning("数据集中数值列不足两列，无法进行相关性分析。")
    else:
        corr_matrix = data[numeric_columns].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, labels=dict(color="相关系数"))
        fig.update_traces(text=corr_matrix.round(2), texttemplate="%{text}")
        st.plotly_chart(fig, use_container_width=True)

# 数据可视化函数
def data_visualization():
    st.title("数据可视化")
    if 'data' not in st.session_state:
        st.warning("请先在数据概览页面上传数据")
        return
    
    data = st.session_state['data']
    
    # 设置统一的主题
    pio.templates.default = "plotly_white"
    
    chart_type = st.selectbox("选择图表类型", ["散点图", "线图", "柱状图", "箱线图", "直方图", "饼图", "热力图"])
    
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    if len(numeric_columns) == 0:
        st.warning("数据集中没有数值列，无法进行可视化。")
        return
    
    # 定义现代科技感的颜色方案
    color_scheme = n_colors('rgb(0, 122, 255)', 'rgb(10, 132, 255)', 6, colortype='rgb')

    x_column = None
    y_column = None
    column = None

    if chart_type in ["散点图", "线图", "柱状图"]:
        x_column = st.selectbox("选择X轴", data.columns)
        y_column = st.selectbox("选择Y轴", numeric_columns)
        color_column = st.selectbox("选择颜色列（可选）", ["无"] + list(categorical_columns))
        
        if chart_type == "散点图":
            fig = px.scatter(data, x=x_column, y=y_column, color=color_column if color_column != "无" else None,
                             color_discrete_sequence=color_scheme)
        elif chart_type == "线图":
            fig = px.line(data, x=x_column, y=y_column, color=color_column if color_column != "无" else None,
                          color_discrete_sequence=color_scheme)
        else:  # 柱状图
            fig = px.bar(data, x=x_column, y=y_column, color=color_column if color_column != "无" else None,
                         color_discrete_sequence=color_scheme)
    
    elif chart_type in ["箱线图", "直方图"]:
        column = st.selectbox("选择列", numeric_columns)
        if chart_type == "箱线图":
            fig = px.box(data, y=column, color_discrete_sequence=color_scheme)
        else:  # 直方图
            fig = px.histogram(data, x=column, nbins=30, marginal="box", 
                               color_discrete_sequence=color_scheme)
            fig.update_traces(opacity=0.75)
            fig.update_layout(bargap=0.1)
    
    elif chart_type == "饼图":
        if len(categorical_columns) == 0:
            st.warning("数据集中没有分类列，无法创建饼图。")
            return
        column = st.selectbox("选择列", categorical_columns)
        value_counts = data[column].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'{column} 的分布',
                     color_discrete_sequence=color_scheme)
    
    elif chart_type == "热力图":
        if len(numeric_columns) < 2:
            st.warning("数据集中数值列不足两列，无法创建热力图。")
            return
        corr_matrix = data[numeric_columns].corr()
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto", 
                        color_continuous_scale='RdBu_r',  # 使用红蓝色阶
                        zmin=-1, 
                        zmax=1,
                        labels=dict(color="相关系数"))
        fig.update_traces(text=corr_matrix.round(2), texttemplate="%{text}")
        fig.update_layout(coloraxis_colorbar=dict(
            title="相关系数",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "-0.5", "0", "0.5", "1"]
        ))
    
    # 更新图表布局
    fig.update_layout(
        title={
            'text': f"{chart_type.capitalize()} - {y_column if chart_type in ['散点图', '线图', '柱状图'] else column if column else ''}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#1D3557')
        },
        xaxis_title=x_column if chart_type in ["散点图", "线图", "柱状图"] else column if column else '',
        yaxis_title=y_column if chart_type in ["散点图", "线图", "柱状图"] else "频率" if chart_type != "热力图" else '',
        legend_title="图例",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif", size=14),
        hovermode="closest",
        plot_bgcolor='rgba(240, 240, 244, 0.8)',
        paper_bgcolor='rgba(240, 240, 244, 0.8)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 122, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 122, 255, 0.1)')
    )
    
    # 创建小图用于UI展示
    fig_small = go.Figure(fig)
    fig_small.update_layout(width=700, height=500)
    st.plotly_chart(fig_small, use_container_width=True)
    
    # 创建下载链接
    fig_large = go.Figure(fig)
    fig_large.update_layout(width=1200, height=800)
    
    # 将Plotly图表转换为JSON
    fig_json = json.dumps(fig_large, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 创建下载链接
    b64 = base64.b64encode(fig_json.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="chart.json">下载图表数据 (JSON格式)</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # 添加说明
    st.markdown("""
    下载的JSON文件可以在 [Plotly Chart Studio](https://chart-studio.plotly.com/create/) 中导入以查看和编辑图表。
    或者，您可以使用Python的Plotly库来加载和显示这个JSON文件。
    """)

# 高级分析函数
def advanced_analysis():
    st.title("高级分析")
    if 'data' not in st.session_state:
        st.warning("请先在数据概览页面上传数据")
        return
    
    data = st.session_state['data']
    
    st.subheader("数据分组和聚合")
    group_column = st.selectbox("选择分组列", data.columns)
    agg_column = st.selectbox("选择聚合列", data.select_dtypes(include=['float64', 'int64']).columns)
    agg_function = st.selectbox("选择聚合函数", ["平均值", "总和", "最大值", "最小值"])
    
    # 新增验证逻辑：分组列和聚合列不能相同
    if group_column == agg_column:
        st.error("⚠️ 分组列和聚合列不能是同一列，请重新选择")
        return  # 提前终止执行
    
    agg_dict = {"平均值": "mean", "总和": "sum", "最大值": "max", "最小值": "min"}
    grouped_data = data.groupby(group_column)[agg_column].agg(agg_dict[agg_function]).reset_index()
    
    st.write("分组聚合结果：")
    st.dataframe(grouped_data)
    
    fig = px.bar(grouped_data, x=group_column, y=agg_column, title=f"{group_column} 分组的 {agg_column} {agg_function}")
    st.plotly_chart(fig, use_container_width=True)

def ai_data_analysis():
    st.title("AI数据处理")
    
    # 初始化聊天记录（如果未初始化）
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 添加选项以选择数据来源
    data_source = st.radio("选择数据来源", ["使用数据概览上传的数据", "在此功能上传新数据"], key="ai_data_source")
    
    if data_source == "使用数据概览上传的数据":
        if 'data' not in st.session_state:
            st.warning("请先在数据概览页面上传数据")
            return
        data = st.session_state['data']
        st.success("已加载数据概览页面上传的数据")
    else:
        uploaded_file = st.file_uploader("选择文件", type=["csv", "xlsx", "xls", "json"], key="ai_file_uploader")
        if uploaded_file is not None:
            data = read_file(uploaded_file)
            if data is None:
                return
            st.success("文件读取成功")
        else:
            st.warning("请上传文件以继续")
            return
    
    # 显示历史聊天记录
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])
    
    # 用户输入部分
    user_message = st.chat_input("请输入您的问题或指令...", key="ai_chat_input")
    
    if user_message:
        # 构建包含历史对话的messages
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}
        ]
        
        # 添加历史对话
        for chat in st.session_state.chat_history:
            messages.append({'role': 'user', 'content': chat["user"]})
            messages.append({'role': 'assistant', 'content': chat["assistant"]})
        
        # 添加当前用户消息和数据
        data_json = data.to_json(orient='records')
        current_message = f"{user_message}\n数据如下：\n{data_json}"
        messages.append({'role': 'user', 'content': current_message})
        
        # 调用API
        completion = client.chat.completions.create(
            model="qwen2.5-7b-instruct-1m",
            messages=messages,
        )
        
        # 解析响应
        response = completion.model_dump_json()
        response_data = json.loads(response)
        analysis = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # 添加到聊天记录
        st.session_state.chat_history.append({
            "user": user_message,
            "assistant": analysis
        })
        
        # 显示当前回复
        with st.chat_message("assistant"):
            st.write(analysis)
    
    # 导出聊天记录
    export_format = st.radio("选择导出格式", ["JSON", "Text"], key="export_format")
    if st.button("导出聊天记录"):
        if export_format == "JSON":
            content = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
            file_name = "chat_history.json"
            mime_type = "application/json"
        else:
            content = "\n".join([
                f"用户: {chat['user']}\nAI回复: {chat['assistant']}"
                for chat in st.session_state.chat_history
            ])
            file_name = "chat_history.txt"
            mime_type = "text/plain"
        
        st.download_button(
            label="下载聊天记录",
            data=content.encode("utf-8"),
            file_name=file_name,
            mime=mime_type
        )

# 使用说明函数
def show_instructions():
    st.title("使用说明")
    st.markdown("""
    1. **数据导入**：在"数据概览"页面上传您的 CSV、Excel 或 JSON 文件。
    2. **数据清洗**：使用"数据清洗"页面处理缺失值和删除重复行。
    3. **数据分析**：在"数据分析"页面查看描述性统计和相关性分析。
    4. **数据可视化**：使用"可视化"页面创建各种图表。
    5. **高级分析**：在"高级分析"页面进行更深入的数据探索。
    6. **AI数据分析**：在"AI数据分析页面"调用qwen2.5大语言模型，对上传的文件进行处理，向AI进行提问。
    
    如需更多帮助，请参阅 [GitHub 仓库](https://github.com/lixinchen-031016/data-visualization-tool_for_smart_farm)。
    """)

if __name__ == '__main__':
    main()
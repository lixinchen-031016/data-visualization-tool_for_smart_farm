import plotly.express as px
import plotly.graph_objects as go

def create_chart(data, chart_type, **params):
    """统一图表生成逻辑"""
    if chart_type == "散点图":
        return px.scatter(data, **params)
    elif chart_type == "线图":
        return px.line(data, **params)
    elif chart_type == "柱状图":
        return px.bar(data, **params)
    elif chart_type == "箱线图":
        return px.box(data, **params)
    elif chart_type == "直方图":
        return px.histogram(data, **params)
    elif chart_type == "饼图":
        return px.pie(data, **params)
    elif chart_type == "热力图":
        return px.imshow(data, **params)
    else:
        raise ValueError("不支持的图表类型")
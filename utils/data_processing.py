import base64
from io import BytesIO


def export_data(data, format_type):
    """统一处理数据导出逻辑"""
    if format_type == "CSV":
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="exported_data.csv">下载 CSV 文件</a>'
    elif format_type == "Excel":
        towrite = BytesIO()
        data.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exported_data.xlsx">下载 Excel 文件</a>'
    elif format_type == "JSON":
        json_data = data.to_json(orient='records', force_ascii=False).encode()
        b64 = base64.b64encode(json_data).decode()
        return f'<a href="data:application/json;base64,{b64}" download="exported_data.json">下载 JSON 文件</a>'
    else:
        raise ValueError("不支持的导出格式")
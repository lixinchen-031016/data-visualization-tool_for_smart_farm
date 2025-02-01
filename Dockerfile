# 使用官方的Python基础镜像
FROM python:3.11

# 设置工作目录
WORKDIR /app

# 复制当前目录下的所有文件到工作目录
COPY . /app

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 暴露端口
EXPOSE 8501

# 启动Streamlit应用
CMD ["streamlit", "run", "app.py"]
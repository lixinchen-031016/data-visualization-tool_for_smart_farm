import streamlit as st
import psutil
import time
def show_performance():
    st.subheader("系统监控")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("内存使用", f"{psutil.virtual_memory().percent}%")
    with col2:
        st.metric("CPU负载", f"{psutil.cpu_percent()}%")
    with col3:
        st.metric("处理时间", f"{time.process_time():.2f}s")

def render_ui():
    show_performance()
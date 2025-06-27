import streamlit as st
import psutil
import time

def show_performance():
    st.subheader("系统监控")
    col1, col2, col3 = st.columns(3)
    
    # 使用科技感卡片样式
    with col1:
        st.markdown("""
        <div class="system-monitor-card">
            <h3>内存使用</h3>
            <div class="value">{:.1f}%</div>
        </div>
        """.format(psutil.virtual_memory().percent), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="system-monitor-card">
            <h3>CPU负载</h3>
            <div class="value">{:.1f}%</div>
        </div>
        """.format(psutil.cpu_percent()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="system-monitor-card">
            <h3>处理时间</h3>
            <div class="value">{:.2f}s</div>
        </div>
        """.format(time.process_time()), unsafe_allow_html=True)

def render_ui():
    show_performance()
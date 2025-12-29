#  C:\Users\孙冰\Desktop\AI助教25-12-07
#  streamlit run C:\Users\孙冰\Desktop\AI助教25-12-07\sklearn_cover.py

import streamlit as st
import numpy as np

# 第一步：导入所有必需的模块（解决 subprocess 未定义问题）
import streamlit as st
import subprocess  # 关键：必须导入这个模块
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 第二步：定义字体安装函数（修正导入问题 + 优化逻辑）
def install_and_setup_chinese_fonts():
    try:
        # 1. 检查 fonts-noto-cjk 包是否已安装
        result = subprocess.run(
            ["dpkg", "-l", "fonts-noto-cjk"],
            capture_output=True,
            text=True
        )
        
        # 2. 未安装则自动安装（Streamlit Cloud 无需 sudo）
        if "ii  fonts-noto-cjk" not in result.stdout:
            with st.spinner("正在安装思源黑体（Noto CJK）字体包..."):
                # 更新软件源（静默执行，不输出冗余信息）
                subprocess.run(
                    ["apt", "update"],
                    capture_output=True,
                    text=True
                )
                # 安装字体包（-y 自动确认）
                install_result = subprocess.run(
                    ["apt", "install", "-y", "fonts-noto-cjk"],
                    capture_output=True,
                    text=True
                )
                # 检查安装是否成功
                if install_result.returncode == 0:
                    # 刷新字体缓存
                    subprocess.run(["fc-cache", "-fv"], capture_output=True)
                    st.success("✅ 思源黑体安装成功！")
                else:
                    st.warning(f"⚠️ 字体安装失败：{install_result.stderr}")
        else:
            st.success("✅ 思源黑体已预装，无需重复安装")
        
        # 3. 配置 Matplotlib 使用思源黑体
        fm._rebuild()  # 强制刷新字体缓存
        # 优先使用安装的思源黑体
        plt.rcParams['font.family'] = 'Noto Sans CJK SC'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 4. 配置 Plotly 使用思源黑体
        px.defaults.font = dict(
            family="Noto Sans CJK SC, sans-serif",
            size=12,
            color="#333333"
        )
        
    except Exception as e:
        st.warning(f"⚠️ 字体配置异常（不影响核心功能）：{str(e)}")
        # 兜底方案：使用系统备选字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
        plt.rcParams['axes.unicode_minus'] = False

# 第三步：调用函数（部署时自动执行）
install_and_setup_chinese_fonts()


# ---------------------- 测试中文显示（可选）----------------------
st.subheader("中文显示测试")
# Matplotlib 测试
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("测试标题：线性回归分析（思源黑体）", fontsize=14)
ax.plot([1,2,3], [4,5,6], label="测试数据：中文标签")
ax.legend()
st.pyplot(fig)




# 设置页面配置
st.set_page_config(
    page_title="明德智学 - 机器学习",
    page_icon="📊",
    layout="wide"
)

# 自定义CSS样式 - 增强按钮（移除颜色相关设置）
st.markdown("""
<style>
    /* 顶部彩色横条 */
    .top-bar {
        height: 8px;
        background: linear-gradient(90deg, #3498db, #9b59b6, #e74c3c, #f39c12);
        border-radius: 4px;
        margin-bottom: 0rem;
    }
    
    /* 分隔彩色横条 */
    .divider-bar {
        height: 4px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        border-radius: 2px;
        margin: 0rem 0;
    }
    
    .main-title {
        font-size: 3.2rem;
        color: #2c3e50;
        text-align: center;
        margin: 3rem 0;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-title {
        font-size: 1.8rem;
        color: #3498db;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
    
    .content-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .intro-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    /* 按钮容器调整 - 上移并居中 */
    .button-container {
        margin: 0rem auto 1rem;  /* 减小顶部margin实现上移，auto实现水平居中 */
        width: 80%;  /* 限制容器宽度，增强居中效果 */
        text-align: center;  /* 文本居中 */
    }
    
    .footer {
        text-align: center;
        color: #7f8c8d;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eaeaea;
    }
    
    /* 按钮样式 - 加宽、加大字体（移除颜色相关设置） */
    .stButton > button {
        font-size: 2rem;  /* 加大字体 */
        padding: 1rem;
        border-radius: 12px;
        font-weight: 800;
        transition: all 0.3s ease;
        height: auto;
        width: 100%;  /* 适当加宽 */
        margin: 0 -10%;  /* 调整外边距配合居中 */
        border-width: 3px;  /* 加粗边框 */
        border-style: solid;  /* 确保边框样式生效 */
    }
    
    /* 按钮悬停效果（移除颜色相关设置） */
    .stButton > button:hover {
        transform: translateY(-3px);
    }
</style>
""", unsafe_allow_html=True)

# 顶部彩色横条
st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)

# 页面标题
st.markdown('<h1 class="main-title">明德智学交互学习平台——Python之机器学习</h1>', unsafe_allow_html=True)

# 分隔彩色横条
st.markdown('<div class="divider-bar"></div>', unsafe_allow_html=True)

# 添加介绍内容
with st.container():
    st.markdown('<h2 class="section-title">机器学习简介</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="content-text">
        机器学习是人工智能的一个重要分支，它使计算机系统能够通过数据学习并改进，而无需显式编程。
        它主要分为监督学习、无监督学习和强化学习三大类，广泛应用于预测分析、模式识别、数据挖掘等领域。
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-title">scikit-learn库</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="content-text">
        scikit-learn（简称sklearn）是Python中最流行的机器学习库之一，它提供了简单高效的工具用于数据挖掘和数据分析。
        该库建立在NumPy、SciPy和matplotlib之上，包含了多种分类、回归和聚类算法，如随机森林、梯度提升、支持向量机等，
        同时提供了数据预处理、模型评估等配套功能，非常适合机器学习初学者和专业人士使用。
    </p>
    """, unsafe_allow_html=True)

# 分隔彩色横条
st.markdown('<div class="divider-bar"></div>', unsafe_allow_html=True)

# 创建四个主要按钮，使用columns来排列
st.markdown('<div class="button-container">', unsafe_allow_html=True)
col1, col2, col3, col4, col5= st.columns(5)

with col1:
    # 线性回归按钮
    if st.button("**线性奥秘**：从数据到趋势的映射"):
        # 清除所有session_state状态
        st.session_state.clear()
        st.session_state.page = "linear_regression_demo"


with col2:
    # 逻辑回归按钮
    if st.button("**分类智慧**：二值世界的概率解码"):
        # 清除所有session_state状态
        st.session_state.clear()
        st.session_state.page = "logistic_regression_demo"


with col3:
    # KMeans聚类按钮
    if st.button("**聚光灯下**：数据自然分组的探索"):
        # 清除所有session_state状态
        st.session_state.clear()
        st.session_state.page = "kMeans_demo"


with col4:
    # 神经网络按钮
    if st.button("**神经元的魔法**：多层感知的力量"):
        # 清除所有session_state状态
        st.session_state.clear()
        st.session_state.page = "neural_network_demo"
        
with col5:
    # 文本分析按钮
    if st.button("**文字的密码**：情感与主题的挖掘"):
        # 清除所有session_state状态
        st.session_state.clear()
        st.session_state.page = "text_analysis_demo"


st.markdown('</div>', unsafe_allow_html=True)

# 页脚信息
st.markdown("""
<p class="footer">
    明德智学项目 ©   2025   孙冰   |   探索Python机器学习的世界
</p>
""", unsafe_allow_html=True)

# 在文件末尾添加页面跳转逻辑
if 'page' in st.session_state and st.session_state.page == "linear_regression_demo":
    # 导入并运行线性回归演示页面
    import linear_regression_demo
    # 如果linear_regression_demo.py中有主函数main()，可以这样调用
    linear_regression_demo.main()
    # 否则直接导入会执行该文件中的代码
    
if 'page' in st.session_state and st.session_state.page == "logistic_regression_demo":
    import logistic_regression_demo
    logistic_regression_demo.main()

if 'page' in st.session_state and st.session_state.page == "kMeans_demo":
    import kMeans_demo
    kMeans_demo.main()

if 'page' in st.session_state and st.session_state.page == "neural_network_demo":
    import neural_network_demo
    neural_network_demo.main()

if 'page' in st.session_state and st.session_state.page == "text_analysis_demo":
    import text_analysis_demo
    text_analysis_demo.main()

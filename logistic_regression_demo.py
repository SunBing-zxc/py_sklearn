# streamlit run logistic_regression_demo.py
# C:\Users\孙冰\Desktop\AI助教25-12-07

# logistic_regression_demo.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import io
from api_deepseek import client, ask_ai_assistant
import logistic_regression_step_by_step
from learning_report import generate_evaluation

# 设置页面
st.set_page_config(page_title="逻辑回归交互式学习平台", layout="wide")
st.title("📚 逻辑回归交互式学习平台")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def display_chat_interface(context=""):
    """显示聊天界面"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 AI助教已就绪")
    
    # 预设问题快捷按钮
    st.sidebar.markdown("**快捷问题:**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        btn1 = st.button("什么是sigmoid函数?")
        btn2 = st.button("逻辑回归与线性回归的区别")
    
    with col2:
        btn3 = st.button("分类阈值如何选择")
        btn4 = st.button("交叉熵损失原理")
    
    # 处理快捷问题
    question = ""
    if btn1:
        question = "什么是sigmoid函数?它在逻辑回归中的作用是什么?"
    elif btn2:
        question = "逻辑回归与线性回归有什么主要区别?分别适用于什么场景?"
    elif btn3:
        question = "逻辑回归中分类阈值(阈值)如何选择?不同阈值有什么影响?"
    elif btn4:
        question = "请解释交叉熵损失函数的原理，为什么逻辑回归不用均方误差?"
    
    # 提问输入框
    user_input = st.sidebar.text_input("输入你的问题:", key="question_input")
    if user_input:
        question = user_input
    
    # 处理提问
    if question:

        # 记录AI交互（新增：用于评价分析）
        if "ai_interactions" not in st.session_state.logistic_records:
            st.session_state.logistic_records["ai_interactions"] = []
        st.session_state.logistic_records["ai_interactions"].append({
            "question": question,
            "timestamp": datetime.now().timestamp()
        })

        # 显示当前问题
        st.sidebar.markdown(f"**你:** {question}")
        
        # 获取回答
        with st.spinner("助教思考中..."):
            answer = ask_ai_assistant(question, context)
        
        # 显示当前回答
        st.sidebar.markdown(f"**助教:** {answer}")
        st.sidebar.markdown("---")

# Sigmoid函数定义与可视化
def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def plot_sigmoid_function(z_value):
    """绘制sigmoid函数图像"""
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2)
    ax.grid(True, linestyle='-', alpha=0.3)  # 实线网格，适当降低透明度
    ax.set_axisbelow(True)  # 网格显示在曲线下方
    ax.axhline(y=0.5, color='r', linestyle='--', label='阈值=0.5')
    ax.axvline(x=0, color='g', linestyle=':', label='x=0')
    # 标注当前z值与函数的交点
    z_prob = sigmoid(z_value)  # 计算当前z值对应的概率
    ax.plot(z_value, z_prob, 'ro', markersize=10)  # 绘制交点（黑色圆点）
    
    # 添加交点的坐标标注
    ax.annotate(
        f'z={z_value:.1f}\n概率={z_prob:.4f}',  # 标注文本
        xy=(z_value, z_prob),  # 标注点坐标
        xytext=(10, 10),  # 文本位置（相对于标注点的偏移）
        textcoords='offset points',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # 添加竖线和水平线连接到坐标轴
    ax.axvline(x=z_value, color='orange', linestyle='-', alpha=0.7)
    ax.axhline(y=z_prob, color='orange', linestyle='-', alpha=0.7)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('z值')
    ax.set_ylabel('sigmoid(z) 概率值')
    ax.set_title('Sigmoid函数与当前z值位置')
    ax.legend()
    return fig

# 数据生成函数（二分类数据）
@st.cache_data
def generate_classification_data(data_type, n_samples, separation):
    """生成分类数据"""
    np.random.seed(42)
    
    if data_type == "线性可分":
        # 生成两个线性可分的类别
        n_class1 = n_samples // 2
        n_class2 = n_samples - n_class1  
        X1 = np.random.randn(n_class1, 2) * 0.8 + np.array([separation, separation])
        X2 = np.random.randn(n_class2, 2) * 0.8 - np.array([separation, separation])
        X = np.vstack((X1, X2))
        y = np.hstack((np.zeros(n_class1), np.ones(n_class2)))
    
    elif data_type == "线性不可分":
        # 生成线性不可分的数据
        X = np.random.randn(n_samples, 2) * 1.2
        # 基于二次函数生成标签，制造非线性边界
        y = (X[:, 0]**2 + X[:, 1]** 2 < 1.5).astype(int)
    
    elif data_type == "不平衡数据":
        # 生成不平衡数据
        n_minority = int(n_samples * 0.2)  # 20%为少数类
        n_majority = n_samples - n_minority  # 80%为多数类（动态调整）
        X_majority = np.random.randn(n_majority, 2) * 0.8 - np.array([separation/2, separation/2])
        X_minority = np.random.randn(n_minority, 2) * 0.8 + np.array([separation/2, separation/2])
        X = np.vstack((X_majority, X_minority))
        y = np.hstack((np.zeros(n_majority), np.ones(n_minority)))
    
    # 打乱数据顺序
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

# 绘制分类数据
def plot_classification_data(X, y, title):
    """绘制分类数据散点图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, label='类别 1')
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 逻辑回归梯度下降模拟
def logistic_regression_gradient_descent(X, y, learning_rate, n_iterations):
    """手动实现逻辑回归的梯度下降"""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    costs = []
    
    for _ in range(n_iterations):
        # 计算线性输出
        linear_model = np.dot(X, weights) + bias
        # 应用sigmoid函数
        y_pred = sigmoid(linear_model)
        
        # 计算交叉熵损失
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        costs.append(cost)
        
        # 计算梯度
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        
        # 更新参数
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias, costs

# 绘制决策边界
def plot_decision_boundary(X, y, weights, bias, threshold=0.5, title="决策边界"):
    """绘制逻辑回归的决策边界"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点
    ax.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, label='类别 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, label='类别 1')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
    Z = (Z >= threshold).astype(int)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

# 绘制sigmoid曲线与分类阈值
def plot_sigmoid_threshold():
    """展示sigmoid函数与不同阈值的关系"""
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='sigmoid函数')
    
    # 绘制不同阈值线
    thresholds = [0.3, 0.5, 0.7]
    colors = ['g', 'r', 'purple']
    for threshold, color in zip(thresholds, colors):
        # 找到对应阈值的x值
        x_threshold = np.log(threshold / (1 - threshold))
        ax.axhline(y=threshold, color=color, linestyle='--', 
                  label=f'阈值={threshold} (x={x_threshold:.2f})')
        ax.axvline(x=x_threshold, color=color, linestyle=':')
    
    ax.set_xlabel('线性输出 (z = wx + b)')
    ax.set_ylabel('概率 p(y=1)')
    ax.set_title('Sigmoid函数与不同分类阈值')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    return fig

# 数据生成与探索模块
def data_generation_section():
    st.header("📊 分类数据生成与探索")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("选择数据类型", 
                               ["线性可分", "线性不可分", "不平衡数据"])
        n_samples = st.slider("样本数量", 50, 500, 200)
        separation = st.slider("类别分离程度", 0.5, 5.0, 2.0, 0.5)
        
        X, y = generate_classification_data(data_type, n_samples, separation)
        
        st.write(f"数据统计:")
        st.write(f"- 类别0数量: {np.sum(y == 0)}")
        st.write(f"- 类别1数量: {np.sum(y == 1)}")
        st.write(f"- 特征1均值: {np.mean(X[:, 0]):.2f}, 标准差: {np.std(X[:, 0]):.2f}")
        st.write(f"- 特征2均值: {np.mean(X[:, 1]):.2f}, 标准差: {np.std(X[:, 1]):.2f}")

        # 记录操作（新增：用于评价分析）
        st.session_state.logistic_records["data_generation"].append({
            "data_type": data_type,
            "n_samples": n_samples,
            "separation": separation,
            "timestamp": datetime.now().timestamp()
        })
    
    with col2:
        fig = plot_classification_data(X, y, f'{data_type}数据分布')
        st.pyplot(fig)
    
    st.info("""
    **分类数据探索要点:**
    - 线性可分数据: 可以用一条直线完美分隔两个类别
    - 线性不可分数据: 无法用一条直线完美分隔
    - 不平衡数据: 一个类别的样本数量远多于另一个类别
    - 类别分离程度影响分类难度，分离越好越容易分类
    """)
    
    # 存储数据供后续模块使用
    #st.session_state.X = X
    #st.session_state.y = y
    
    return f"数据生成模块: 创建了{data_type}数据，样本数={n_samples}，分离程度={separation}"

# Sigmoid函数交互模块
def sigmoid_interactive_section():
    st.header("🔄 Sigmoid函数交互演示")
    st.markdown("""
        **Sigmoid函数公式：**
        $$\sigma(z) = \\frac{1}{1 + e^{-z}}$$  ，其中 $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$ 是线性组合
        
        **Sigmoid函数的特点：输出值范围在(0, 1)之间，可解释为概率；函数光滑且可导，适合梯度下降优化**
        - 当z→+∞时，σ(z)→1
        - 当z→-∞时，σ(z)→0
        - 当z=0时，σ(z)=0.5
        """)    
    col1, col2 = st.columns([2,3])
    
    with col1:        
        z_value = st.slider("选择z值", -10.0, 10.0, 0.0, 0.1)
        sigmoid_value = sigmoid(z_value)
        st.metric("sigmoid(z)值", f"{sigmoid_value:.4f}")
        
        if sigmoid_value >= 0.5:
            st.success(f"当z={z_value:.1f}时，预测为类别1（概率={sigmoid_value:.4f}）")
        else:
            st.info(f"当z={z_value:.1f}时，预测为类别0（概率={1-sigmoid_value:.4f}）")

        # 记录操作（新增：用于评价分析）
        st.session_state.logistic_records["sig_function"].append({
            "z_value": z_value,
            "timestamp": datetime.now().timestamp()
        })
    
    with col2:
        # 绘制sigmoid函数
        fig1 = plot_sigmoid_function(z_value)
        st.pyplot(fig1)
 
    return f"Sigmoid函数模块: 探索了z={z_value:.1f}时的函数值"

# 手动调整参数模块（学生考试场景优化版）
def manual_tuning_section():
    st.header("🎛️ 逻辑回归参数手动调整（学生考试场景）")
    st.info("基于「考试成绩」和「缺勤次数」预测是否通过期末考试")
    
    # 生成学生数据（两个特征：成绩[0-100]、缺勤次数[0-15]）
    np.random.seed(42)
    n_samples = 200
    
    # 生成通过和未通过的学生数据
    pass_scores = np.random.normal(75, 10, n_samples//2)
    pass_absences = np.random.normal(2, 1, n_samples//2)
    pass_absences = np.clip(pass_absences, 0, 15)
    
    fail_scores = np.random.normal(40, 15, n_samples//2)
    fail_absences = np.random.normal(10, 3, n_samples//2)
    fail_absences = np.clip(fail_absences, 0, 15)
    
    # 合并数据
    X = np.vstack([
        np.column_stack((pass_scores, pass_absences)),
        np.column_stack((fail_scores, fail_absences))
    ])
    y = np.hstack([np.ones(n_samples//2), np.zeros(n_samples//2)])
    
    # 打乱顺序
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    # 特征分离
    scores = X[:, 0]
    absences = X[:, 1]
    st.subheader("调整模型参数")    
    col1, col2 = st.columns([2,3])    
    with col1:        
        # 成绩权重（正向特征，扩大正向范围）
        score_weight = st.slider(
            "成绩权重 (w1)", 
            -1.0,  # 最小负向值（缩小负向范围）
            1.0,   # 最大正向值（扩大正向范围）
            0.4,   # 默认值
            0.05   # 步长
        )
        # 成绩权重解释（滑块下方即时说明）
        st.write(f"成绩应为正向权重，即成绩越高，通过概率越大")
        
        # 缺勤权重（负向特征，扩大负向范围）
        absence_weight = st.slider(
            "缺勤权重 (w2)", 
            -5.0,  # 最大负向值（扩大负向范围）
            1.0,   # 最小正向值（缩小正向范围）
            -3.0,  # 默认值
            0.05   # 步长
        )
        # 缺勤权重解释（滑块下方即时说明）
        st.write(f"缺勤应为负向权重，即缺勤越多，通过概率越低")
        
        # 偏置项
        bias = st.slider("偏置 (b)", -10.0, 10.0, -5.0, 0.5)
        # 偏置解释（滑块下方即时说明）
        st.write(f"设置偏置可以整体提高或降低通过概率基准线")
        
        # 分类阈值
        threshold = st.slider("通过概率阈值", 0.1, 0.9, 0.5, 0.05)
        # 阈值解释（滑块下方即时说明）
        if threshold > 0.5:
            st.write(f"当前阈值：{threshold:.2f} → 判定通过的标准更严格（减少误判通过）")
        elif threshold < 0.5:
            st.write(f"当前阈值：{threshold:.2f} → 判定通过的标准更宽松（减少误判不通过）")
        else:
            st.write(f"当前阈值：0.5 → 中立判定标准")
            
        # 计算预测结果
        z = score_weight * scores + absence_weight * absences + bias
        y_prob = sigmoid(z)
        y_pred = (y_prob >= threshold).astype(int)
        
        # 模型评估
        accuracy = accuracy_score(y, y_pred)
        st.metric("预测准确率", f"{accuracy:.4f}")
        
    with col2:
        # 绘制可视化图表（保持原有逻辑）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 成绩对通过概率的影响
        fixed_absence = np.mean(absences)
        score_range = np.linspace(0, 100, 200)
        z_scores = score_weight * score_range + absence_weight * fixed_absence + bias
        prob_scores = sigmoid(z_scores)
        
        ax1.plot(score_range, prob_scores, 'b-', label=f'固定缺勤={fixed_absence:.1f}次')
        ax1.axhline(threshold, color='r', linestyle='--', label=f'通过阈值={threshold}')
        ax1.scatter(scores[y==1], np.ones_like(scores[y==1]), c='green', alpha=0.5, label='实际通过')
        ax1.scatter(scores[y==0], np.zeros_like(scores[y==0]), c='red', alpha=0.5, label='实际挂科')
        ax1.set_xlabel('模拟考试成绩')
        ax1.set_ylabel('通过概率')
        ax1.set_title('成绩对通过概率的影响（固定缺勤次数）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 双特征决策边界
        x_min, x_max = scores.min() - 10, scores.max() + 10  # 增大范围
        y_min, y_max = absences.min() - 3, absences.max() + 3  # 增大范围
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                             np.arange(y_min, y_max, 0.2))
        
        Z = sigmoid(score_weight * xx + absence_weight * yy + bias)
        Z_class = (Z >= threshold).astype(int)
        
        ax2.contourf(xx, yy, Z_class, alpha=0.3, cmap=plt.cm.coolwarm)
        ax2.scatter(scores[y==1], absences[y==1], c='green', label='实际通过', alpha=0.7)
        ax2.scatter(scores[y==0], absences[y==0], c='red', label='实际挂科', alpha=0.7)
        
        if score_weight != 0:
            absence_line = np.linspace(y_min, y_max, 100)
            score_line = (np.log(threshold/(1-threshold)) - absence_weight * absence_line - bias) / score_weight
            valid = (score_line >= x_min) & (score_line <= x_max)
            ax2.plot(score_line[valid], absence_line[valid], 'k-', linewidth=2, label=f'决策边界（概率={threshold}）')
        
        ax2.set_xlabel('模拟考试成绩')
        ax2.set_ylabel('缺勤次数')
        ax2.set_title('双特征决策边界（绿色区域=预测通过）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

    col1, col2 = st.columns([2,3])    
    with col1:        
        # 显示混淆矩阵
        st.subheader("混淆矩阵")
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=['实际挂科', '实际通过'], columns=['预测挂科', '预测通过'])
        st.dataframe(cm_df)
    with col2:
        st.subheader('混淆矩阵理解')
        st.write("""
            - **误判（右上角）**：实际挂科却被预测为通过（假阳性）
            - **漏判（左下角）**：实际通过却被预测为挂科（假阴性）
            - 左上角：实际挂科且预测挂科（真阴性）
            - 右下角：实际通过且预测通过（真阳性）
            """)      
        
    st.info("""
    **参数调整指南:**
    - 尝试将成绩权重保持为正值，缺勤权重保持为负值（符合实际逻辑）
    - 调整权重大小可以改变对应特征对结果的影响强度
    - 偏置项可以整体抬高或降低通过概率的基准线
    """)


    # 记录操作
    st.session_state.logistic_records["para_tuning"].append({
        "score_weight": score_weight,
        "absence_weight": absence_weight,
        "bias": bias,
        "threshold": threshold,
        "accuracy": accuracy,
        "timestamp": datetime.now().timestamp()
    })     
    return f"手动调整模块: 成绩权重={score_weight:.1f}, 缺勤权重={absence_weight:.1f}, 偏置={bias:.1f}, 阈值={threshold:.2f}, 准确率={accuracy:.4f}"


# 梯度下降可视化模块
def gradient_descent_section():
    st.header("📉 逻辑回归梯度下降可视化")
    X,y = generate_classification_data("线性可分", 300, 0.6) 

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    col1, col2 = st.columns([2,3])
    
    with col1:
        learning_rate = st.slider("学习率", 0.1, 50.0, 1.0)
        n_iterations = st.slider("迭代次数", 0, 15, 8)
        st.markdown("""
        **学习率选择建议:**
        - 太小: 收敛速度慢，需要更多迭代
        - 太大: 可能导致不收敛，损失波动甚至增大
        """)
    with col2:
        st.markdown("""
        **逻辑回归梯度下降原理:**
        
        1. **初始化**权重和偏置为0
        2. 计算**线性输出** $z = wx + b$
        3. 应用**sigmoid函数**得到概率预测 $\\hat{y} = \\sigma(z)$
        4. 计算**交叉熵损失**:
           $$L = -\\frac{1}{n}\\sum(y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y}))$$
        5. 计算损失对权重和偏置的**梯度**
        6. 沿梯度反方向**更新参数**:
           $$w = w - \\alpha \\cdot \\frac{\\partial L}{\\partial w}$$
           $$b = b - \\alpha \\cdot \\frac{\\partial L}{\\partial b}$$
        7. 重复步骤2-6直到收敛
        """)        
    if st.button("开始梯度下降演示"):
        # 运行梯度下降
        weights, bias, costs = logistic_regression_gradient_descent(
            X_scaled, y, learning_rate, n_iterations
        )
            
        # 显示过程
        placeholder = st.empty()
        # 只显示部分迭代步骤，避免太慢
        step = max(1, n_iterations // 20)
        for i in range(0, n_iterations + 1, step):
            with placeholder.container():
                # 计算当前迭代的参数（如果超出范围则用最后一组）
                current_weights = weights if i == n_iterations else \
                                    logistic_regression_gradient_descent(
                                        X_scaled, y, learning_rate, i)[0]
                current_bias = bias if i == n_iterations else \
                                logistic_regression_gradient_descent(
                                    X_scaled, y, learning_rate, i)[1]
                    
                # 绘制决策边界和损失曲线
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                # 决策边界
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                                        np.arange(y_min, y_max, 0.01))
                    
                # 标准化网格点
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                grid_points_scaled = scaler.transform(grid_points)
                    
                Z = sigmoid(np.dot(grid_points_scaled, current_weights) + current_bias)
                Z = (Z >= 0.5).astype(int)
                Z = Z.reshape(xx.shape)
                    
                ax1.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
                ax1.scatter(X[y==0, 0], X[y==0, 1], alpha=0.7, label='类别 0')
                ax1.scatter(X[y==1, 0], X[y==1, 1], alpha=0.7, label='类别 1')
                ax1.set_title(f'迭代 {i}/{n_iterations}')
                ax1.legend()
                    
                # 损失曲线
                ax2.plot(range(min(i+1, len(costs))), costs[:min(i+1, len(costs))])
                ax2.set_xlabel('迭代次数')
                ax2.set_ylabel('交叉熵损失')
                ax2.set_title(f'损失: {costs[min(i, len(costs)-1)]:.4f}')
                ax2.grid(True)
                    
                plt.tight_layout()
                st.pyplot(fig)
                time.sleep(0.05)
            
        st.success(f"梯度下降完成! 最终损失: {costs[-1]:.4f}")
        # 记录操作（新增：用于评价分析）
        st.session_state.logistic_records["gradient_descent"].append({
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "final_cost": costs[-1],
            "timestamp": datetime.now().timestamp()
        })
    
   
    
    return f"梯度下降模块: 学习率={learning_rate}, 迭代次数={n_iterations}"

# 模型评估模块（专注于混淆矩阵和评估指标解释）
def model_evaluation_section():
    st.header("📊 模型评估与指标解释")
    
    # 选择解释场景
    st.subheader("选择一个场景帮助理解混淆矩阵：")
    scenario = st.selectbox(
        "场景示例：",
        ["疾病检测", "垃圾邮件过滤"]
    )
    
    # 根据场景生成模拟的混淆矩阵数据及术语
    if scenario == "疾病检测":
        # 疾病检测场景的模拟数据（TN, FP, FN, TP）
        tn, fp, fn, tp = 85, 5, 3, 7
        classes = ['健康', '患病']
        terms = {
            "tn": "健康人被正确判断为健康（TN）",
            "fp": "健康人被错误判断为患病（FP）",
            "fn": "患病者被错误判断为健康（FN）",
            "tp": "患病者被正确判断为患病（TP）",
            "title": "疾病检测场景下的混淆矩阵"
        }
        # 错误后果
        fp_consequence = "健康人接受不必要的治疗，造成经济损失和心理负担"
        fn_consequence = "真正的患者错过治疗时机，导致病情恶化甚至危及生命"
        # 指标示例
        precision_example = "预测为患病的人里，真正患病的比例"
        recall_example = "所有真正患病的人里，被检测出来的比例"
    else:  # 垃圾邮件过滤场景
        # 垃圾邮件过滤场景的模拟数据（TN, FP, FN, TP）
        tn, fp, fn, tp = 90, 2, 5, 3
        classes = ['正常邮件', '垃圾邮件']
        terms = {
            "tn": "正常邮件被正确判断为正常（TN）",
            "fp": "正常邮件被错误判断为垃圾邮件（FP）",
            "fn": "垃圾邮件被错误判断为正常邮件（FN）",
            "tp": "垃圾邮件被正确判断为垃圾邮件（TP）",
            "title": "垃圾邮件过滤场景下的混淆矩阵"
        }
        # 错误后果
        fp_consequence = "重要邮件被误删，可能错过关键信息（如工作邮件、通知）"
        fn_consequence = "垃圾邮件充斥邮箱，干扰用户正常使用，甚至包含诈骗信息"
        # 指标示例
        precision_example = "被标记为垃圾邮件的邮件中，真正是垃圾邮件的比例"
        recall_example = "所有实际是垃圾邮件的邮件中，被正确识别的比例"
    
    # 计算评估指标
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 显示混淆矩阵表格（带TP/TN/FP/FN标注）
    st.subheader(terms["title"])
    cm_data = {
        f"预测为{classes[0]}": [f"TN: {tn}", f"FN: {fn}"],
        f"预测为{classes[1]}": [f"FP: {fp}", f"TP: {tp}"]
    }
    cm_df = pd.DataFrame(cm_data, index=[f"实际为{classes[0]}", f"实际为{classes[1]}"])
    st.dataframe(cm_df, use_container_width=True)
    
    # 显示数值解释（强化术语对应关系）
    st.markdown(f"""
    **混淆矩阵核心术语解释：**
    - **TN（真阴性）**：{terms['tn'].split('（')[0]}，共{tn}例
    - **FP（假阳性）**：{terms['fp'].split('（')[0]}，共{fp}例
    - **FN（假阴性）**：{terms['fn'].split('（')[0]}，共{fn}例
    - **TP（真阳性）**：{terms['tp'].split('（')[0]}，共{tp}例
    """)
    
    # 混淆矩阵场景化解读
    st.subheader("混淆矩阵实战解读")
    st.markdown(f"""
    在**{scenario}** 场景中，四个指标的业务含义：
    
    | 真实情况 \\ 预测结果 | 预测为{classes[0]} | 预测为{classes[1]} |
    |-------------------|----------------|----------------|
    | **实际为{classes[0]}** | TN（正确） | FP（错误） |
    | **实际为{classes[1]}** | FN（错误） | TP（正确） |
    
    **关键错误影响分析：**
    - FP（{terms['fp'].split('（')[1][:-1]}）：{fp_consequence}
    - FN（{terms['fn'].split('（')[1][:-1]}）：{fn_consequence}
    
    **场景化指标选择策略：**
    - 当FN代价更高（如疾病检测）：优先保证**召回率**（减少漏诊）
    - 当FP代价更高（如垃圾邮件过滤）：优先保证**精确率**（减少误删）
    """)
        
    # 评估指标计算与解释
    st.subheader("核心指标计算与业务意义")
    st.markdown(f"""
    - **准确率 (Accuracy)**：{accuracy:.4f}  
      → 计算公式：(TP + TN) / 总样本数 = ({tp} + {tn}) / {total}  
      → 含义：所有判断中正确的比例
    
    - **精确率 (Precision)**：{precision:.4f}  
      → 计算公式：TP / (TP + FP) = {tp} / ({tp} + {fp})  
      → 含义：{precision_example}
    
    - **召回率 (Recall)**：{recall:.4f}  
      → 计算公式：TP / (TP + FN) = {tp} / ({tp} + {fn})  
      → 含义：{recall_example}
    
    - **F1分数**：{f1:.4f}  
      → 计算公式：2 × (精确率 × 召回率) / (精确率 + 召回率)  
      → 含义：平衡精确率和召回率的综合指标
    """)

    # 记录操作（新增：用于评价分析）
    st.session_state.logistic_records["model_evaluation"].append({
        "scenario": scenario,
        "timestamp": datetime.now().timestamp()
    })
    
    return f"模型评估模块: 准确率={accuracy:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}"

# 概念测验模块
def quiz_section():
    st.header("🎯 概念测验")
    st.write("请完成以下5道单选题，全部答完后可提交查看结果")
    
    # 定义测验题目、选项、正确答案及解析
    quiz_data = [
        {
            "question": "1. 逻辑回归的输出是什么?",
            "options": [
                "A. 连续的预测值",
                "B. 0或1的分类结果",
                "C. 属于某个类别的概率"
            ],
            "correct": "C",
            "explanation": "逻辑回归输出的是样本属于正类的概率，范围在0到1之间。"
        },
        {
            "question": "2. sigmoid函数的作用是什么?",
            "options": [
                "A. 增加模型复杂度",
                "B. 将线性输出转换为概率",
                "C. 加速模型训练"
            ],
            "correct": "B",
            "explanation": "Sigmoid函数能将任意实数映射到(0,1)区间，适合表示概率。"
        },
        {
            "question": "3. 逻辑回归为什么使用交叉熵损失?",
            "options": [
                "A. 交叉熵损失计算更简单",
                "B. 交叉熵损失是凸函数，更容易优化",
                "C. 没有特别原因，只是传统习惯"
            ],
            "correct": "B",
            "explanation": "对于逻辑回归，交叉熵损失是凸函数，存在唯一最小值，而均方误差是non-convex的。"
        },
        {
            "question": "4. 分类阈值如何影响模型性能?",
            "options": [
                "A. 阈值不影响模型性能",
                "B. 高阈值会提高精确率但降低召回率",
                "C. 高阈值会同时提高精确率和召回率"
            ],
            "correct": "B",
            "explanation": "高阈值意味着更严格的正类判断标准，减少误报但可能增加漏报。"
        },
        {
            "question": "5. 逻辑回归可以处理非线性问题吗?",
            "options": [
                "A. 不能，逻辑回归只能处理线性可分问题",
                "B. 可以，通过特征工程引入非线性特征",
                "C. 可以，逻辑回归本身是非线性模型"
            ],
            "correct": "B",
            "explanation": "逻辑回归的决策边界本身是线性的，但通过添加多项式特征等方式，可以处理非线性问题。"
        }
    ]
    
    # 初始化会话状态存储用户答案
    st.session_state.user_answers = [None] * len(quiz_data)
    
    # 显示所有题目和选项（初始无选中状态）
    for i, item in enumerate(quiz_data):
        st.markdown(f"**{item['question']}**")
        # 设置默认值为None实现初始无选中状态，通过会话状态保存答案
        answer = st.radio(
            "选择答案:",
            item["options"],
            key=f"quiz_{i}",
            index=None,  # 关键：初始无选中项
            label_visibility="collapsed"
        )
        
        # 更新会话状态中的答案（提取选项字母A/B/C）
        if answer is not None:
            st.session_state.user_answers[i] = answer[0]
        
    # 检查是否所有题目都已作答
    all_answered = all(ans is not None for ans in st.session_state.user_answers)
    
    # 提交按钮：只有全部答完才可用
    submit_btn = st.button(
        "提交答案", 
        key="submit_quiz",
        disabled=not all_answered  # 未答完时禁用
    )
    
    # 未答完时显示提示
    if not all_answered:
        st.info("请完成所有5道题目后再提交")
    
    # 处理提交
    if submit_btn and all_answered:
        # 计算得分和错误题目
        score = 0
        results = []
        incorrect_questions = []
        for i, item in enumerate(quiz_data):
            is_correct = st.session_state.user_answers[i] == item["correct"]
            if is_correct:
                score += 20  # 每题20分
            else:
                incorrect_questions.append({"topic": item["question"], "user_answer": st.session_state.user_answers[i]})
            results.append({
                "question": item["question"],
                "user_answer": st.session_state.user_answers[i],
                "correct_answer": item["correct"],
                "is_correct": is_correct,
                "explanation": item["explanation"]
            })


        # 记录测验结果（新增：用于评价分析）
        st.session_state.logistic_records["logistic_quiz"] = {
            "score": score,
            "incorrect_questions": incorrect_questions,
            "timestamp": datetime.now().timestamp()
        }
        
        # 显示得分
        st.success(f"📊 测验完成！你的得分是：{score}分")
        st.write("### 答案解析：")
        
        # 显示每题结果（修正后）
        for res in results:
            # 使用emoji和文字标记正确/错误状态
            if res["is_correct"]:
                status_text = "✅ 正确"
            else:
                status_text = "❌ 错误"
            
            with st.expander(f"{res['question']} {status_text}"):
                if res["is_correct"]:
                    st.success(f"你的答案：{res['user_answer']}（正确）")
                else:
                    st.error(f"你的答案：{res['user_answer']}（错误）")
                    st.info(f"正确答案：{res['correct_answer']}")
                st.write(f"解析：{res['explanation']}")
        
        # 准备AI分析的输入
        incorrect_topics = [
            res["question"] for res in results if not res["is_correct"]
        ]
        
        analysis_prompt = f"""
        以下是学生在线性回归测验中的答题情况：
        - 总得分：{score}分
        - 错误题目：{len(incorrect_topics)}道
        - 错误知识点：{'; '.join(incorrect_topics) if incorrect_topics else '无'}
        
        请分析该学生的知识掌握情况，指出未掌握的核心概念，并给出具体的学习建议和指导方向，帮助学生针对性提升。
        答案必须控制在450字以内
        """
        
        # 调用AI分析
        with st.spinner("AI正在分析你的答题情况..."):
            ai_analysis = ask_ai_assistant(analysis_prompt, "线性回归测验分析")
        
        # 显示AI分析结果
        st.write("### 🤖 AI学习诊断：")
        st.info(ai_analysis)
    
    return "概念测验模块：完成5题单选题测试"

# 实际应用案例模块
def real_world_example_section():
    st.header("🌍 逻辑回归实际应用案例")
    
    example = st.selectbox(
        "选择实际应用案例:",
        ["信用卡欺诈检测", "客户流失预测", "疾病风险预测", "上传自己的数据"]
    )
    
    if example == "上传自己的数据":
        uploaded_file = st.file_uploader("上传CSV文件", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("数据预览:", data.head())
            
            # 检查是否有分类变量
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.warning("检测到分类变量，本演示将自动忽略这些列。")
                data = data.select_dtypes(exclude=['object'])
            
            # 选择目标列
            if len(data.columns) < 2:
                st.error("数据至少需要包含一个特征列和一个目标列!")
                return
            
            target_col = st.selectbox("选择目标列(应包含0和1)", data.columns)
            
            # 检查目标列是否为二分类
            unique_vals = data[target_col].unique()
            if len(unique_vals) != 2 or not set(unique_vals).issubset({0, 1}):
                st.error("目标列必须是二分类(只包含0和1)!")
                return
            
            # 选择特征列
            feature_cols = [col for col in data.columns if col != target_col]
            if not feature_cols:
                st.error("没有可用的特征列!")
                return
            
            X = data[feature_cols].values
            y = data[target_col].values
            
            analyze_custom_data(X, y, feature_cols, target_col)
            return f"实际应用模块: 上传自定义数据, 目标列={target_col}"
    else:
        # 生成示例数据
        X, y, description = load_example_dataset(example)
        st.write(description)
        
        analyze_custom_data(X, y, ["特征1", "特征2", "特征3"], "目标变量")
        return f"实际应用模块: 使用{example}数据集"

# 加载示例数据集
def load_example_dataset(example_name):
    np.random.seed(42)
    
    if example_name == "信用卡欺诈检测":
        # 生成欺诈检测数据：大多数是正常交易，少数是欺诈
        n_samples = 500
        n_fraud = int(n_samples * 0.1)  # 10%欺诈率
        
        # 正常交易特征
        normal_amount = np.random.normal(500, 300, n_samples - n_fraud)
        normal_time = np.random.normal(12, 6, n_samples - n_fraud)
        normal_freq = np.random.normal(2, 1, n_samples - n_fraud)
        
        # 欺诈交易特征（金额更大，时间更晚，频率更低）
        fraud_amount = np.random.normal(2000, 800, n_fraud)
        fraud_time = np.random.normal(20, 4, n_fraud)
        fraud_freq = np.random.normal(0.5, 0.3, n_fraud)
        
        # 合并数据
        X = np.vstack([
            np.column_stack((normal_amount, normal_time, normal_freq)),
            np.column_stack((fraud_amount, fraud_time, fraud_freq))
        ])
        y = np.hstack([np.zeros(n_samples - n_fraud), np.ones(n_fraud)])
        
        # 打乱顺序
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        description = "信用卡欺诈检测数据: 包含交易金额、时间和频率特征，预测交易是否为欺诈(1=欺诈)"
        return X, y, description
    
    elif example_name == "客户流失预测":
        # 生成客户流失数据
        n_samples = 500
        
        # 特征：使用时长(月)、月消费、客服联系次数
        tenure = np.random.normal(30, 20, n_samples)
        monthly_charge = np.random.normal(50, 30, n_samples)
        support_calls = np.random.randint(0, 10, n_samples)
        
        X = np.column_stack((tenure, monthly_charge, support_calls))
        
        # 流失概率：使用时长越短、月消费越高、客服联系越多，流失概率越大
        z = -0.05*tenure + 0.03*monthly_charge + 0.3*support_calls - 2
        prob = sigmoid(z)
        y = np.random.binomial(1, prob)
        
        description = "客户流失预测数据: 包含使用时长、月消费和客服联系次数，预测客户是否会流失(1=流失)"
        return X, y, description
    
    elif example_name == "疾病风险预测":
        # 生成疾病风险预测数据
        n_samples = 500
        
        # 特征：年龄、BMI、血压
        age = np.random.normal(50, 15, n_samples)
        bmi = np.random.normal(25, 5, n_samples)
        blood_pressure = np.random.normal(120, 15, n_samples)
        
        X = np.column_stack((age, bmi, blood_pressure))
        
        # 患病概率：年龄越大、BMI越高、血压越高，患病概率越大
        z = 0.04*age + 0.1*bmi + 0.03*blood_pressure - 10
        prob = sigmoid(z)
        y = np.random.binomial(1, prob)
        
        description = "疾病风险预测数据: 包含年龄、BMI和血压，预测患病风险(1=患病)"
        return X, y, description
    
    return None, None, ""

# 分析自定义数据（不使用标准化）
def analyze_custom_data(X, y, feature_names, target_name):
    if len(X) != len(y):
        st.error("特征和目标的长度不匹配!")
        return
    
    if len(X) < 10:
        st.error("数据点太少，至少需要10个样本!")
        return
    
    # 训练模型（不使用标准化）
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # 预测概率
    y_prob = model.predict_proba(X)[:, 1]
    
    # 评估模型
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    
    st.subheader("模型性能")
    st.text(report)
    
    # 显示系数
    st.subheader("特征重要性（系数）")
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': model.coef_[0]
    }).sort_values('系数', ascending=False)
    st.dataframe(coef_df)
    
    st.info("""
    **系数解释:**
    - 正系数: 该特征值越大，属于正类的概率越高
    - 负系数: 该特征值越大，属于正类的概率越低
    - 系数绝对值越大，特征对预测的影响越大
    
    注意：系数大小受特征尺度影响，这里使用的是原始数据，未进行标准化处理。
    """)
    
    # 如果是二维数据，绘制决策边界
    if X.shape[1] == 2:
        fig = plot_decision_boundary(X, y, model.coef_[0], model.intercept_[0])
        st.pyplot(fig)

        
# 主程序
def main():

    # 初始化会话状态
    if 'section' not in st.session_state:
        st.session_state.section = "数据生成与探索"

    # 初始化逻辑回归专属的学习记录（与线性回归区分开）
    if "logistic_records" not in st.session_state:
        st.session_state.logistic_records = {
            "data_generation": [],  # 数据生成模块记录
            "sig_function":[],  #sigmoid函数交互
            "module_sequence": [],
            "module_timestamps": {},
            "para_tuning": [],  # 参数手动调整
            "gradient_descent": [],  # 梯度下降模块记录
            "model_evaluation": [],  # 模型评估模块记录
            "logistic_quiz": {},  # 逻辑回归专属测验
            "ai_interactions": []
        }
    # 记录模块访问顺序（进入模块时触发）AI
    current_section = st.session_state.section
    st.session_state.logistic_records["module_sequence"].append(current_section)
    if current_section not in st.session_state.logistic_records["module_timestamps"]:
        st.session_state.logistic_records["module_timestamps"][current_section] = {
            "enter_time": time.time()
        }       

    
    st.sidebar.title("导航菜单")
    section = st.sidebar.radio("选择学习模块", [
        "数据生成与探索",
        "Sigmoid函数交互演示",
        "参数手动调整",
        "梯度下降可视化",
        "模型评估",
        "概念测验",
        "实际应用案例",
        "编程实例（乳腺癌数据集）" 
    ])
  
    # 更新会话状态
    st.session_state.section = section
    
    context = ""
    if section == "数据生成与探索":
        context = data_generation_section()
    elif section == "Sigmoid函数交互演示":
        context = sigmoid_interactive_section()
    elif section == "参数手动调整":
        context = manual_tuning_section()
    elif section == "梯度下降可视化":
        context = gradient_descent_section()
    elif section == "模型评估":
        context = model_evaluation_section()
    elif section == "概念测验":
        context = quiz_section()
    elif section == "实际应用案例":
        context = real_world_example_section()
    elif section == "编程实例（乳腺癌数据集）":
        # 初始化step变量（如果不存在）
        if 'step' not in st.session_state:
            st.session_state.step = 0
        logistic_regression_step_by_step.main()
        context = "编程实例模块: 乳腺癌数据集逻辑回归分步练习"
    
    display_chat_interface(context)

    # 记录模块退出时间（新增：用于计算停留时间）
    if current_section in st.session_state.logistic_records["module_timestamps"]:
        st.session_state.logistic_records["module_timestamps"][current_section]["exit_time"] = datetime.now().timestamp()

    if section != "编程实例（乳腺癌数据集）":
        # 侧边栏添加学习报告按钮（调用独立模块）
        st.sidebar.markdown("---")
        if st.sidebar.button("逻辑回归模块学习报告"):
            # 传入模块类型、原始记录、AI调用函数
            report = generate_evaluation(
                module_type="logistic_regression",
                raw_records=st.session_state.logistic_records
            )
            st.write("### 逻辑回归学习情况报告")
            st.info(report)
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **逻辑回归交互式学习平台**
    
    设计用于机器学习教学，帮助学生理解:
    - 逻辑回归基本原理
    - Sigmoid函数的作用与特性
    - 分类阈值的选择策略
    - 模型评估指标与解释
    """)


if __name__ == "__main__":
    main()
